import numpy as np
import openmm as mm
from openmm import app, unit, Platform
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import argparse
import sys
import os
import time
import csv
import logging
import cupy as cp
from cupyx.scipy.sparse import coo_matrix as coo_gpu


class AnsiColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[90m',            # Grey
        'INFO': '\033[0m',              # Reset
        'WARNING': '\033[33m',          # Yellow
        'ERROR': '\033[31m',            # Red
        'CRITICAL': '\033[91m\033[1m',  # Bright Red + Bold
    }
    RESET_COLOR = '\033[0m'

    def format(self, record):
        """
        Format a log record, wrapping the message in ANSI colour codes for
        its severity level and resetting colour at the end.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            Colour-coded, formatted log message string.
        """
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET_COLOR)
        return f"{color}{log_message}{self.RESET_COLOR}"

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler with color formatting
console_handler = logging.StreamHandler()
console_formatter = AnsiColorFormatter("..:ENM> {levelname}: {message}", style="{")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for enm.out
file_handler = logging.FileHandler('enm.out')
file_formatter = logging.Formatter("{asctime} ..:ENM> {levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def parse_arguments():
    """
    Parse command-line arguments for the ENM normal mode analysis pipeline.

    Reads sys.argv and populates an argument namespace with all options
    required to configure the ENM system, diagonalisation, and output
    generation.  Default values are applied where arguments are omitted.

    Returns
    -------
    args : argparse.Namespace
        Parsed argument namespace with the following attributes:

        input : str
            Path to the input PDB file (required).
        output : str
            Name of the output folder (default: 'output').
        type : str
            ENM model type; 'ca' for Cα-only or 'heavy' for all heavy
            atoms (default: 'ca').
        cutoff : float or None
            Interaction cutoff in Å.  If None, defaults to 15.0 Å for
            'ca' and 12.0 Å for 'heavy'.
        spring_constant : float
            Harmonic spring constant in kcal mol⁻¹ Å⁻² (default: 1.0).
        max_modes : int or None
            Number of non-rigid modes to compute.  If None, the full
            spectrum is computed.
        output_modes : int
            Number of modes to save to disk and analyse (default: 10).
        no_nm_vec : bool
            False when --no_nm_vec is supplied, disabling XYZ
            eigenvector output; True otherwise.
        no_nm_trj : bool
            False when --no_nm_trj is supplied, disabling PDB
            trajectory output; True otherwise.
        no_collectivity : bool
            False when --no_collectivity is supplied; True
            otherwise.
        no_contributions : bool
            False when --no_contributions is supplied; True
            otherwise.
        no_rmsf : bool
            False when --no_rmsf is supplied; True otherwise.
        no_dccm : bool
            False when --no_dccm is supplied; True otherwise.
        no_gpu : bool
            False when --no_gpu is supplied, disabling GPU
            acceleration; True otherwise.

    Raises
    ------
    SystemExit
        If required arguments are missing or an invalid choice is supplied
        (handled internally by argparse).
    """
    parser = argparse.ArgumentParser(description='Elastic Network Model Normal Mode Analysis')

    # Required arguments
    parser.add_argument('-i', '--input', required=True, help='Input PDB file')

    # Optional arguments with defaults
    parser.add_argument('-o', '--output', default='output', help='Output folder name')

    parser.add_argument('-t', '--type', choices=['ca', 'heavy'], default='ca',
                       help='Model type: ca (Cα-only) or heavy (heavy atoms). Default: ca')
    parser.add_argument('-c', '--cutoff', type=float, default=None,
                       help='Cutoff distance for interactions in Å. Default: 15.0 for CA, 12.0 for heavy atoms')
    parser.add_argument('-k', '--spring_constant', type=float, default=1.0,
                       help='Spring constant for ENM bonds in kcal/mol/Å². Default: 1.0')
    parser.add_argument('-m', '--max_modes', type=int, default=None,
                       help='Number of non-rigid modes to compute. Default: all modes')
    parser.add_argument('-n', '--output_modes', type=int, default=10,
                       help='Number of modes to save and analyze. Default: 10 modes')

    # Boolean flags to enable/disable features
    parser.add_argument('--no_nm_vec', action='store_false',
                       help='Disable writing mode vectors')
    parser.add_argument('--no_nm_trj', action='store_false',
                       help='Disable writing mode trajectories')
    parser.add_argument('--no_collectivity', action='store_false',
                       help='Disable collectivity calculation')
    parser.add_argument('--no_contributions', action='store_false',
                       help='Disable mode contributions plot')
    parser.add_argument('--no_rmsf', action='store_false',
                       help='Disable RMSF plot')
    parser.add_argument('--no_dccm', action='store_false',
                       help='Disable DCCM plot')
    parser.add_argument('--no_gpu', action='store_false',
                       help='Disable GPU acceleration')

    return parser.parse_args()

def create_system(pdb_file, model_type='ca', cutoff=None, spring_constant=1.0, output_prefix="input"):
    """
    Create an Elastic Network Model system based on the specified model type.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file
    model_type : str, optional
        Type of model to create: 'ca' for Cα-only or 'heavy' for heavy-atom ENM
    cutoff : float, optional
        Cutoff distance for interactions in Å. If None, uses default values:
        15.0Å for CA model, 12.0Å for heavy-atom model
    spring_constant : float, optional
        Spring constant for the ENM bonds in kcal/mol/Å²
    output_prefix : str, optional
        Prefix for output files

    Returns
    -------
    system : openmm.System
        The created OpenMM system
    topology : openmm.app.Topology
        The topology of the system
    positions : openmm.unit.Quantity
        The positions of particles in the system

    Raises
    ------
    ValueError
        If an invalid model type is specified or no relevant atoms are found
    """
    # Set default cutoffs if not provided
    if cutoff is None:
        cutoff = 15.0 if model_type == 'ca' else 12.0

    if model_type == 'ca':
        return _create_ca_system(pdb_file, cutoff, spring_constant, output_prefix)
    elif model_type == 'heavy':
        return _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix)
    else:
        raise ValueError("Invalid model type. Choose 'ca' or 'heavy'")

def _create_ca_system(pdb_file, cutoff, spring_constant, output_prefix):
    """
    Create a Cα-only ENM system from a PDB file.

    Extracts Cα atoms, assigns uniform carbon masses (12.011 Da), and
    connects pairs within [2.9 Å, cutoff] with a harmonic spring.  The
    reduced structure is saved to {output_prefix}_ca_structure.pdb.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    cutoff : float
        Maximum Cα–Cα distance in Å for ENM bond formation.
    spring_constant : float
        Harmonic spring constant in kcal mol⁻¹ Å⁻².
    output_prefix : str
        Prefix used when writing the Cα PDB output file.

    Returns
    -------
    system : openmm.System
        System with one particle per Cα atom and a CustomBondForce
        encoding the ENM potential.
    topology : openmm.app.Topology
        Reduced topology containing only Cα atoms.
    positions : openmm.unit.Quantity
        Cα positions in nanometres.

    Raises
    ------
    ValueError
        If no Cα atoms are found in the PDB file.
    """
    logger.info("Creating Cα-only system using Elastic Network Model...")
    pdb = app.PDBFile(pdb_file)

    # Extract Cα atoms and their positions
    ca_info = []
    positions_list = []
    for atom in pdb.topology.atoms():
        if atom.name == 'CA':
            pos = pdb.positions[atom.index]
            ca_info.append((atom.index, atom.residue))
            positions_list.append([pos.x, pos.y, pos.z])

    if not ca_info:
        raise ValueError("No Cα atoms found in the structure")

    n_atoms = len(ca_info)
    logger.info(f"Selected {n_atoms} Cα atoms")

    # Create a simplified topology with only Cα atoms
    new_topology = app.Topology()
    new_chain = new_topology.addChain()
    residue_map = {}

    for i, (orig_idx, residue) in enumerate(ca_info):
        if residue not in residue_map:
            new_res = new_topology.addResidue(f"{residue.name}{residue.id}", new_chain)
            residue_map[residue] = new_res
        new_topology.addAtom("CA", app.element.carbon, residue_map[residue])

    # Create the system and add particles
    system = mm.System()
    positions = [mm.Vec3(*pos) * unit.nanometer for pos in positions_list]
    positions_quantity = unit.Quantity(positions)

    carbon_mass = 12.011 * unit.daltons
    for _ in range(n_atoms):
        system.addParticle(carbon_mass)

    # Create ENM force field
    enm_force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
    enm_force.addGlobalParameter("k", spring_constant)
    enm_force.addPerBondParameter("r0")

    # Calculate distance matrix and create bonds
    pos_np = np.array(positions_list, dtype=np.float32)
    dist_matrix = squareform(pdist(pos_np))
    cutoff_nm = cutoff * 0.1    # Convert Å to nm
    min_distance_nm = 2.9 * 0.1 # Minimum distance in nm (2.9 Å)

    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = dist_matrix[i, j]
            # Apply both minimum distance and cutoff
            if min_distance_nm <= dist <= cutoff_nm:
                bonds.append((i, j, dist))

    for i, j, dist in bonds:
        enm_force.addBond(i, j, [dist])

    system.addForce(enm_force)
    logger.info(f"Added {len(bonds)} ENM bonds with cutoff={cutoff}Å, min_distance=2.9Å, k={spring_constant} kcal/mol/Å²")
    system.addForce(mm.CMMotionRemover())

    # Save the Cα structure
    ca_pdb_file = f"{output_prefix}_ca_structure.pdb"
    with open(ca_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    logger.info(f"C-alpha structure saved to {ca_pdb_file}\n")

    # Convert HETATM to ATOM
    convert_hetatm_to_atom(ca_pdb_file)

    return system, new_topology, positions_quantity

def _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix):
    """
    Create a heavy-atom ENM system from a PDB file.

    Extracts all non-hydrogen atoms, assigns element-specific masses, and
    connects pairs within [2.0 Å, cutoff] with a harmonic spring.  The
    reduced structure is saved to {output_prefix}_heavy.pdb.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    cutoff : float
        Maximum inter-atom distance in Å for ENM bond formation.
    spring_constant : float
        Harmonic spring constant in kcal mol⁻¹ Å⁻².
    output_prefix : str
        Prefix used when writing the heavy-atom PDB output file.

    Returns
    -------
    system : openmm.System
        System with one particle per heavy atom and a CustomBondForce
        encoding the ENM potential.
    topology : openmm.app.Topology
        Reduced topology containing only heavy atoms.
    positions : openmm.unit.Quantity
        Heavy-atom positions in nanometres.

    Raises
    ------
    ValueError
        If no heavy atoms are found in the PDB file.
    """
    logger.info("Creating heavy-atom system using Elastic Network Model...")
    pdb = app.PDBFile(pdb_file)

    # Identify heavy atoms (non-hydrogen)
    heavy_atoms = []
    positions_list = []
    for atom in pdb.topology.atoms():
        if atom.element != app.element.hydrogen:
            pos = pdb.positions[atom.index]
            heavy_atoms.append((atom.index, atom.residue, atom.name, atom.element))
            positions_list.append([pos.x, pos.y, pos.z])

    if not heavy_atoms:
        raise ValueError("No heavy atoms found in the structure")

    n_atoms = len(heavy_atoms)
    logger.info(f"Selected {n_atoms} heavy atoms")

    # Create new topology with only heavy atoms
    new_topology = app.Topology()
    new_chain = new_topology.addChain()
    residue_map = {}

    for i, (orig_idx, residue, name, element) in enumerate(heavy_atoms):
        if residue not in residue_map:
            new_res = new_topology.addResidue(f"{residue.name}{residue.id}", new_chain)
            residue_map[residue] = new_res
        new_topology.addAtom(name, element, residue_map[residue])

    # Create the system and add particles with appropriate masses
    system = mm.System()
    positions = [mm.Vec3(*pos) * unit.nanometer for pos in positions_list]
    positions_quantity = unit.Quantity(positions)

    for _, _, _, element in heavy_atoms:
        system.addParticle(element.mass)

    # Create ENM force field
    enm_force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
    enm_force.addGlobalParameter("k", spring_constant)
    enm_force.addPerBondParameter("r0")

    # Calculate distance matrix and create bonds
    pos_np = np.array(positions_list, dtype=np.float32)
    dist_matrix = squareform(pdist(pos_np))
    cutoff_nm = cutoff * 0.1    # Convert Å to nm
    min_distance_nm = 2.0 * 0.1 # Minimum distance in nm (2.0 Å)

    # Add bonds within cutoff range
    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = dist_matrix[i, j]
            # Apply both minimum distance and cutoff
            if min_distance_nm <= dist <= cutoff_nm:
                bonds.append((i, j, dist))

    for i, j, dist in bonds:
        enm_force.addBond(i, j, [dist])

    system.addForce(enm_force)
    logger.info(f"Added {len(bonds)} ENM bonds with cutoff={cutoff}Å, min_distance=2.0Å, k={spring_constant} kcal/mol/Å²")
    system.addForce(mm.CMMotionRemover())

    # Save heavy atom structure
    heavy_pdb_file = f"{output_prefix}_heavy.pdb"
    with open(heavy_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    logger.info(f"Heavy-atom structure saved to {heavy_pdb_file}\n")

    return system, new_topology, positions_quantity

def _extract_bonds_array(enm_force):
    """
    Extract bond atom-index pairs from an OpenMM CustomBondForce into a
    compact integer array.

    Parameters
    ----------
    enm_force : openmm.CustomBondForce
        The ENM bond force to read connectivity from.

    Returns
    -------
    ij : ndarray, shape (n_bonds, 2), dtype intp
        Zero-based atom index pairs [i, j] for each bond.
    """
    nb = enm_force.getNumBonds()
    ij = np.empty((nb, 2), dtype=np.intp)
    for b in range(nb):
        i, j, _ = enm_force.getBondParameters(b)
        ij[b, 0] = i
        ij[b, 1] = j
    return ij

def _build_hessian_cpu(pos, ij, k, n):
    """
    Assemble the 3N × 3N ENM Hessian on the CPU.

    For each bond (p, q) with unit vector e, four 3×3 blocks are updated:
    the two diagonal blocks gain k · e⊗e and the two off-diagonal blocks
    lose it.

    Parameters
    ----------
    pos : ndarray, shape (N, 3), float64
        Atom positions in nanometres.
    ij : ndarray, shape (n_bonds, 2), dtype intp
        Zero-based atom index pairs for each bond.
    k : float
        Harmonic spring constant in kcal mol⁻¹ nm⁻².
    n : int
        Total number of particles N.

    Returns
    -------
    H : ndarray, shape (3N, 3N), float64
        Dense Hessian matrix.
    """
    from scipy.sparse import coo_matrix

    p_idx = ij[:, 0]                                    # (n_bonds,)
    q_idx = ij[:, 1]

    r   = pos[q_idx] - pos[p_idx]                       # (n_bonds, 3)
    d   = np.linalg.norm(r, axis=1, keepdims=True)      # (n_bonds, 1)
    e   = r / d                                         # unit vectors
    B   = k * np.einsum('bi,bj->bij', e, e)             # (n_bonds, 3, 3)

    # Build the 9 (row-offset, col-offset) pairs for a 3×3 block once
    ai, ci = np.divmod(np.arange(9, dtype=np.intp), 3)  # offsets 0..2

    # Global row/col indices for each of the 4 block contributions
    rp = (3 * p_idx[:, None] + ai).ravel()
    cp_ = (3 * p_idx[:, None] + ci).ravel()
    rq = (3 * q_idx[:, None] + ai).ravel()
    cq = (3 * q_idx[:, None] + ci).ravel()
    v  = B.reshape(-1, 9).ravel()                       # (n_bonds*9,)

    rows = np.concatenate([rp, rq, rp, rq])
    cols = np.concatenate([cp_, cq, cq, cp_])
    data = np.concatenate([ v,  v, -v, -v])

    n3 = 3 * n
    return coo_matrix((data, (rows, cols)), shape=(n3, n3),
                      dtype=np.float64).toarray()

def _build_hessian_gpu(pos, ij, k, n):
    """
    Assemble the 3N × 3N ENM Hessian on the GPU

    Parameters
    ----------
    pos : ndarray, shape (N, 3), float64
        Atom positions in nanometres (transferred to GPU internally).
    ij : ndarray, shape (n_bonds, 2), dtype intp
        Zero-based atom index pairs for each bond (transferred to GPU
        internally).
    k : float
        Harmonic spring constant in kcal mol⁻¹ nm⁻².
    n : int
        Total number of particles N.

    Returns
    -------
    H : ndarray, shape (3N, 3N), float64
        Dense Hessian matrix on the CPU.  Computed on the GPU if available,
        otherwise falls back to the CPU result.
    """
    try:
        p_gpu  = cp.asarray(ij[:, 0], dtype=cp.intp)
        q_gpu  = cp.asarray(ij[:, 1], dtype=cp.intp)
        pos_gpu = cp.asarray(pos, dtype=cp.float64)

        r   = pos_gpu[q_gpu] - pos_gpu[p_gpu]
        d   = cp.linalg.norm(r, axis=1, keepdims=True)
        e   = r / d
        B   = k * cp.einsum('bi,bj->bij', e, e)

        ai, ci = cp.divmod(cp.arange(9, dtype=cp.intp), 3)
        rp  = (3 * p_gpu[:, None] + ai).ravel()
        cp_ = (3 * p_gpu[:, None] + ci).ravel()
        rq  = (3 * q_gpu[:, None] + ai).ravel()
        cq  = (3 * q_gpu[:, None] + ci).ravel()
        v   = B.reshape(-1, 9).ravel()

        rows = cp.concatenate([rp, rq, rp, rq])
        cols = cp.concatenate([cp_, cq, cq, cp_])
        data = cp.concatenate([v, v, -v, -v])

        n3    = 3 * n
        H_gpu = coo_gpu((data, (rows, cols)), shape=(n3, n3),
                         dtype=cp.float64).toarray()
        H = cp.asnumpy(H_gpu)
        del H_gpu, pos_gpu, B, e, r
        cp.get_default_memory_pool().free_all_blocks()
        return H

    except Exception as exc:
        logger.warning(f"GPU Hessian failed ({exc}); falling back to CPU.")
        return _build_hessian_cpu(pos, ij, k, n)

def hessian_enm(system, positions, use_gpu=False):
    """
    Build the analytical ENM Hessian matrix (3N × 3N, float64).

    Reads the spring constant and bond list from the system's CustomBondForce,
    then assembles the second-derivative matrix. The result is symmetrised and
    given a minimal diagonal regularisation (1 × 10^-10) to stabilise the six
    rigid-body zero modes.

    Parameters
    ----------
    system : openmm.System
        System containing a CustomBondForce ENM potential.  The first
        global parameter is taken as the spring constant k.
    positions : openmm.unit.Quantity
        Reference atomic positions in any OpenMM length unit.
    use_gpu : bool, optional
        If True, attempt GPU assembly via CuPy; falls back to CPU on
        any error (default: False).

    Returns
    -------
    hessian : ndarray, shape (3N, 3N), float64
        Symmetrised and regularised Hessian matrix.

    Raises
    ------
    ValueError
        If no CustomBondForce is found in the system.
    """
    n = system.getNumParticles()

    enm_force = next((f for f in system.getForces()
                      if isinstance(f, mm.CustomBondForce)), None)
    if enm_force is None:
        raise ValueError("No CustomBondForce (ENM) found in system.")

    k   = enm_force.getGlobalParameterDefaultValue(0)
    ij  = _extract_bonds_array(enm_force)   # (n_bonds, 2)

    # Float64 positions in nm
    pos = np.array([[p.x, p.y, p.z]
                    for p in positions.value_in_unit(unit.nanometer)],
                   dtype=np.float64)

    logger.info(f"Computing Hessian for {n} particles "
                f"({ij.shape[0]} bonds, {'GPU' if use_gpu else 'CPU'})...")
    t0 = time.time()

    builder = _build_hessian_gpu if use_gpu else _build_hessian_cpu
    H = builder(pos, ij, k, n)

    # Enforce exact symmetry (removes sub-ULP asymmetry from floating-point ops)
    H = 0.5 * (H + H.T)
    # Minimal regularisation — shifts zero modes by ~1e-10, physical modes unaffected
    np.fill_diagonal(H, H.diagonal() + 1e-10)

    logger.info(f"ENM Hessian computed in {time.time() - t0:.2f} s\n")
    return H

def mass_weight_hessian(hessian, system):
    """
    Apply mass-weighting to the Hessian to produce the dynamical matrix
    M^-1/2 * H * M^-1/2.

    Parameters
    ----------
    hessian : ndarray, shape (3N, 3N), float64
        Raw unweighted Hessian matrix.
    system : openmm.System
        Provides per-particle masses.

    Returns
    -------
    mw_hessian : ndarray, shape (3N, 3N), float64
        Mass-weighted dynamical matrix whose eigenvalues are the squared
        angular frequencies ω².
    """
    n = system.getNumParticles()
    masses = np.fromiter(
        (system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)),
        dtype=np.float64, count=n
    )
    masses = np.where(masses > 0.0, masses, 1.0)    # guard zero-mass virtual sites
    M = np.repeat(1.0 / np.sqrt(masses), 3)         # (3N,) inverse sqrt mass vector
    return M[:, None] * hessian * M[None, :]        # O(9N²) element-wise, no alloc

def _diagonalize_cpu(H, n_modes):
    """
    Diagonalise a real symmetric matrix using scipy.linalg.eigh.

    Parameters
    ----------
    H : ndarray, shape (3N, 3N), float64
        Real symmetric matrix to diagonalise. Overwritten in place.
    n_modes : int or None
        Number of non-rigid modes requested.  When not None, computes
        the lowest n_modes + 6 eigenpairs; when None, computes the
        full spectrum.

    Returns
    -------
    eigenvalues : ndarray, shape (K,), float64
        Eigenvalues in ascending order.
    eigenvectors : ndarray, shape (3N, K), float64
        Corresponding normalised eigenvectors as columns.
    """
    n = H.shape[0]
    if n_modes is not None:
        # Request the first (n_modes + 6) eigenvalues to cover rigid-body modes
        end = min(n_modes + 5, n - 1)     # subset_by_index is inclusive, 0-based
        return eigh(H, subset_by_index=[0, end],
                    driver='evr', overwrite_a=True, check_finite=False)
    else:
        return eigh(H, driver='evd', overwrite_a=True, check_finite=False)

def _diagonalize_gpu(H, n_modes):
    """
    Diagonalise a real symmetric matrix on the GPU.

    Parameters
    ----------
    H : ndarray, shape (3N, 3N), float64
        Real symmetric matrix to diagonalise (transferred to GPU internally).
    n_modes : int or None
        Number of non-rigid modes requested.  When not None, retains
        only the lowest n_modes + 6 eigenpairs before CPU transfer.

    Returns
    -------
    eigenvalues : ndarray, shape (K,), float64
        Eigenvalues in ascending order, on the CPU.
    eigenvectors : ndarray, shape (3N, K), float64
        Corresponding normalised eigenvectors as columns, on the CPU.

    Raises
    ------
    cupy.cuda.runtime.CUDARuntimeError
        On CUDA failure; callers should catch and fall back to
        _diagonalize_cpu.
    ImportError
        If CuPy is not installed.
    """
    import cupy as cp
    pool = cp.get_default_memory_pool()

    with cp.cuda.Device(0):
        H_gpu = cp.asarray(H, dtype=cp.float64)
        w_gpu, v_gpu = cp.linalg.eigh(H_gpu, UPLO='L')
        del H_gpu
        pool.free_all_blocks()

        if n_modes is not None:
            keep = min(n_modes + 6, w_gpu.shape[0])
            w_gpu = w_gpu[:keep]
            v_gpu = v_gpu[:, :keep]

        w = cp.asnumpy(w_gpu)
        v = cp.asnumpy(v_gpu)
        del w_gpu, v_gpu
        pool.free_all_blocks()

    return w, v

def compute_normal_modes(hessian, n_modes=None, use_gpu=False):
    """
    Diagonalise the mass-weighted Hessian and return sorted normal modes.


    Parameters
    ----------
    hessian : ndarray, shape (3N, 3N), float64
        Mass-weighted Hessian (dynamical matrix).
    n_modes : int, optional
        Number of non-rigid modes to compute.  If None, the full spectrum
        is computed.
    use_gpu : bool, optional
        Attempt GPU diagonalisation via CuPy; falls back to CPU on any error
        (default: False).

    Returns
    -------
    frequencies : ndarray, shape (K,), float64
        Signed angular frequencies: ω_i = sign(λ_i) · √|λ_i|.  Negative
        values flag imaginary frequencies (structural instabilities).
    modes : ndarray, shape (3N, K), float64
        Normalised eigenvectors sorted by ascending eigenvalue.
    eigenvalues : ndarray, shape (K,), float64
        Raw eigenvalues sorted ascending.  eigenvalues[i] and
        modes[:, i] correspond to the same mode.
    """
    t0 = time.time()

    if use_gpu:
        logger.info("Diagonalising mass-weighted Hessian on GPU...")
        try:
            w, v = _diagonalize_gpu(hessian, n_modes)
        except Exception as exc:
            logger.warning(f"GPU diagonalisation failed ({exc}); falling back to CPU.")
            use_gpu = False

    if not use_gpu:
        logger.info("Diagonalising mass-weighted Hessian on CPU...")
        w, v = _diagonalize_cpu(hessian, n_modes)

    logger.info(f"Diagonalisation completed in {time.time() - t0:.2f} s\n")

    # eigh returns ascending order, but guarantee it after any potential GPU sort
    idx = np.argsort(w)
    w   = w[idx]
    v   = v[:, idx]

    # Signed frequency: ω = sign(λ)·√|λ|.  Negative → imaginary (instability).
    frequencies = np.sign(w) * np.sqrt(np.abs(w))

    # Warn if significant imaginary modes appear beyond the 6 rigid-body ones
    max_abs = np.abs(w).max() if w.size else 1.0
    if np.any(w[6:] < -1e-6 * max_abs):
        logger.warning(
            "Negative eigenvalues found beyond mode 6 — the structure may "
            "not be at a true energy minimum.  Check for steric clashes."
        )

    return frequencies, v, w

def convert_hetatm_to_atom(pdb_file):
    """
    Replace HETATM records with ATOM records in a PDB file in place.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to modify.  Overwritten in place.
    """
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('HETATM'):
            # Replace HETATM with ATOM while preserving spacing
            new_line = 'ATOM  ' + line[6:]
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(pdb_file, 'w') as f:
        f.writelines(new_lines)

def write_nm_vectors(modes, frequencies, system, topology, output_prefix, n_modes=10, start_mode=6):
    """
    Write normal mode eigenvectors to XYZ-format text files, one per mode.

    Parameters
    ----------
    modes : ndarray, shape (3N, M), float64
        Normal mode eigenvectors; column i is mode i (0-based).
    frequencies : ndarray, shape (M,), float64
        Signed angular frequencies in internal units; converted to cm⁻¹
        (factor 108.58) for the file header.
    system : openmm.System
        Used to query the total particle count.
    topology : openmm.app.Topology
        Used to retrieve the element symbol for each atom.
    output_prefix : str
        Directory and filename prefix for output XYZ files.
    n_modes : int, optional
        Number of modes to write (default: 10).  Clamped to available
        modes.
    start_mode : int, optional
        Zero-based index of the first mode to write (default: 6, the
        second non-rigid mode after the six rigid-body modes at 0–5).
    """
    n_particles = system.getNumParticles()

    # Ensure we don't exceed available modes
    n_modes = min(n_modes, modes.shape[1] - start_mode)
    if n_modes <= 0:
        logger.warning("No modes available to write.")
        return

    # Get element symbols from topology
    elements = []
    for atom in topology.atoms():
        elements.append(atom.element.symbol)

    # Write each mode to a separate XYZ file
    for mode_idx in range(start_mode, start_mode + n_modes):
        freq = frequencies[mode_idx] * 108.58  # Convert to cm⁻¹
        mode_number = mode_idx + 1
        output_file = f"{output_prefix}_mode_{mode_number}.xyz"

        with open(output_file, 'w') as f:
            # Write header
            f.write(f"{n_particles}\n")
            f.write(f"Normal Mode {mode_number}, Frequency: {freq:.2f} cm⁻¹\n")

            # Extract and reshape the mode vector
            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)

            # Write coordinates for each atom
            for i in range(n_particles):
                x, y, z = mode_vector[i]
                f.write(f"{elements[i]:2s} {x:14.10f} {y:14.10f} {z:14.10f}\n")

def write_nm_trajectories(topology, positions, modes, frequencies, output_prefix, system, model_type, n_modes=10, start_mode=6, amplitude=4, num_frames=34):
    """
    Write multi-frame PDB trajectories visualising atomic motion along normal
    modes.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the system, used for PDB record writing.
    positions : openmm.unit.Quantity
        Equilibrium atomic positions in any OpenMM length unit.
    modes : ndarray, shape (3N, M), float64
        Normal mode eigenvectors; column i is mode i (0-based).
    frequencies : ndarray, shape (M,), float64
        Signed angular frequencies in internal units; used only for log
        output (converted to cm⁻¹).
    output_prefix : str
        Directory and filename prefix for the output PDB files.
    system : openmm.System
        Used to obtain per-particle masses for mass-weighting displacements.
    model_type : str
        ENM model type ('ca' or 'heavy'); used for logging only.
    n_modes : int, optional
        Number of modes for which to write trajectories (default: 10).
        Clamped to available modes.
    start_mode : int, optional
        Zero-based index of the first mode to write (default: 6).
    amplitude : float, optional
        Peak displacement amplitude in Å (default: 4).
    num_frames : int, optional
        Total number of MODEL frames per trajectory (default: 34).
    """
    n_particles = system.getNumParticles()

    # Ensure we don't exceed available modes
    n_modes = min(n_modes, modes.shape[1] - start_mode)
    if n_modes <= 0:
        logger.warning("No modes available to write trajectories.")
        return

    for mode_idx in range(start_mode, start_mode + n_modes):
        freq = frequencies[mode_idx] * 108.58  # Convert to cm⁻¹
        mode_number = mode_idx + 1
        output_file = f"{output_prefix}_mode_{mode_number}_traj.pdb"

        # Mass-weight the mode vector
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])
        masses[masses == 0] = 1.0
        inv_sqrt_m = np.repeat(1 / np.sqrt(masses), 3)
        u = modes[:, mode_idx] * inv_sqrt_m

        rms = np.linalg.norm(u) / np.sqrt(n_particles)
        if rms < 1e-10:
            logger.warning(f"Skipping near-zero mode {mode_number}")
            continue

        # Scale displacement to the desired amplitude
        scaled_disp = u.reshape(n_particles, 3) * (amplitude / rms * 0.1)
        orig_pos = positions.value_in_unit(unit.nanometer)
        orig_pos_np = np.array([[p.x, p.y, p.z] for p in orig_pos])

        # Create a smooth oscillation trajectory
        seg1 = int(num_frames * 0.25)
        seg2 = int(num_frames * 0.25)
        seg3 = int(num_frames * 0.25)
        seg4 = num_frames - seg1 - seg2 - seg3

        displacements = np.zeros((num_frames, n_particles, 3))

        for frame in range(num_frames):
            if frame < seg1:
                factor = -frame / seg1
            elif frame < seg1 + seg2:
                factor = -1 + (frame - seg1) / seg2
            elif frame < seg1 + seg2 + seg3:
                factor = (frame - seg1 - seg2) / seg3
            else:
                factor = 1 - (frame - seg1 - seg2 - seg3) / seg4

            displacements[frame] = scaled_disp * factor

        # Write the trajectory to a PDB file
        with open(output_file, 'w') as f:
            for frame in range(num_frames):
                new_pos_np = orig_pos_np + displacements[frame]

                new_positions = []
                for i in range(n_particles):
                    x, y, z = new_pos_np[i]
                    new_positions.append(mm.Vec3(x, y, z))

                positions_quantity = unit.Quantity(new_positions, unit.nanometer)

                f.write(f"MODEL     {frame+1:5d}\n")
                app.PDBFile.writeFile(topology, positions_quantity, f, keepIds=True)
                f.write("ENDMDL\n")

        # Convert HETATM to ATOM in trajectory
        convert_hetatm_to_atom(output_file)

def compute_collectivity(mode_vector, n_atoms):
    """
    Compute the collectivity of a normal mode (Tama & Sanejouand, 2001).

    Parameters
    ----------
    mode_vector : ndarray, shape (3N,), float64
        Eigenvector of the mode.  Need not be pre-normalised.
    n_atoms : int
        Total number of atoms N.

    Returns
    -------
    collectivity : float
        Collectivity in (0, 1].  Returns 0.0 if the mode vector is zero.
    """
    u     = mode_vector.reshape(n_atoms, 3)
    norms = np.linalg.norm(u, axis=1)   # |Δr_i| for each atom
    p     = norms ** 2
    total = p.sum()
    if total < 1e-30:
        return 0.0
    p /= total  # normalise: Σ p_i = 1
    # Mask true zeros to avoid log(0); they contribute 0 to entropy anyway
    mask  = p > 0.0
    entropy = -np.dot(p[mask], np.log(p[mask]))
    return np.exp(entropy) / n_atoms

def write_collectivity(frequencies, modes, system, output_file, n_modes=20):
    """
    Write mode collectivities into a CSV file.

    Parameters
    ----------
    frequencies : ndarray, shape (M,), float64
        Signed angular frequencies in internal units.
    modes : ndarray, shape (3N, M), float64
        Normal mode eigenvectors; column i is mode i (0-based).
    system : openmm.System
        Used to obtain per-particle masses for mass-weighting.
    output_file : str
        Path to the output CSV file.  Overwritten if it already exists.
    n_modes : int, optional
        Number of non-rigid modes to include (default: 20).
    """
    n_particles = system.getNumParticles()
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)]
    inv_sqrt_m = 1 / np.sqrt(masses)
    m_vector = np.repeat(inv_sqrt_m, 3)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mode', 'Frequency (cm⁻¹)', 'Collectivity'])

        for i in range(6, min(6+n_modes, modes.shape[1])):
            # Mass-weight the mode
            mw_mode = modes[:, i] * m_vector
            mw_mode /= np.linalg.norm(mw_mode)

            # Convert to wavenumber
            freq_cm = frequencies[i] * 108.58  # Conversion factor

            # Calculate collectivity
            kappa = compute_collectivity(mw_mode, n_particles)

            writer.writerow([i+1, f"{freq_cm:.2f}", f"{kappa:.4f}"])

    logger.info(f"Saved collectivity data to {output_file}\n")

def plot_mode_contributions(eigenvalues, output_file=None, n_modes=10):
    """
    Plot per-mode and cumulative variance contributions for the first N
    non-rigid normal modes.

    Parameters
    ----------
    eigenvalues : ndarray, shape (K,), float64
        Sorted eigenvalues as returned by compute_normal_modes.
    output_file : str, optional
        Path to save the figure (PNG, 300 dpi).  If None, displays
        interactively.
    n_modes : int, optional
        Number of non-rigid modes to include in the plot (default: 10).
    """
    # Exclude rigid-body modes (first 6 near-zero eigenvalues)
    non_rigid_evals = eigenvalues[6:]

    # Calculate the variance explained by each mode
    # In NMA, the variance is proportional to 1/λ (fluctuation magnitude)
    epsilon = 1e-10
    variances = 1 / (np.abs(non_rigid_evals) + epsilon)

    # Calculate the total variance
    total_variance = np.sum(variances)

    # Calculate proportion of variance for each mode
    proportion_variance = variances / total_variance

    # Calculate cumulative proportion of variance
    cumulative_variance = np.cumsum(proportion_variance[:n_modes]) * 100

    # Create plot
    plt.figure(figsize=(12, 6))

    # Create subplot for proportion of variance
    plt.subplot(1, 2, 1)
    modes_indices = np.arange(1, n_modes+1)
    plt.bar(modes_indices, proportion_variance[:n_modes] * 100, alpha=0.7, color='skyblue')
    plt.title('Proportion of Variance by Mode')
    plt.xlabel('Mode Index (excluding rigid-body modes)')
    plt.ylabel('Proportion of Variance (%)')
    plt.xticks(modes_indices)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create subplot for cumulative proportion of variance
    plt.subplot(1, 2, 2)
    plt.plot(modes_indices, cumulative_variance, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    plt.title('Cumulative Proportion of Variance')
    plt.xlabel('Mode Index (excluding rigid-body modes)')
    plt.ylabel('Cumulative Variance (%)')
    plt.xticks(modes_indices)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add percentage labels to points
    for i, val in enumerate(cumulative_variance):
        plt.annotate(f'{val:.1f}%', (modes_indices[i], val),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9)

    plt.tight_layout()

    # Add explanatory text
    plt.figtext(0.5, 0.01,
                f"First {n_modes} non-rigid modes account for {cumulative_variance[-1]:.1f}% of total variance",
                ha="center", fontsize=10, style='italic')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved mode contribution plot to {output_file}\n")
    else:
        plt.show()

def plot_atomic_fluctuations(system, eigenvalues, modes, topology, output_file=None, temperature=300, n_modes=None, start_mode=6):
    """
    Compute per-residue RMSF from normal modes and plot as a line graph.

    Parameters
    ----------
    system : openmm.System
        Used to obtain per-particle masses.
    eigenvalues : ndarray, shape (K,), float64
        Sorted eigenvalues from compute_normal_modes.
    modes : ndarray, shape (3N, K), float64
        Normal mode eigenvectors; eigenvalues[i] and modes[:, i]
        correspond to the same mode.
    topology : openmm.app.Topology
        Used to map atoms to residues for grouping and axis labelling.
    output_file : str, optional
        Path to save the figure (PNG, 300 dpi).  If None, displays
        interactively.
    temperature : float, optional
        Temperature in Kelvin (default: 300).
    n_modes : int, optional
        Number of non-rigid modes to include.  If None, uses all modes
        from start_mode onward.
    start_mode : int, optional
        Zero-based index of the first mode to include (default: 6).
    """
    n_particles = system.getNumParticles()

    # If n_modes not specified, use all non-rigid modes
    if n_modes is None:
        n_modes = modes.shape[1] - start_mode

    # Calculate RMSF for each atom
    rmsf_atom = np.zeros(n_particles)

    # Boltzmann constant in kcal/(mol·K)
    k_B = 0.0019872041

    # Get masses
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])

    # Calculate contribution from each mode
    for mode_idx in range(start_mode, start_mode + n_modes):
        # Skip near-zero eigenvalues to avoid division by zero
        if abs(eigenvalues[mode_idx]) < 1e-10:
            continue

        # Get the mode vector and reshape to (n_particles, 3)
        mode_vector = modes[:, mode_idx].reshape(n_particles, 3)

        # Calculate the mean square fluctuation for this mode
        # MSF = (k_B * T / ω²) * |u_i|² / m_i
        # where u_i is the displacement vector for atom i in this mode
        omega_sq = eigenvalues[mode_idx]
        msf_contribution = (k_B * temperature / omega_sq) * np.sum(mode_vector**2, axis=1) / masses

        # Add to total RMSF
        rmsf_atom += msf_contribution

    # Take square root to get RMSF in nm
    rmsf_atom = np.sqrt(rmsf_atom)

    # Convert to Angstrom (1 nm = 10 Å)
    rmsf_atom *= 10

    # Group atoms by residue
    residue_rmsf = {}
    residue_indices = {}
    for atom in topology.atoms():
        residue = atom.residue
        residue_id = residue.id
        if residue_id not in residue_rmsf:
            residue_rmsf[residue_id] = []
            residue_indices[residue_id] = len(residue_rmsf) - 1
        residue_rmsf[residue_id].append(rmsf_atom[atom.index])

    # Calculate average RMSF per residue
    residue_ids = sorted(residue_rmsf.keys())
    residue_means = [np.mean(residue_rmsf[resid]) for resid in residue_ids]
    residue_nums = list(range(len(residue_ids)))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot RMSF
    plt.plot(residue_nums, residue_means, 'b-', linewidth=1, alpha=0.7)
    plt.fill_between(residue_nums, 0, residue_means, alpha=0.3)

    plt.xlabel('Residue Index')
    plt.ylabel('RMS Fluctuation (Å)')
    plt.title(f'Residue Fluctuations from Normal Modes\n(T={temperature}K, {n_modes} modes)')
    plt.grid(True, alpha=0.3)

    # Add statistics to the plot
    avg_rmsf = np.mean(residue_means)
    max_rmsf = np.max(residue_means)
    plt.axhline(y=avg_rmsf, color='r', linestyle='--', alpha=0.7,
                label=f'Average: {avg_rmsf:.2f} Å')
    plt.legend()

    # Set x-axis ticks to show residue indices
    if len(residue_nums) > 0:
        tick_step = max(1, len(residue_nums) // 10)
        x_ticks = np.arange(0, len(residue_nums), tick_step)
        plt.xticks(x_ticks, x_ticks)

    # Adjust layout
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Residue RMSF plot saved to {output_file}\n")
    else:
        plt.show()

def plot_residue_cross_correlation(system, eigenvalues, modes, topology, output_file=None, temperature=300, n_modes=None, start_mode=6, use_gpu=True, use_multithreading=True):
    """
    Compute and plot the residue Dynamical Cross-Correlation Matrix (DCCM).

    Parameters
    ----------
    system : openmm.System
        Used to obtain per-particle masses.
    eigenvalues : ndarray, shape (K,), float64
        Sorted eigenvalues from compute_normal_modes.
    modes : ndarray, shape (3N, K), float64
        Normal mode eigenvectors; eigenvalues[i] and modes[:, i]
        correspond to the same mode.
    topology : openmm.app.Topology
        Used to map atoms to residues and to label heatmap axes.
    output_file : str, optional
        Path to save the figure (PNG, 300 dpi) and the matrix (.npy).
        If None, displays the figure interactively.
    temperature : float, optional
        Temperature in Kelvin (default: 300).
    n_modes : int, optional
        Number of non-rigid modes to include.  If None, uses all modes
        from start_mode onward.
    start_mode : int, optional
        Zero-based index of the first mode to include (default: 6).
    use_gpu : bool, optional
        Use CuPy for GPU-accelerated matrix operations if available;
        falls back to CPU on any error (default: True).
    use_multithreading : bool, optional
        Reserved for future use; the CPU path is already NumPy-vectorised
        (default: True).

    Raises
    ------
    RuntimeError
        If no valid eigenvalues are found in the selected mode range (GPU
        path only).
    """
    start_time = time.time()

    n_particles = system.getNumParticles()

    # If n_modes not specified, use all non-rigid modes
    if n_modes is None:
        n_modes = modes.shape[1] - start_mode

    # Get residue information
    residues = list(topology.residues())
    n_residues = len(residues)

    # Map atoms to residues
    atom_to_residue = np.zeros(n_particles, dtype=int)
    for atom in topology.atoms():
        residue = atom.residue
        atom_to_residue[atom.index] = residues.index(residue)

    # Create residue assignment matrix (n_particles x n_residues)
    R = np.zeros((n_particles, n_residues))
    for i in range(n_residues):
        R[np.where(atom_to_residue == i)[0], i] = 1.0

    # Boltzmann constant in kcal/(mol·K)
    k_B = 0.0019872041

    # Get masses
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])

    # Precompute mass factors
    mass_factor = 1 / np.sqrt(masses)

    # Initialize correlation matrix
    correlation_matrix = np.zeros((n_residues, n_residues))

    if use_gpu and cp.is_available():
        logger.info("Using GPU acceleration for DCCM calculation...")
        try:
            evals_slice = eigenvalues[start_mode:start_mode + n_modes]
            valid_mask  = np.abs(evals_slice) > 1e-10
            evals_valid = evals_slice[valid_mask]
            mode_start_valid = start_mode + np.where(valid_mask)[0][0] if valid_mask.any() else None

            if mode_start_valid is None:
                raise RuntimeError("No valid eigenvalues in selected mode range.")

            # Indices of valid modes (in full modes array)
            valid_mode_idx = start_mode + np.where(valid_mask)[0]

            w_gpu   = cp.asarray(k_B * temperature / evals_valid, dtype=cp.float64) # (M,)
            mf_gpu  = cp.asarray(mass_factor, dtype=cp.float64)                     # (N,)
            R_gpu   = cp.asarray(R, dtype=cp.float64)                               # (N, n_res)
            M_valid = len(evals_valid)

            # modes shape: (3N, M_valid); we need per-xyz slices of shape (N, M_valid)
            atom_corr_gpu = cp.zeros((n_particles, n_particles), dtype=cp.float64)
            for alpha in range(3):
                # Row indices for component alpha: alpha, 3+alpha, 6+alpha, ...
                rows_alpha = np.arange(alpha, 3 * n_particles, 3)
                U_alpha = cp.asarray(
                    modes[np.ix_(rows_alpha, valid_mode_idx)], dtype=cp.float64
                )  # (N, M_valid)
                # Mass-weight each atom's displacement in this component
                U_alpha *= mf_gpu[:, None]
                # Weighted outer product: Σ_m w_m · u_iα u_jα  = (U_alpha * w) @ U_alpha.T
                atom_corr_gpu += cp.dot(U_alpha * w_gpu[None, :], U_alpha.T)

            # Aggregate to residue level: C_res = R.T @ C_atom @ R
            correlation_matrix_gpu = cp.dot(cp.dot(R_gpu.T, atom_corr_gpu), R_gpu)
            correlation_matrix = cp.asnumpy(correlation_matrix_gpu)

            del atom_corr_gpu, correlation_matrix_gpu, R_gpu, mf_gpu, w_gpu
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            logger.warning(f"GPU acceleration failed: {e}. Falling back to CPU.")
            use_gpu = False

    if not use_gpu or not cp.is_available():
        logger.info("Using CPU for DCCM calculation...")

        # Fully vectorised CPU path — mirrors the GPU logic but uses NumPy.
        # Select valid modes and corresponding eigenvalues.
        evals_slice = eigenvalues[start_mode:start_mode + n_modes]
        valid_mask  = np.abs(evals_slice) > 1e-10
        evals_valid = evals_slice[valid_mask]
        valid_mode_idx = start_mode + np.where(valid_mask)[0]

        if valid_mode_idx.size == 0:
            logger.warning("No valid eigenvalues in selected mode range; DCCM will be zero.")
        else:
            w = k_B * temperature / evals_valid    # (M_valid,) weights

            atom_corr = np.zeros((n_particles, n_particles), dtype=np.float64)
            for alpha in range(3):
                rows_alpha = np.arange(alpha, 3 * n_particles, 3)
                U_alpha = modes[np.ix_(rows_alpha, valid_mode_idx)]     # (N, M_valid)
            U_alpha = U_alpha * mass_factor[:, None]                    # mass-weight
                atom_corr += np.dot(U_alpha * w[None, :], U_alpha.T)    # Σ_m w_m u_iα u_jα

            # Aggregate to residue level
            correlation_matrix = R.T @ atom_corr @ R

    # Normalize to get correlation coefficients between -1 and 1
    diag = np.diag(correlation_matrix)
    norm_matrix = np.sqrt(np.outer(diag, diag))
    correlation_matrix = correlation_matrix / (norm_matrix + 1e-10)  # Avoid division by zero

    duration = time.time() - start_time
    logger.info(f"DCCM calculation completed in {duration:.2f} seconds")

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create a diverging colormap
    cmap = plt.cm.RdBu_r
    norm = colors.Normalize(vmin=-1, vmax=1)

    # Plot the correlation matrix with inverted y-axis
    im = plt.imshow(correlation_matrix, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    # Set labels
    plt.xlabel('Residue Index', fontsize=12)
    plt.ylabel('Residue Index', fontsize=12)
    plt.title(f'Residue Cross-Correlation Matrix\n(T={temperature}K, {n_modes} modes)', fontsize=14)

    # Set ticks to show approximately 10 ticks per axis
    tick_step = max(1, n_residues // 10)
    residue_ticks = np.arange(0, n_residues, tick_step)
    residue_labels = [f'{residues[i].id}' for i in residue_ticks]

    plt.xticks(residue_ticks, residue_labels, rotation=90)
    plt.yticks(residue_ticks, residue_labels)

    # Adjust layout
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Residue dynamical cross-correlation plot saved to {output_file}\n")

        # Also save the correlation matrix as a numpy file
        np.save(output_file.replace('.png', '.npy'), correlation_matrix)
    else:
        plt.show()

def main():
    """
    Entry point for the ENM normal mode analysis pipeline.

    Configuration is set via the CONFIG dictionary, which includes:
    - PDB_FILE: Input PDB file
    - MODEL_TYPE: 'ca' for Cα-only or 'heavy' for heavy-atom ENM
    - CUTOFF: Cutoff distance for interactions
    - SPRING_CONSTANT: Spring constant for ENM bonds
    - MAX_MODES: Number of non-rigid modes to compute
    - OUTPUT_FOLDER: Folder name where to write the output files
    - OUTPUT_MODES: Number of modes to save
    - WRITE_NM_VEC: Whether to write mode vectors to text files
    - WRITE_NM_TRJ: Whether to write mode trajectory files
    - COLLECTIVITY: Whether to compute modes collectivity
    - PLOT_CONTRIBUTIONS: Whether to plot modes cumulative contribution to internal dynamics
    - PLOT_RMSF: Whether to plot modes RMSF
    - PLOT_DCCM: Whether to build and plot residue dynamical cross correlation matrix
    - USE_GPU: Whether to use GPU acceleration

    Raises
    ------
    SystemExit
        Exits with code 1 on any unhandled exception; full traceback is
        printed to stderr.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Map arguments to CONFIG dictionary
    CONFIG = {
        "PDB_FILE": args.input,
        "MODEL_TYPE": args.type,
        "CUTOFF": args.cutoff,
        "SPRING_CONSTANT": args.spring_constant,
        "MAX_MODES": args.max_modes,
        "OUTPUT_FOLDER": args.output,
        "OUTPUT_MODES": args.output_modes,
        "WRITE_NM_VEC": args.no_nm_vec,
        "WRITE_NM_TRJ": args.no_nm_trj,
        "COLLECTIVITY": args.no_collectivity,
        "PLOT_CONTRIBUTIONS": args.no_contributions,
        "PLOT_RMSF": args.no_rmsf,
        "PLOT_DCCM": args.no_dccm,
        "USE_GPU": args.no_gpu
    }

    # Create output folder
    output_folder = CONFIG["OUTPUT_FOLDER"]
    os.makedirs(output_folder, exist_ok=True)

    # Get input filename without extension
    input_file = CONFIG["PDB_FILE"]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_prefix = os.path.join(output_folder, base_name)

    logger.info("Starting Normal Mode Analysis...\n")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Mode: {'Cα-only ENM' if CONFIG['MODEL_TYPE'] == 'ca' else 'Heavy-atom ENM'}")
    logger.info(f"Cutoff: {CONFIG['CUTOFF'] or ('15.0Å' if CONFIG['MODEL_TYPE'] == 'ca' else '12.0Å')}")
    logger.info(f"Spring constant: {CONFIG['SPRING_CONSTANT']} kcal/mol/Å²")
    logger.info(f"Number of modes to output: {CONFIG['OUTPUT_MODES']}\n")

    try:
        # Prefix to output files
        if CONFIG["MODEL_TYPE"] == 'ca':
            prefix = "ca"
        else:
            prefix = "heavy"

        # Create system
        system, topology, positions = create_system(
            CONFIG["PDB_FILE"],
            model_type=CONFIG["MODEL_TYPE"],
            cutoff=CONFIG["CUTOFF"],
            output_prefix = output_prefix,
            spring_constant=CONFIG["SPRING_CONSTANT"],
        )

        # Compute Hessian
        hessian = hessian_enm(
            system,
            positions,
            use_gpu=CONFIG["USE_GPU"]
        )

        # Mass-weight Hessian
        mw_hessian = mass_weight_hessian(
            hessian,
            system
        )

        # Compute Normal Modes
        frequencies, modes, eigenvalues = compute_normal_modes(
            mw_hessian,
            n_modes=CONFIG["MAX_MODES"],
            use_gpu=CONFIG["USE_GPU"]
        )

        np.save(f"{output_prefix}_{prefix}_frequencies.npy", frequencies)
        np.save(f"{output_prefix}_{prefix}_modes.npy", modes)
        logger.info(f"Results saved to {output_prefix}_{prefix}_*.npy files")

        # Write collectivity data
        if CONFIG["COLLECTIVITY"]:
            collectivity_file = f"{output_prefix}_{prefix}_collectivity.csv"
            write_collectivity(
                frequencies, modes, system,
                collectivity_file,
                n_modes=20  # Use first 20 non-rigid modes
            )

        # Plot internal dynamics contributions
        if CONFIG["PLOT_CONTRIBUTIONS"]:
            output_file = f"{output_prefix}_{prefix}_contributions.png"
            plot_mode_contributions(
                eigenvalues,
                output_file,
                n_modes=20  # Use first 20 non-rigid modes
            )

        # Plot RMSF
        if CONFIG["PLOT_RMSF"]:
            output_file=f"{output_prefix}_{prefix}_rmsf.png"
            rmsf = plot_atomic_fluctuations(
                system, eigenvalues, modes, topology,
                output_file,
                temperature=300,  # Room temperature
                n_modes=50,       # Use first 50 non-rigid modes
            )

        # Plot Residue Cross Correlation
        if CONFIG["PLOT_DCCM"]:
            output_file=f"{output_prefix}_{prefix}_dccm.png"
            dccm = plot_residue_cross_correlation(
                system, eigenvalues, modes, topology,
                output_file,
                temperature=300,            # Room temperature
                n_modes=50,                 # Use first 50 non-rigid modes
                use_gpu=CONFIG["USE_GPU"],  # Enable GPU usage
                use_multithreading=True     # Enable multithreading for CPU
            )

        # Write mode vectors
        if CONFIG["WRITE_NM_VEC"]:
            num_modes = min(CONFIG["OUTPUT_MODES"], len(frequencies)-6)
            logger.info(f"Writing vectors for {num_modes} modes...\n")
            mode_vectors_prefix = f"{output_prefix}_{prefix}"
            write_nm_vectors(
                modes, frequencies, system, topology,
                mode_vectors_prefix,
                n_modes=CONFIG["OUTPUT_MODES"],
                start_mode=6  # Start from mode 7 (index 6)
            )

        # Write mode trajectories
        if CONFIG["WRITE_NM_TRJ"]:
            num_modes = min(CONFIG["OUTPUT_MODES"], len(frequencies)-6)
            logger.info(f"Generating trajectories for {num_modes} modes...\n")
            write_nm_trajectories(
                topology, positions, modes, frequencies,
                f"{output_prefix}_{prefix}", system, CONFIG["MODEL_TYPE"],
                n_modes=CONFIG["OUTPUT_MODES"],
                start_mode=6  # Start from mode 7 (index 6)
            )

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
