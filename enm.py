import numpy as np
import openmm as mm
from openmm import app, unit, Platform
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, identity
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib import colors
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cupy as cp
import argparse
import sys
import os
import time
import csv
import logging

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
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET_COLOR)
        return f"{color}{log_message}{self.RESET_COLOR}"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_formatter = AnsiColorFormatter("..:ENM> {levelname}: {message}", style="{")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('enm.out')
file_formatter = logging.Formatter("{asctime} ..:ENM> {levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def parse_arguments():
    """
    Parse command-line arguments for the ENM analysis.
    """
    parser = argparse.ArgumentParser(description='Elastic Network Model Normal Mode Analysis')

    parser.add_argument('-i', '--input', required=True, help='Input PDB file')
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

    parser.add_argument('--no_nm_vec', action='store_false', help='Disable writing mode vectors')
    parser.add_argument('--no_nm_trj', action='store_false', help='Disable writing mode trajectories')
    parser.add_argument('--no_collectivity', action='store_false', help='Disable collectivity calculation')
    parser.add_argument('--no_contributions', action='store_false', help='Disable mode contributions plot')
    parser.add_argument('--no_rmsf', action='store_false', help='Disable RMSF plot')
    parser.add_argument('--no_dccm', action='store_false', help='Disable DCCM plot')
    parser.add_argument('--no_gpu', action='store_false', help='Disable GPU acceleration')

    return parser.parse_args()

def create_system(pdb_file, model_type='ca', cutoff=None, spring_constant=1.0, output_prefix="input"):
    """
    Create an ENM system from a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    model_type : str, optional
        'ca' for Cα-only ENM or 'heavy' for all heavy atoms. Default: 'ca'.
    cutoff : float, optional
        Interaction cutoff in Å. Defaults to 15.0 Å (CA) or 12.0 Å (heavy).
    spring_constant : float, optional
        ENM spring constant in kcal/mol/Å². Default: 1.0.
    output_prefix : str, optional
        Prefix for output files.

    Returns
    -------
    system : openmm.System
    topology : openmm.app.Topology
    positions : openmm.unit.Quantity

    Raises
    ------
    ValueError
        If model_type is invalid or no relevant atoms are found.
    """
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
    Build a Cα-only ENM system.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    cutoff : float
        Interaction cutoff in Å.
    spring_constant : float
        ENM spring constant in kcal/mol/Å².
    output_prefix : str
        Prefix for the saved Cα PDB file.

    Returns
    -------
    system : openmm.System
        OpenMM system with one particle per Cα atom (mass 12.011 Da).
    topology : openmm.app.Topology
        Topology containing only Cα atoms.
    positions : openmm.unit.Quantity
        Cα positions in nm.

    Raises
    ------
    ValueError
        If no Cα atoms are found in the structure.
    """
    logger.info("Creating Cα-only system using Elastic Network Model...")
    pdb = app.PDBFile(pdb_file)

    # Collect Cα atoms and their Cartesian positions from the full PDB topology
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

    # Build a reduced topology containing only Cα atoms, preserving residue identity
    new_topology = app.Topology()
    new_chain = new_topology.addChain()
    residue_map = {}

    for i, (orig_idx, residue) in enumerate(ca_info):
        if residue not in residue_map:
            new_res = new_topology.addResidue(f"{residue.name}{residue.id}", new_chain)
            residue_map[residue] = new_res
        new_topology.addAtom("CA", app.element.carbon, residue_map[residue])

    # Create the OpenMM system and register one particle per Cα with uniform carbon mass
    system = mm.System()
    positions = [mm.Vec3(*pos) * unit.nanometer for pos in positions_list]
    positions_quantity = unit.Quantity(positions)

    carbon_mass = 12.011 * unit.daltons
    for _ in range(n_atoms):
        system.addParticle(carbon_mass)

    # Define the harmonic ENM potential: V = 0.5 * k * (r - r0)²
    # r0 for each bond is set to the equilibrium distance in the input structure
    enm_force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
    enm_force.addGlobalParameter("k", spring_constant)
    enm_force.addPerBondParameter("r0")

    # Compute the full pairwise distance matrix (in nm) and convert the cutoff
    pos_np = np.array(positions_list, dtype=np.float32)
    dist_matrix = squareform(pdist(pos_np))
    cutoff_nm = cutoff * 0.1        # Å to nm
    min_distance_nm = 2.9 * 0.1     # exclude bonded Cα pairs (< 2.9 Å apart)

    # Select all pairs within the ENM cutoff, excluding nearest-neighbour Cα pairs
    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = dist_matrix[i, j]
            if min_distance_nm <= dist <= cutoff_nm:
                bonds.append((i, j, dist))

    # Register each bond with its equilibrium distance as the rest length r0
    for i, j, dist in bonds:
        enm_force.addBond(i, j, [dist])

    system.addForce(enm_force)
    logger.info(f"Added {len(bonds)} ENM bonds with cutoff={cutoff}Å, min_distance=2.9Å, k={spring_constant} kcal/mol/Å²")

    # Suppress rigid-body translation in any downstream OpenMM simulations (does not affect the Hessian calculation)
    system.addForce(mm.CMMotionRemover())

    # Write the reduced Cα structure to PDB and normalise HETATM records
    ca_pdb_file = f"{output_prefix}_ca_structure.pdb"
    with open(ca_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    logger.info(f"C-alpha structure saved to {ca_pdb_file}\n")

    convert_hetatm_to_atom(ca_pdb_file)

    return system, new_topology, positions_quantity

def _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix):
    """
    Build a heavy-atom ENM system.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    cutoff : float
        Interaction cutoff in Å.
    spring_constant : float
        ENM spring constant in kcal/mol/Å².
    output_prefix : str
        Prefix for the saved heavy-atom PDB file.

    Returns
    -------
    system : openmm.System
        OpenMM system with one particle per heavy atom, each with its
        element's mass.
    topology : openmm.app.Topology
        Topology containing only heavy atoms.
    positions : openmm.unit.Quantity
        Heavy-atom positions in nm.

    Raises
    ------
    ValueError
        If no heavy atoms are found in the structure.
    """
    logger.info("Creating heavy-atom system using Elastic Network Model...")
    pdb = app.PDBFile(pdb_file)

    # Collect all non-hydrogen atoms, retaining element info for element-specific masses
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

    # Build a reduced topology preserving residue membership for each heavy atom
    new_topology = app.Topology()
    new_chain = new_topology.addChain()
    residue_map = {}

    for i, (orig_idx, residue, name, element) in enumerate(heavy_atoms):
        if residue not in residue_map:
            new_res = new_topology.addResidue(f"{residue.name}{residue.id}", new_chain)
            residue_map[residue] = new_res
        new_topology.addAtom(name, element, residue_map[residue])

    # Create the OpenMM system and register each particle with its element's standard mass
    system = mm.System()
    positions = [mm.Vec3(*pos) * unit.nanometer for pos in positions_list]
    positions_quantity = unit.Quantity(positions)

    for _, _, _, element in heavy_atoms:
        system.addParticle(element.mass)

    # Define the harmonic ENM potential: V = 0.5 * k * (r - r0)²
    # r0 for each bond is set to the equilibrium distance in the input structure
    enm_force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
    enm_force.addGlobalParameter("k", spring_constant)
    enm_force.addPerBondParameter("r0")

    # Compute the full pairwise distance matrix (in nm) and convert the cutoff
    pos_np = np.array(positions_list, dtype=np.float32)
    dist_matrix = squareform(pdist(pos_np))
    cutoff_nm = cutoff * 0.1        # Å to nm
    min_distance_nm = 2.0 * 0.1     # exclude covalently bonded heavy-atom pairs (< 2.0 Å)

    # Select all pairs within the ENM cutoff, skipping likely covalent neighbours
    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = dist_matrix[i, j]
            if min_distance_nm <= dist <= cutoff_nm:
                bonds.append((i, j, dist))

    # Register each bond with its equilibrium distance as the rest length r0
    for i, j, dist in bonds:
        enm_force.addBond(i, j, [dist])

    system.addForce(enm_force)
    logger.info(f"Added {len(bonds)} ENM bonds with cutoff={cutoff}Å, min_distance=2.0Å, k={spring_constant} kcal/mol/Å²")

    # Suppress rigid-body translation in any downstream OpenMM simulations
    system.addForce(mm.CMMotionRemover())

    # Write the heavy-atom structure to PDB for reference
    heavy_pdb_file = f"{output_prefix}_heavy_structure.pdb"
    with open(heavy_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    logger.info(f"Heavy-atom structure saved to {heavy_pdb_file}\n")

    return system, new_topology, positions_quantity

def compute_hessian_sparse(pos_array, bonds, k_val, n_particles):
    """
    Assemble the ENM Hessian as a sparse CSR matrix.

    Each bonded pair (i, j) contributes four 3×3 blocks: +block on diagonals
    (i,i) and (j,j), -block on off-diagonals (i,j) and (j,i). Contributions
    are collected in flat COO arrays and summed on conversion to CSR.

    Parameters
    ----------
    pos_array : ndarray
        Particle positions (N×3), float64, in nm.
    bonds : ndarray
        Bond table with columns [i, j, r0], float64.
    k_val : float
        Spring constant in kcal/mol/Å².
    n_particles : int
        Number of particles.

    Returns
    -------
    hessian : scipy.sparse.csr_array
        Sparse Hessian (3N × 3N). Duplicate diagonal entries from overlapping
        bonds have been summed.

    Raises
    ------
    ValueError
        If no entries are assembled (empty bond list).
    """
    from scipy.sparse import coo_array

    n_bonds = bonds.shape[0]
    # Each bond contributes 4 blocks of 3×3 scalars = 36 entries; pre-allocate at maximum
    max_entries = n_bonds * 36

    row_idx = np.empty(max_entries, dtype=np.int32)
    col_idx = np.empty(max_entries, dtype=np.int32)
    values  = np.empty(max_entries, dtype=np.float64)
    ptr = 0

    for idx in range(n_bonds):
        i = int(bonds[idx, 0])
        j = int(bonds[idx, 1])

        # Displacement vector from atom i to atom j
        r_ij = pos_array[j] - pos_array[i]
        dist = np.sqrt(r_ij[0]**2 + r_ij[1]**2 + r_ij[2]**2)
        if dist < 1e-6:
            # Skip degenerate bonds where atoms are effectively coincident
            continue

        # Unit vector along the bond: e_ij = r_ij / |r_ij|
        e_ij = r_ij / dist

        # 3×3 Kirchhoff block for this bond: B = k * (e_ij ⊗ e_ij)
        # This is the second derivative of V = 0.5*k*(r-r0)² with respect
        # to the Cartesian coordinates of i and j, evaluated at r = r0
        block = k_val * np.outer(e_ij, e_ij)

        # DOF offsets for atoms i and j (3 DOFs each: x, y, z)
        i3, j3 = 3 * i, 3 * j

        for a in range(3):
            for b in range(3):
                v = block[a, b]
                # Diagonal blocks: +B at (i,i) and (j,j) — restoring forces on each atom
                row_idx[ptr] = i3 + a;  col_idx[ptr] = i3 + b;  values[ptr] =  v;  ptr += 1
                row_idx[ptr] = j3 + a;  col_idx[ptr] = j3 + b;  values[ptr] =  v;  ptr += 1
                # Off-diagonal blocks: -B at (i,j) and (j,i) — coupling between atoms
                row_idx[ptr] = i3 + a;  col_idx[ptr] = j3 + b;  values[ptr] = -v;  ptr += 1
                row_idx[ptr] = j3 + a;  col_idx[ptr] = i3 + b;  values[ptr] = -v;  ptr += 1

    if ptr == 0:
        raise ValueError("No Hessian entries assembled — check bond list.")

    # Assemble the COO matrix and convert to CSR; tocsr() automatically sums
    # duplicate entries on the diagonal (contributions from multiple bonds sharing
    # the same atom pair) without any explicit symmetrisation step
    n_dof = 3 * n_particles
    hessian = coo_array(
        (values[:ptr], (row_idx[:ptr], col_idx[:ptr])),
        shape=(n_dof, n_dof),
        dtype=np.float64
    )
    return hessian.tocsr()

def hessian_enm(system, positions):
    """
    Build and regularize the sparse ENM Hessian.

    Extracts bond parameters from the OpenMM CustomBondForce, assembles the
    (3N × 3N) Hessian as a CSR sparse matrix, and applies a small diagonal
    regularization (1e-8) for numerical stability.

    Parameters
    ----------
    system : openmm.System
        System containing the ENM CustomBondForce.
    positions : openmm.unit.Quantity
        Particle positions.

    Returns
    -------
    hessian : scipy.sparse.csr_array
        Regularized sparse Hessian (3N × 3N).

    Raises
    ------
    ValueError
        If no CustomBondForce is found in the system.
    """
    n_particles = system.getNumParticles()
    n_dof = 3 * n_particles

    # Locate the CustomBondForce that encodes the ENM harmonic springs
    enm_force = next((f for f in system.getForces() if isinstance(f, mm.CustomBondForce)), None)
    if enm_force is None:
        raise ValueError("No ENM force found in system")

    # Retrieve the global spring constant k and convert positions to a plain numpy array (nm)
    k_val = enm_force.getGlobalParameterDefaultValue(0)
    num_bonds = enm_force.getNumBonds()
    pos_array = np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(unit.nanometer)], dtype=np.float64)

    logger.info(f"Computing sparse Hessian for {n_particles} particles ({num_bonds} bonds)...")
    t0 = time.time()

    # Extract bond parameters from the OpenMM force into a compact (num_bonds × 3) array
    # Columns: [atom_i, atom_j, r0] where r0 is the equilibrium bond length in nm
    bonds_list = np.empty((num_bonds, 3), dtype=np.float64)
    for bond_idx in range(num_bonds):
        i, j, [r0] = enm_force.getBondParameters(bond_idx)
        bonds_list[bond_idx] = (i, j, r0)

    # Assemble the sparse Hessian from bond geometry (see compute_hessian_sparse)
    hessian = compute_hessian_sparse(pos_array, bonds_list, k_val, n_particles)

    # Ensure CSR format for efficient arithmetic operations
    hessian = hessian.tocsr()

    # Apply a small Tikhonov regularisation (λI, λ = 1e-8) to the diagonal to prevent
    # exact singularity; the true zero eigenvalues (rigid-body modes) remain near-zero
    # and are filtered out during diagonalisation
    reg = diags([1e-8] * n_dof, 0, format='csr')
    hessian = hessian + reg

    nnz = hessian.nnz
    dense_elements = n_dof ** 2
    logger.info(f"Sparse Hessian: {nnz:,} non-zeros ({100*nnz/dense_elements:.3f}% density)")
    logger.info(f"ENM Hessian computed in {time.time() - t0 :.2f} seconds\n")

    return hessian

def mass_weight_hessian(hessian, system):
    """
    Return the mass-weighted Hessian M^{-1/2} H M^{-1/2}.

    Parameters
    ----------
    hessian : scipy.sparse.csr_array
        Sparse Hessian (3N × 3N).
    system : openmm.System
        Source of particle masses.

    Returns
    -------
    mw_hessian : scipy.sparse.csr_array
        Mass-weighted Hessian, same sparsity pattern as input.
    """
    n_particles = system.getNumParticles()

    # Extract per-atom masses in Da; replace any zero-mass virtual sites with 1.0 Da
    # to avoid division-by-zero in the inverse square-root
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(n_particles)])
    masses[masses == 0] = 1.0

    # Expand the per-atom inverse square-root mass to a per-DOF vector (x, y, z repeated)
    # so that a single diagonal matrix D covers all 3N degrees of freedom
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)

    # Construct the sparse diagonal scaling matrix D = M^{-1/2}
    D = diags(inv_sqrt_m, 0, format='csr')

    # Return the mass-weighted Hessian H_mw = D H D = M^{-1/2} H M^{-1/2}
    # All three matrices are sparse, so no dense allocation occurs
    return D @ hessian @ D

def gpu_diagonalization(hessian, n_modes=None):
    """
    Diagonalize a dense Hessian using GPU acceleration.

    Used only for small systems (n_dof < 5000). Uses float32 for matrices
    larger than 3000 DOF to reduce VRAM usage.

    Parameters
    ----------
    hessian : ndarray
        Dense Hessian (3N × 3N).
    n_modes : int, optional
        Number of lowest modes to compute. If None, computes all modes.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues sorted in ascending order, shape (k,).
    eigenvectors : ndarray
        Corresponding eigenvectors, shape (3N, k).
    """
    mem_pool = cp.get_default_memory_pool()
    pinned_mem_pool = cp.get_default_pinned_memory_pool()

    with cp.cuda.Device(0):
        # Use float32 for large matrices to halve VRAM usage; float64 otherwise
        dtype = cp.float32 if hessian.shape[0] > 3000 else cp.float64
        hessian_gpu = cp.array(hessian, dtype=dtype)
        # Free the CPU copy immediately to reduce peak memory footprint
        del hessian

        if n_modes is not None:
            # Include 6 extra modes to account for rigid-body modes that will be discarded;
            # subset_by_index requests only the k lowest eigenvalue/eigenvector pairs
            n_modes = min(n_modes + 6, hessian_gpu.shape[0])
            eigenvalues, eigenvectors = cp.linalg.eigh(
                hessian_gpu, UPLO='L', subset_by_index=[0, n_modes - 1]
            )
        else:
            # Full diagonalisation — only feasible for small systems (n_dof < 5000)
            eigenvalues, eigenvectors = cp.linalg.eigh(hessian_gpu, UPLO='L')

        # Free GPU memory before transferring results to avoid holding two copies simultaneously
        del hessian_gpu
        mem_pool.free_all_blocks()
        pinned_mem_pool.free_all_blocks()

        # Transfer results back to CPU via a synchronised copy using the null stream
        eigenvalues_cpu = cp.asnumpy(eigenvalues, stream=cp.cuda.Stream.null)
        eigenvectors_cpu = cp.asnumpy(eigenvectors, stream=cp.cuda.Stream.null)
        del eigenvalues, eigenvectors

    return eigenvalues_cpu, eigenvectors_cpu

def compute_normal_modes(hessian, n_modes=None, use_gpu=False):
    """
    Compute normal modes by partially diagonalizing the mass-weighted Hessian.

    For sparse input, uses the ARPACK shift-invert solver (``eigsh`` with
    ``sigma=1e-6``) to compute only the requested lowest-frequency modes.
    For dense input with n_dof < 5000 and GPU available, falls back to
    ``gpu_diagonalization``; otherwise uses ``scipy.linalg.eigh``.

    Parameters
    ----------
    hessian : scipy.sparse matrix or ndarray
        Mass-weighted Hessian (3N × 3N).
    n_modes : int, optional
        Number of non-rigid modes to compute. Defaults to 50.
    use_gpu : bool, optional
        Attempt GPU acceleration for small dense systems.

    Returns
    -------
    frequencies : ndarray
        Sorted non-zero frequencies in internal units.
    modes : ndarray
        Eigenvectors, shape (3N, k).
    eigenvalues : ndarray
        Raw eigenvalues from the solver, length k.
    """
    from scipy.sparse import issparse

    n_dof = hessian.shape[0]
    # Request n_modes + 6 eigenvalues to guarantee that, after discarding the 6 rigid-body
    # modes (near-zero eigenvalues), at least n_modes vibrational modes remain
    n_request = min((n_modes or 50) + 6, n_dof - 1)

    t0 = time.time()

    if issparse(hessian):
        # Shift-invert ARPACK: transforming the problem to (H - σI)⁻¹ maps the smallest
        # eigenvalues of H to the largest of the shifted operator, making convergence fast.
        # σ = 1e-6 sits just above zero to avoid the exact null space of rigid-body modes
        logger.info(f"Diagonalizing sparse mass-weighted Hessian "
                    f"(requesting {n_request-6} non-trivial modes)...")
        eigenvalues, eigenvectors = eigsh(hessian, k=n_request, which='LM', sigma=1e-6)

    else:
        if use_gpu and n_dof < 5000:
            try:
                logger.info("Diagonalizing dense Hessian using GPU...")
                eigenvalues, eigenvectors = gpu_diagonalization(hessian, n_modes)
            except Exception as e:
                logger.warning(f"GPU diagonalization failed: {e}. Falling back to CPU.")
                use_gpu = False

        if not use_gpu or n_dof >= 5000:
            # subset_by_index restricts LAPACK to the lowest n_request eigenpairs,
            # avoiding the O(N³) cost of a full diagonalisation
            logger.info("Diagonalizing dense Hessian using CPU...")
            eigenvalues, eigenvectors = eigh(
                hessian,
                subset_by_index=[0, n_request - 1],
                driver='evr',
                overwrite_a=True,
                check_finite=False
            )

    logger.info(f"Diagonalization completed in {time.time() - t0 :.2f} seconds\n")

    # Identify and discard rigid-body modes: eigenvalues below a relative threshold
    # (1e-10 × max eigenvalue) are treated as numerically zero
    abs_evals = np.abs(eigenvalues)
    threshold = max(np.max(abs_evals) * 1e-10, 1e-10)
    valid_idx = abs_evals > threshold

    # Frequencies are the square root of the eigenvalues of the mass-weighted Hessian
    frequencies = np.sqrt(np.abs(eigenvalues[valid_idx]))
    valid_modes = eigenvectors[:, valid_idx]

    # Sort modes from lowest to highest frequency
    sort_idx = np.argsort(frequencies)
    return frequencies[sort_idx], valid_modes[:, sort_idx], eigenvalues

def convert_hetatm_to_atom(pdb_file):
    """
    Replace HETATM records with ATOM in a PDB file for visualization compatibility.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file. Edited in place.

    Returns
    -------
    None
    """
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('HETATM'):
            new_lines.append('ATOM  ' + line[6:])
        else:
            new_lines.append(line)

    with open(pdb_file, 'w') as f:
        f.writelines(new_lines)

def write_nm_vectors(modes, frequencies, system, topology, output_prefix, n_modes=10, start_mode=7):
    """
    Write normal mode eigenvectors to XYZ files, one file per mode.

    Parameters
    ----------
    modes : ndarray
        Mode vectors (3N × M).
    frequencies : ndarray
        Mode frequencies in internal units.
    system : openmm.System
        Source of particle count.
    topology : openmm.app.Topology
        Source of element symbols for XYZ header.
    output_prefix : str
        Prefix for output XYZ filenames.
    n_modes : int, optional
        Number of modes to write. Default: 10.
    start_mode : int, optional
        First mode index (0-based). Default: 7 (skips 6 rigid-body modes).

    Returns
    -------
    None
    """
    n_particles = system.getNumParticles()

    n_modes = min(n_modes, modes.shape[1] - start_mode)
    if n_modes <= 0:
        logger.warning("No modes available to write.")
        return

    elements = [atom.element.symbol for atom in topology.atoms()]

    for mode_idx in range(start_mode, start_mode + n_modes):
        freq = frequencies[mode_idx] * 108.58  # internal units → cm⁻¹
        mode_number = mode_idx + 1
        output_file = f"{output_prefix}_mode_{mode_number}.xyz"

        with open(output_file, 'w') as f:
            f.write(f"{n_particles}\n")
            f.write(f"Normal Mode {mode_number}, Frequency: {freq:.2f} cm⁻¹\n")

            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)
            for i in range(n_particles):
                x, y, z = mode_vector[i]
                f.write(f"{elements[i]:2s} {x:14.10f} {y:14.10f} {z:14.10f}\n")

def write_nm_trajectories(topology, positions, modes, frequencies, output_prefix, system, model_type, n_modes=10, start_mode=7, amplitude=4, num_frames=34):
    """
    Write PDB trajectories showing oscillatory motion along normal modes.

    Each mode is written as a separate multi-model PDB file. The trajectory
    is a piecewise linear oscillation: 0 → -amplitude → 0 → +amplitude → 0.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the system, used for writing PDB frames.
    positions : openmm.unit.Quantity
        Equilibrium positions in nm.
    modes : ndarray
        Mode vectors (3N × M).
    frequencies : ndarray
        Mode frequencies in internal units.
    output_prefix : str
        Prefix for output PDB filenames.
    system : openmm.System
        Source of particle masses.
    model_type : str
        Model type identifier, either 'ca' or 'heavy'.
    n_modes : int, optional
        Number of modes to write. Default: 10.
    start_mode : int, optional
        First mode index (0-based). Default: 7 (skips 6 rigid-body modes).
    amplitude : float, optional
        Peak displacement amplitude in Å. Default: 4.
    num_frames : int, optional
        Number of frames per trajectory. Default: 34.

    Returns
    -------
    None
    """
    n_particles = system.getNumParticles()

    n_modes = min(n_modes, modes.shape[1] - start_mode)
    if n_modes <= 0:
        logger.warning("No modes available to write trajectories.")
        return

    for mode_idx in range(start_mode, start_mode + n_modes):
        freq = frequencies[mode_idx] * 108.58  # cm⁻¹
        mode_number = mode_idx + 1
        output_file = f"{output_prefix}_mode_{mode_number}_traj.pdb"

        # Undo the mass-weighting applied during diagonalisation to recover
        # Cartesian displacement vectors: u_cart = M^{-1/2} · u_mw
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])
        masses[masses == 0] = 1.0
        inv_sqrt_m = np.repeat(1 / np.sqrt(masses), 3)
        u = modes[:, mode_idx] * inv_sqrt_m

        # Compute the RMS Cartesian displacement norm for amplitude normalisation
        rms = np.linalg.norm(u) / np.sqrt(n_particles)
        if rms < 1e-10:
            logger.warning(f"Skipping near-zero mode {mode_number}")
            continue

        # Scale the displacement vector so its RMS equals the requested amplitude (Å to nm)
        scaled_disp = u.reshape(n_particles, 3) * (amplitude / rms * 0.1)

        # Extract the equilibrium positions as a plain numpy array (nm)
        orig_pos = positions.value_in_unit(unit.nanometer)
        orig_pos_np = np.array([[p.x, p.y, p.z] for p in orig_pos])

        # Divide the trajectory into four equal segments tracing the oscillation
        # cycle: 0 → −A → 0 → +A → 0, giving a smooth back-and-forth motion
        seg1 = int(num_frames * 0.25)   # equilibrium → −amplitude
        seg2 = int(num_frames * 0.25)   # −amplitude  → equilibrium
        seg3 = int(num_frames * 0.25)   # equilibrium → +amplitude
        seg4 = num_frames - seg1 - seg2 - seg3  # +amplitude → equilibrium

        displacements = np.zeros((num_frames, n_particles, 3))

        # Compute the piecewise-linear scaling factor for each frame
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

        # Write one MODEL/ENDMDL block per frame as a multi-model PDB trajectory
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

        # Replace HETATM records so standard viewers treat all atoms consistently
        convert_hetatm_to_atom(output_file)

def compute_collectivity(mode_vector, n_atoms):
    """
    Compute mode collectivity (Tama & Sanejouand, 2001).

    Parameters
    ----------
    mode_vector : ndarray
        Flattened mode vector (3N,).
    n_atoms : int
        Number of atoms in the system.

    Returns
    -------
    collectivity : float
        Collectivity value in (0, 1]. Values near 1 indicate the mode
        involves concerted motion of all atoms; values near 0 indicate
        motion localised to a few atoms.
    """
    # Reshape the flattened mode vector into (N, 3) to work with per-atom displacements
    u = mode_vector.reshape(n_atoms, 3)

    # Compute the squared Euclidean norm of each atom's displacement vector
    norms = np.linalg.norm(u, axis=1)
    p = norms**2

    # Treat p as an unnormalised probability distribution (participation of each atom);
    # the small offset guards against log(0) for atoms with exactly zero displacement
    p += 1e-12  # guard against log(0)

    # Shannon entropy of the displacement distribution: S = -∑ p_i ln(p_i)
    entropy = -np.sum(p * np.log(p))

    # Collectivity κ = exp(S) / N: equals 1 when all atoms contribute equally,
    # approaches 0 when motion is localised to a single atom
    return np.exp(entropy) / n_atoms

def write_collectivity(frequencies, modes, system, output_file, n_modes=20):
    """
    Write mode collectivities to a CSV file.

    Parameters
    ----------
    frequencies : ndarray
        Mode frequencies in internal units.
    modes : ndarray
        Mode vectors (3N × M).
    system : openmm.System
        Source of particle masses.
    output_file : str
        Output CSV path.
    n_modes : int, optional
        Number of non-rigid modes to include. Default: 20.

    Returns
    -------
    None
    """
    n_particles = system.getNumParticles()

    # Build the per-DOF inverse square-root mass vector for mass-weighting the mode vectors
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)]
    inv_sqrt_m = 1 / np.sqrt(masses)
    m_vector = np.repeat(inv_sqrt_m, 3)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mode', 'Frequency (cm⁻¹)', 'Collectivity'])

        # Skip the first 6 modes (rigid-body translations and rotations)
        for i in range(6, min(6+n_modes, modes.shape[1])):
            # Mass-weight and normalise the eigenvector before computing collectivity
            mw_mode = modes[:, i] * m_vector
            mw_mode /= np.linalg.norm(mw_mode)

            freq_cm = frequencies[i] * 108.58  # internal units → cm⁻¹
            kappa = compute_collectivity(mw_mode, n_particles)
            writer.writerow([i+1, f"{freq_cm:.2f}", f"{kappa:.4f}"])

    logger.info(f"Saved collectivity data to {output_file}\n")

def plot_mode_contributions(eigenvalues, output_file=None, n_modes=10):
    """
    Plot per-mode and cumulative variance contributions for the first N non-rigid modes.

    Variance is proportional to 1/λ (mean-square fluctuation amplitude).

    Parameters
    ----------
    eigenvalues : ndarray
        Hessian eigenvalues.
    output_file : str, optional
        Path to save the figure. If None, displays interactively.
    n_modes : int, optional
        Number of non-rigid modes to plot. Default: 10.

    Returns
    -------
    None
    """
    # The first 6 eigenvalues correspond to rigid-body translations and rotations;
    # discard them so the variance analysis covers only vibrational modes
    non_rigid_evals = eigenvalues[6:]

    # Variance contribution of each mode is proportional to 1/λ
    # (from equipartition: ⟨x²⟩ = k_B T / λ)
    variances = 1 / (np.abs(non_rigid_evals) + 1e-10)
    proportion_variance = variances / np.sum(variances)

    # Cumulative sum over the first n_modes modes, expressed as a percentage
    cumulative_variance = np.cumsum(proportion_variance[:n_modes]) * 100

    plt.figure(figsize=(12, 6))

    # Left panel: per-mode proportion of variance as a bar chart
    plt.subplot(1, 2, 1)
    modes_indices = np.arange(1, n_modes+1)
    plt.bar(modes_indices, proportion_variance[:n_modes] * 100, alpha=0.7, color='skyblue')
    plt.title('Proportion of Variance by Mode')
    plt.xlabel('Mode Index (excluding rigid-body modes)')
    plt.ylabel('Proportion of Variance (%)')
    plt.xticks(modes_indices)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Right panel: cumulative variance as a connected scatter plot with annotations
    plt.subplot(1, 2, 2)
    plt.plot(modes_indices, cumulative_variance, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    plt.title('Cumulative Proportion of Variance')
    plt.xlabel('Mode Index (excluding rigid-body modes)')
    plt.ylabel('Cumulative Variance (%)')
    plt.xticks(modes_indices)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, val in enumerate(cumulative_variance):
        plt.annotate(f'{val:.1f}%', (modes_indices[i], val),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9)

    plt.tight_layout()
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
    Plot per-residue RMSF calculated from normal modes.

    Atomic MSF is computed as (k_B T / ω²) · |u_i|² / m_i and averaged
    per residue.

    Parameters
    ----------
    system : openmm.System
        Source of particle masses.
    eigenvalues : ndarray
        Hessian eigenvalues in internal units.
    modes : ndarray
        Mode vectors (3N × M).
    topology : openmm.app.Topology
        Used to map atoms to residues.
    output_file : str, optional
        Path to save the figure. If None, displays interactively.
    temperature : float, optional
        Temperature in K. Default: 300.
    n_modes : int, optional
        Number of modes to accumulate. Defaults to all non-rigid modes.
    start_mode : int, optional
        First mode index (0-based). Default: 6.

    Returns
    -------
    rmsf : ndarray
        Per-atom RMSF values in Å, shape (N,).
    """
    n_particles = system.getNumParticles()

    if n_modes is None:
        n_modes = modes.shape[1] - start_mode

    rmsf_atom = np.zeros(n_particles)
    k_B = 0.0019872041  # kcal/(mol·K)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])

    # Accumulate mean-square fluctuations over all requested modes.
    # For mode k with eigenvalue ω²_k the MSF contribution of atom i is:
    #   Δr²_i = (k_B T / ω²_k) · |u_i|² / m_i
    # where u_i is the 3-component eigenvector slice for atom i
    for mode_idx in range(start_mode, start_mode + n_modes):
        if abs(eigenvalues[mode_idx]) < 1e-10:
            # Skip residual near-zero modes that were not fully filtered during diagonalisation
            continue
        mode_vector = modes[:, mode_idx].reshape(n_particles, 3)
        omega_sq = eigenvalues[mode_idx]
        # MSF contribution: (k_B T / ω²) · |u_i|² / m_i
        rmsf_atom += (k_B * temperature / omega_sq) * np.sum(mode_vector**2, axis=1) / masses

    # Convert accumulated MSF to RMS fluctuation and rescale from nm to Å
    rmsf_atom = np.sqrt(rmsf_atom) * 10  # nm → Å

    # Average per-atom RMSF values to a single representative value per residue
    residue_rmsf = {}
    residue_indices = {}
    for atom in topology.atoms():
        residue = atom.residue
        residue_id = residue.id
        if residue_id not in residue_rmsf:
            residue_rmsf[residue_id] = []
            residue_indices[residue_id] = len(residue_rmsf) - 1
        residue_rmsf[residue_id].append(rmsf_atom[atom.index])

    residue_ids = sorted(residue_rmsf.keys())
    residue_means = [np.mean(residue_rmsf[resid]) for resid in residue_ids]
    residue_nums = list(range(len(residue_ids)))

    # Plot residue RMSF as a filled line graph with the mean RMSF marked as a reference line
    plt.figure(figsize=(12, 6))
    plt.plot(residue_nums, residue_means, 'b-', linewidth=1, alpha=0.7)
    plt.fill_between(residue_nums, 0, residue_means, alpha=0.3)

    plt.xlabel('Residue Index')
    plt.ylabel('RMS Fluctuation (Å)')
    plt.title(f'Residue Fluctuations from Normal Modes\n(T={temperature}K, {n_modes} modes)')
    plt.grid(True, alpha=0.3)

    avg_rmsf = np.mean(residue_means)
    max_rmsf = np.max(residue_means)
    plt.axhline(y=avg_rmsf, color='r', linestyle='--', alpha=0.7,
                label=f'Average: {avg_rmsf:.2f} Å')
    plt.legend()

    if len(residue_nums) > 0:
        tick_step = max(1, len(residue_nums) // 10)
        x_ticks = np.arange(0, len(residue_nums), tick_step)
        plt.xticks(x_ticks, x_ticks)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Residue RMSF plot saved to {output_file}\n")
    else:
        plt.show()

    return rmsf_atom

def plot_residue_cross_correlation(system, eigenvalues, modes, topology, output_file=None, temperature=300, n_modes=None, start_mode=6, use_gpu=True, use_multithreading=True):
    """
    Compute and plot the residue dynamical cross-correlation matrix (DCCM).

    Atomic correlations are computed as C = (k_B T / ω²) · U Uᵀ for each
    mode, then aggregated to residue level via R.T @ C @ R, where R is the
    atom-to-residue assignment matrix.

    Parameters
    ----------
    system : openmm.System
        Source of particle masses.
    eigenvalues : ndarray
        Hessian eigenvalues in internal units.
    modes : ndarray
        Mode vectors (3N × M).
    topology : openmm.app.Topology
        Used to map atoms to residues.
    output_file : str, optional
        Path to save the figure and the correlation matrix as a .npy file.
        If None, displays interactively.
    temperature : float, optional
        Temperature in K. Default: 300.
    n_modes : int, optional
        Number of modes to accumulate. Defaults to all non-rigid modes.
    start_mode : int, optional
        First mode index (0-based). Default: 6.
    use_gpu : bool, optional
        Use CuPy for GPU acceleration. Falls back to CPU on failure. Default: True.
    use_multithreading : bool, optional
        Use multithreading on the CPU path. Default: True.

    Returns
    -------
    correlation_matrix : ndarray
        Normalised residue cross-correlation matrix, shape (n_residues, n_residues),
        with values in [-1, 1].
    """
    t0 = time.time()

    n_particles = system.getNumParticles()

    if n_modes is None:
        n_modes = modes.shape[1] - start_mode

    residues = list(topology.residues())
    n_residues = len(residues)

    # Map each atom index to its parent residue index
    atom_to_residue = np.zeros(n_particles, dtype=int)
    for atom in topology.atoms():
        atom_to_residue[atom.index] = residues.index(atom.residue)

    # Build the binary atom-to-residue assignment matrix R (N × n_residues).
    # R[i, r] = 1 if atom i belongs to residue r, 0 otherwise.
    # Used to aggregate atom-level correlations to residue-level via Cᵣₑₛ = Rᵀ C R
    R = np.zeros((n_particles, n_residues))
    for i in range(n_residues):
        R[np.where(atom_to_residue == i)[0], i] = 1.0

    k_B = 0.0019872041  # kcal/(mol·K)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])
    # Per-atom scaling factor M^{-1/2} for converting mass-weighted eigenvectors
    # to Cartesian displacement vectors
    mass_factor = 1 / np.sqrt(masses)
    correlation_matrix = np.zeros((n_residues, n_residues))

    if use_gpu and cp.is_available():
        logger.info("Using GPU acceleration for DCCM calculation...")
        try:
            # Transfer all required arrays to GPU memory at once
            masses_gpu = cp.array(masses)
            eigenvalues_gpu = cp.array(eigenvalues[start_mode:start_mode+n_modes])
            modes_gpu = cp.array(modes[:, start_mode:start_mode+n_modes])
            R_gpu = cp.array(R)
            mass_factor_gpu = 1 / cp.sqrt(masses_gpu)
            correlation_matrix_gpu = cp.zeros((n_residues, n_residues))

            for mode_idx in range(n_modes):
                if cp.abs(eigenvalues_gpu[mode_idx]) < 1e-10:
                    continue

                # Recover Cartesian displacement vectors: u_cart = M^{-1/2} · u_mw
                mode_vector = modes_gpu[:, mode_idx].reshape(n_particles, 3)
                weighted_vectors = mode_vector * mass_factor_gpu[:, cp.newaxis]

                # Thermal prefactor: k_B T / ω²_k
                factor = k_B * temperature / eigenvalues_gpu[mode_idx]

                # Atomic correlation matrix for this mode: C = (k_B T / ω²) · U Uᵀ
                # where U is the (N×3) Cartesian displacement matrix
                atom_corr_matrix = factor * cp.dot(weighted_vectors, weighted_vectors.T)

                # Aggregate from atoms to residues: Cᵣₑₛ += Rᵀ C R
                # C = (k_B T / ω²) · U Uᵀ, aggregated to residues via R.T @ C @ R
                correlation_matrix_gpu += cp.dot(cp.dot(R_gpu.T, atom_corr_matrix), R_gpu)

            # Transfer the accumulated residue correlation matrix back to CPU
            correlation_matrix = cp.asnumpy(correlation_matrix_gpu)

        except Exception as e:
            logger.warning(f"GPU acceleration failed: {e}. Falling back to CPU.")
            use_gpu = False

    if not use_gpu or not cp.is_available():
        logger.info("Using CPU for DCCM calculation...")
        for mode_idx in range(start_mode, start_mode + n_modes):
            if abs(eigenvalues[mode_idx]) < 1e-10:
                continue

            # Recover Cartesian displacement vectors: u_cart = M^{-1/2} · u_mw
            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)
            weighted_vectors = mode_vector * mass_factor[:, np.newaxis]

            # Thermal prefactor: k_B T / ω²_k
            factor = k_B * temperature / eigenvalues[mode_idx]

            # Atomic correlation matrix for this mode and aggregate to residues
            atom_corr_matrix = factor * np.dot(weighted_vectors, weighted_vectors.T)
            correlation_matrix += np.dot(np.dot(R.T, atom_corr_matrix), R)

    # Normalise the accumulated covariance matrix to Pearson correlation coefficients
    # in [-1, 1]: C_norm[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j])
    # The small offset (1e-10) prevents division by zero for isolated residues
    diag = np.diag(correlation_matrix)
    norm_matrix = np.sqrt(np.outer(diag, diag))
    correlation_matrix = correlation_matrix / (norm_matrix + 1e-10)

    logger.info(f"DCCM calculation completed in {time.time() - t0 :.2f} seconds")

    # Plot the normalised correlation matrix as a diverging heatmap centred at zero
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.RdBu_r
    norm = colors.Normalize(vmin=-1, vmax=1)
    im = plt.imshow(correlation_matrix, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    plt.xlabel('Residue Index', fontsize=12)
    plt.ylabel('Residue Index', fontsize=12)
    plt.title(f'Residue Cross-Correlation Matrix\n(T={temperature}K, {n_modes} modes)', fontsize=14)

    # Thin out axis tick labels to avoid overcrowding on large structures
    tick_step = max(1, n_residues // 10)
    residue_ticks = np.arange(0, n_residues, tick_step)
    residue_labels = [f'{residues[i].id}' for i in residue_ticks]
    plt.xticks(residue_ticks, residue_labels, rotation=90)
    plt.yticks(residue_ticks, residue_labels)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Residue dynamical cross-correlation plot saved to {output_file}\n")
        # Also save the raw matrix for downstream analysis
        np.save(output_file.replace('.png', '.npy'), correlation_matrix)
    else:
        plt.show()

    return correlation_matrix

def main():
    """Run ENM normal mode analysis from the command line."""
    args = parse_arguments()

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

    output_folder = CONFIG["OUTPUT_FOLDER"]
    os.makedirs(output_folder, exist_ok=True)

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
        prefix = "ca" if CONFIG["MODEL_TYPE"] == 'ca' else "heavy"

        system, topology, positions = create_system(
            CONFIG["PDB_FILE"],
            model_type=CONFIG["MODEL_TYPE"],
            cutoff=CONFIG["CUTOFF"],
            output_prefix=output_prefix,
            spring_constant=CONFIG["SPRING_CONSTANT"],
        )

        hessian = hessian_enm(system, positions)
        mw_hessian = mass_weight_hessian(hessian, system)

        frequencies, modes, eigenvalues = compute_normal_modes(
            mw_hessian,
            n_modes=CONFIG["MAX_MODES"],
            use_gpu=CONFIG["USE_GPU"]
        )

        np.save(f"{output_prefix}_{prefix}_frequencies.npy", frequencies)
        np.save(f"{output_prefix}_{prefix}_modes.npy", modes)
        logger.info(f"Results saved to {output_prefix}_{prefix}_*.npy files")

        if CONFIG["COLLECTIVITY"]:
            collectivity_file = f"{output_prefix}_{prefix}_collectivity.csv"
            write_collectivity(
                frequencies, modes, system,
                collectivity_file,
                n_modes=20
            )

        if CONFIG["PLOT_CONTRIBUTIONS"]:
            output_file = f"{output_prefix}_{prefix}_contributions.png"
            plot_mode_contributions(
                eigenvalues,
                output_file,
                n_modes=20
            )

        if CONFIG["PLOT_RMSF"]:
            output_file=f"{output_prefix}_{prefix}_rmsf.png"
            rmsf = plot_atomic_fluctuations(
                system, eigenvalues, modes, topology,
                output_file,
                temperature=300,
                n_modes=50,
            )

        if CONFIG["PLOT_DCCM"]:
            output_file=f"{output_prefix}_{prefix}_dccm.png"
            dccm = plot_residue_cross_correlation(
                system, eigenvalues, modes, topology,
                output_file,
                temperature=300,
                n_modes=50,
                use_gpu=CONFIG["USE_GPU"],
                use_multithreading=True
            )

        if CONFIG["WRITE_NM_VEC"]:
            num_modes = min(CONFIG["OUTPUT_MODES"], len(frequencies)-6)
            logger.info(f"Writing vectors for {num_modes} modes...\n")
            write_nm_vectors(
                modes, frequencies, system, topology,
                f"{output_prefix}_{prefix}",
                n_modes=CONFIG["OUTPUT_MODES"],
                start_mode=6
            )

        if CONFIG["WRITE_NM_TRJ"]:
            num_modes = min(CONFIG["OUTPUT_MODES"], len(frequencies)-6)
            logger.info(f"Generating trajectories for {num_modes} modes...\n")
            write_nm_trajectories(
                topology, positions, modes, frequencies,
                f"{output_prefix}_{prefix}", system, CONFIG["MODEL_TYPE"],
                n_modes=CONFIG["OUTPUT_MODES"],
                start_mode=6
            )

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
