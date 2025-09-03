# Elastic Network Model Analysis

This program performs Elastic Network Model analysis to study protein dynamics by simplifying structures into spring networks. It calculates normal modes to identify collective motions, supports both Cα and heavy-atom models, and generates various analyses including fluctuations, correlations, and mode visualizations.

The implementation features GPU acceleration and parallel processing for efficient computation of large biomolecular systems. It offers comprehensive analysis tools including RMSF, DCCM, mode contribution plots, collectivity calculations, and trajectory visualizations, all accessible through an intuitive command-line interface with automated output organization and detailed progress reporting.

* ****

- [Elastic Network Model Analysis](#elastic-network-model-analysis)
  - [Method Overview](#method-overview)
    - [Theoretical Basis](#theoretical-basis)
  - [Input Requirements](#input-requirements)
    - [Required Input](#required-input)
    - [Optional Parameters](#optional-parameters)
    - [Feature Flags](#feature-flags)
  - [Output Files](#output-files)
    - [1. Structure Files](#1-structure-files)
    - [2. Mode Data Files](#2-mode-data-files)
    - [3. Trajectory Files](#3-trajectory-files)
    - [4. Analysis Files](#4-analysis-files)
  - [Analysis](#analysis)
    - [Collectivity Calculation](#collectivity-calculation)
    - [Proportion of Variance Plot](#proportion-of-variance-plot)
    - [RMSF Calculation](#rmsf-calculation)
    - [Dynamical Cross-Correlation Matrix (DCCM)](#dynamical-cross-correlation-matrix-dccm)
  - [Usage Examples](#usage-examples)
    - [Basic Cα analysis:](#basic-cα-analysis)
    - [Heavy-atom analysis with custom parameters:](#heavy-atom-analysis-with-custom-parameters)
    - [Minimal analysis (only essential outputs):](#minimal-analysis-only-essential-outputs)
    - [Force CPU-only computation:](#force-cpu-only-computation)
  - [Dependencies](#dependencies)
  - [Contact](#contact)
  - [License](#license)

* ****

## Method Overview

Elastic Network Model Analysis (ENM) is a computational technique used to study the large-scale collective motions of biomolecules. This method simplifies the complex potential energy surface of a protein by representing it as a network of harmonic springs connecting atoms or residues within a specified cutoff distance.

### Theoretical Basis
The ENM approach is based on the following principles:

1. **Coarse-Graining:** The protein structure is reduced to a set of representative points (Cα atoms or all heavy atoms)

2. **Harmonic Approximation:** Interactions between points are modeled as harmonic springs with a uniform force constant

3. **Contact-Based:** Springs connect all pairs of points within a specified cutoff distance (typically 10-15 Å)

The potential energy of the system is given by:

$$
V = \frac{1}{2} \sum_{i \lt j} k_{ij} {(r_{ij} - {r^0}_{ij})}^2
$$

where $k_{ij}$ is the spring constant, $r_{ij}$ is the current distance between points $i$ and $j$, and $r^0_{ij}$ is their equilibrium distance.

The Hessian matrix (second derivative of potential energy) is diagonalized to obtain normal modes, which represent collective motions of the protein. The first six modes with near-zero eigenvalues correspond to rigid-body translations and rotations, while the subsequent modes represent internal motions with increasing frequency.

[Back to top ↩](#)
* ****

## Input Requirements

### Required Input

- **PDB File:** A protein structure file in PDB format (**`-i`** or **`--input`**)

### Optional Parameters

- **Output Directory:** Specify output directory (**`-o`** or **`--output`**, default: output)

- **Model Type:** Choose between Cα-only or heavy-atom model (**`-t`** or **`--type`**, default: ca)

- **Cutoff Distance:** Interaction cutoff in Ångströms (**`-c`** or **`--cutoff`**, default: 15.0 for CA, 12.0 for heavy atoms)

- **Spring Constant:** Force constant in kcal/mol/Å² (**`-k`** or **`--spring_constant`**, default: 1.0)

- **Max Modes:** Number of non-rigid modes to compute (**`-m`** or **`--max_modes`**, default: all modes)

- **Output Modes:** Number of modes to save and analyze (**`-n`** or **`--output_modes`**, default: 10)

### Feature Flags

- **`--no_nm_vec`**: Disable writing mode vectors

- **`--no_nm_trj`**: Disable writing mode trajectories

- **`--no_collectivity`**: Disable collectivity calculation

- **`--no_contributions`**: Disable mode contributions plot

- **`--no_rmsf`**: Disable RMSF plot

- **`--no_dccm`**: Disable DCCM plot

- **`--no_gpu`**: Disable GPU acceleration

[Back to top ↩](#)
* ****

## Output Files

The script generates several output files organized in the specified output directory:

### 1. Structure Files
- **`*_ca_structure.pdb`** or **`*_aa_structure.pdb`**: Simplified structure file containing only the selected atoms

### 2. Mode Data Files
- **`*_frequencies.npy`**: NumPy array containing mode frequencies

- **`*_modes.npy`**: NumPy array containing mode vectors

- **`*_mode_*.xyz`**: Individual modes vectors written in [XYZ format](https://www.cgl.ucsf.edu/chimera/current/docs/UsersGuide/xyz.html)

### 3. Trajectory Files
- **`*_mode_*_traj.pdb`**: PDB trajectories visualizing mode motions

### 4. Analysis Files
- **`*_collectivity.csv`**: CSV file containing mode collectivity values

- **`*_contributions.png`**: Plot showing proportion of variance explained by modes

- **`*_rmsf.png`**: Plot of residue RMS fluctuations

- **`*_dccm.png`**: Residue dynamical cross-correlation matrix plot

- **`*_dccm.npy`**: NumPy array containing residue dynamical cross-correlation matrix

[Back to top ↩](#)
* ****

## Analysis

### Collectivity Calculation
The collectivity metric ($\kappa$) quantifies how many atoms participate in a normal mode. It is based on the Shannon entropy of the squared atomic displacements:

$$
\kappa = \frac{1}{N}\exp \left(-\sum_{i=1}^{N} p_i \ln p_i \right)
$$

where $p_i={|\vec{u_i}|}^2 / \sum_{j=1}^N {|\vec{u_j}|}^2$, and $\vec{u}_i$ and $\vec{u}_j$ are the displacement vector of atoms $i$ and $j$ in the mode, respectively.

A collectivity value of 1 indicates all atoms participate equally in the motion, while values close to 0 indicate localized motions.

### Proportion of Variance Plot
This plot shows the contribution of each mode to the total variance of atomic fluctuations. The variance explained by mode $i$ is proportional to $1/\lambda_i$, where $\lambda_i$ is the eigenvalue of the mode. The plot displays:

1. Individual contributions: Bar chart showing the percentage of total variance explained by each mode

2. Cumulative variance: Line chart showing the cumulative percentage of variance explained by the first $N$ modes

### RMSF Calculation
The Root Mean Square Fluctuation (RMSF) calculates the expected fluctuation of each residue based on the normal modes:

$$
RMSF_{i} = \sqrt{\frac{k_B T}{m_i} \sum_{k=7}^{M} \frac{{|\vec{u}_{i,k}|}^2}{\lambda_k}}
$$

where $k_B$ is Boltzmann's constant, ${T }$ is temperature, $m_i$ is the mass of atom $i$, $\vec{u}_{i,k}$ is the displacement vector of atom $i$ in mode $k$, and $\lambda_k$ is the eigenvalue of mode $k$.

### Dynamical Cross-Correlation Matrix (DCCM)
The DCCM shows correlated motions between residues:

$$
C_ij = \frac{\langle \Delta \vec{r}_i \cdot \Delta \vec{r}_j \rangle}{\sqrt{\langle \Delta \vec{r}^2_i \cdot \Delta \vec{r}^2_j \rangle}}
$$

where $\Delta \vec{r}_i$ and $\Delta \vec{r}_j$ are the displacement vector of residues $i$ and $j$, respectively. Values range from -1 (perfectly anti-correlated) to +1 (perfectly correlated).

[Back to top ↩](#)
* ****

## Usage Examples

### Basic Cα analysis:

```
python enm.py -i protein.pdb -o results
```

### Heavy-atom analysis with custom parameters:

```
python enm.py -i protein.pdb -o results -t heavy -c 10.0 -k 0.5 -n 20
```

### Minimal analysis (only essential outputs):

```
python enm.py -i protein.pdb -o results --no_nm_trj --no_collectivity --no_contributions --no_rmsf --no_dccm
```

### Force CPU-only computation:

```
python enm.py -i protein.pdb -o results --no_gpu
```

[Back to top ↩](#)
* ****

## Dependencies

- `Python >= 3.8`

- `NumPy >= 1.21`

- `SciPy >= 1.7`

- `OpenMM >= 7.7`

- `Numba >= 0.55`

- `CuPy >= 8.0` (optional, for GPU acceleration. **Note:** cupy requires matching CUDA toolkit.)

- `Matplotlib >= 3.5`

[Back to top ↩](#)
* ****

## Contact

If you experience a bug or have any doubt or suggestion, feel free to contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
