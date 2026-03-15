# Elastic Network Model Analysis
[![DOI](https://zenodo.org/badge/1042955611.svg)](https://doi.org/10.5281/zenodo.17898498)


This program performs Elastic Network Model analysis to study protein dynamics by simplifying structures into spring networks. It calculates normal modes to identify collective motions, supports both Cα and heavy-atom models, and generates various analyses including fluctuations, correlations, and mode visualizations.

The implementation features GPU acceleration and parallel processing for efficient computation of large biomolecular systems. It offers comprehensive analysis tools including RMSF, DCCM, mode contribution plots, collectivity calculations, and trajectory visualizations, all accessible through an intuitive command-line interface with automated output organization and detailed progress reporting.

* ****

- [Elastic Network Model Analysis](#elastic-network-model-analysis)
  - [Method Overview](#method-overview)
    - [Physical Motivation and Coarse-Graining](#physical-motivation-and-coarse-graining)
    - [Network Construction](#network-construction)
    - [The Hessian Matrix](#the-hessian-matrix)
    - [Normal Mode Analysis and Diagonalization](#normal-mode-analysis-and-diagonalization)
    - [Rigid-Body Modes and the Null Space](#rigid-body-modes-and-the-null-space)
    - [Equipartition and the Physical Meaning of Eigenvalues](#equipartition-and-the-physical-meaning-of-eigenvalues)
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
  - [Citing](#citing)
  - [Contact](#contact)

* ****

## Method Overview

### Physical Motivation and Coarse-Graining

Proteins are not rigid bodies: their function is intimately connected to their internal dynamics, ranging from local side-chain rotations and loop fluctuations to large-scale collective domain motions. Classical molecular dynamics (MD) simulations can capture these phenomena in atomic detail, but they are computationally expensive and often struggle to reach the timescales (microseconds to milliseconds) relevant to biologically important conformational changes.

Elastic Network Models offer a powerful and computationally inexpensive alternative. The central insight behind ENMs is that the low-frequency, large-amplitude collective motions of a protein — those most relevant to function — are predominantly determined by the overall topology of the molecular structure, not by the precise details of atomic interactions. In other words, the shape of the molecule, encoded by which atoms or residues are spatially close to one another, largely dictates the repertoire of accessible motions.

This motivates a **coarse-graining** strategy: instead of representing every atom with a detailed force field, the protein is reduced to a set of representative interaction sites connected by harmonic springs. In the **Cα model** (also called ANM, the Anisotropic Network Model), each residue is represented by a single point placed at its α-carbon. In the **heavy-atom model**, all non-hydrogen atoms are retained, yielding a finer-grained representation at the cost of a larger Hessian matrix. The choice of model involves a tradeoff between computational cost and the resolution of the dynamical description.

### Network Construction

Given a set of $N$ interaction sites (Cα atoms or heavy atoms) with equilibrium positions $`\mathbf{r}_i^0`$, the elastic network is constructed by connecting every pair of sites $i$ and $j$ whose equilibrium distance $`r_{ij}^0 = |\mathbf{r}_i^0 - \mathbf{r}_j^0|`$ falls within a specified **cutoff distance** $r_c$:

$$
r_{ij}^0 \leq r_c
$$

Typical cutoff values are 10–15 Å for the Cα model and 7–12 Å for the heavy-atom model. The cutoff is a key parameter: too small a value yields a disconnected or sparse network that fails to capture long-range coupling, while too large a value over-densifies the network and can wash out functionally relevant fluctuation patterns.

The total potential energy of the system under the harmonic approximation is:

$$
V = \frac{1}{2} \sum_{i < j} k_{ij} \left(r_{ij} - r_{ij}^0\right)^2
$$

where $k_{ij}$ is the spring constant between sites $i$ and $j$, $r_{ij}$ is the instantaneous distance between them, and $r_{ij}^0$ is their equilibrium distance. In the simplest ANM formulation, a **uniform spring constant** $k_{ij} = k$ is used for all connected pairs. This is a deliberate simplification: the spring constant encodes the stiffness of the local environment, and setting it uniformly to $k$ (with default $k = 1.0$ kcal/mol/Å²) means the model's predictions are expressed in units relative to $k$. More sophisticated variants assign distance-dependent spring constants (e.g., $k_{ij} \propto (r_{ij}^0)^{-\alpha}$), but the uniform model already captures the essential topology of collective motions.

### The Hessian Matrix

The dynamical properties of the network are encoded in the **Hessian matrix** $\mathbf{H}$, a $3N \times 3N$ symmetric matrix of second derivatives of the potential energy with respect to atomic displacements, evaluated at the equilibrium configuration:

$$
H_{i\alpha,\, j\beta} = \frac{\partial^2 V}{\partial u_{i\alpha}\, \partial u_{j\beta}}\Bigg|_{\mathbf{u}=0}
$$

where $u_{i\alpha}$ is the displacement of site $i$ along Cartesian direction $\alpha \in \{x, y, z\}$, and similarly for $u_{j\beta}$. The factor of three degrees of freedom per site is what distinguishes this **anisotropic** (ANM) formulation from simpler isotropic models: the Hessian retains the full directional information of each pairwise spring, which is essential for producing oriented mode trajectories and the vector dot products required by the DCCM.

Carrying out the differentiation of the pairwise harmonic potential, the off-diagonal $3 \times 3$ super-element connecting sites $i \neq j$ is:

$$
\mathbf{H}_{ij} = -\frac{k_{ij}}{(r_{ij}^0)^2} \begin{pmatrix} \Delta x^2 & \Delta x \Delta y & \Delta x \Delta z \\
\Delta y \Delta x & \Delta y^2 & \Delta y \Delta z \\
\Delta z \Delta x & \Delta z \Delta y & \Delta z^2 \end{pmatrix}
$$

where $\Delta x = x_i^0 - x_j^0$, $\Delta y = y_i^0 - y_j^0$, $\Delta z = z_i^0 - z_j^0$ are the components of the equilibrium difference vector $`\mathbf{r}_{ij}^0`$. This super-element is nonzero only when $r_{ij}^0 \leq r_c$, so $\mathbf{H}$ is sparse for typical cutoff distances. The diagonal blocks are set by the self-consistency condition (Newton's third law):

$$
\mathbf{H}_{ii} = -\sum_{j \neq i} \mathbf{H}_{ij}
$$

which ensures that $\mathbf{H}$ is positive semi-definite and that rigid-body motions have zero energy cost (see below).

### Normal Mode Analysis and Diagonalization

The equations of motion for the mass-weighted displacements $\tilde{\mathbf{u}}_i = \sqrt{m_i}\, \mathbf{u}_i$ (where $m_i$ is the mass of site $i$) take the form:

$$
\mathbf{M}^{-1/2} \mathbf{H}\, \mathbf{M}^{-1/2}\, \tilde{\mathbf{u}} = -\lambda\, \tilde{\mathbf{u}}
$$

where $\mathbf{M}$ is the $3N \times 3N$ diagonal mass matrix. Seeking solutions of the form $\tilde{\mathbf{u}}(t) = \mathbf{e}^{(k)} e^{i\omega_k t}$ leads to the standard **eigenvalue problem**:

$$
\tilde{\mathbf{H}}\, \mathbf{e}^{(k)} = \lambda_k\, \mathbf{e}^{(k)}
$$

where $\tilde{\mathbf{H}} = \mathbf{M}^{-1/2} \mathbf{H}\, \mathbf{M}^{-1/2}$ is the mass-weighted Hessian. The eigenvalues $\lambda_k \geq 0$ are proportional to the squared angular frequencies $\omega_k^2 = \lambda_k$ (in appropriate units), and the eigenvectors $\mathbf{e}^{(k)}$ — also called **normal mode vectors** or **mode shapes** — define the direction and pattern of collective atomic displacement in mode $k$.

Because the Hessian is real, symmetric, and positive semi-definite, it can always be diagonalized by an orthogonal transformation:

$$
\mathbf{H} = \mathbf{U}\, \boldsymbol{\Lambda}\, \mathbf{U}^T
$$

where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_{3N})$ and $\mathbf{U}$ is the matrix of column eigenvectors.

### Rigid-Body Modes and the Null Space

The Hessian of any translationally and rotationally invariant potential has exactly **six zero eigenvalues** (five for linear molecules), corresponding to three global translations and three global rotations. These are the "trivial" modes: they represent rigid-body motions of the entire molecule that cost no energy. Because the spring network is built around pairwise distances (which are invariant under rigid-body transformations), these six modes are guaranteed to have $\lambda_k = 0$ by construction.

In practice, numerical diagonalization yields six eigenvalues very close to — but not exactly — zero, due to floating-point arithmetic. These modes are identified and systematically excluded from all physical analyses. The first **non-trivial** mode is mode 7 (using 1-based indexing), corresponding to the lowest-frequency collective internal motion, typically involving the largest-amplitude domain movements. Modes are ordered by increasing frequency: low-frequency modes are large-scale and collective, while high-frequency modes are localized and stiff.

### Equipartition and the Physical Meaning of Eigenvalues

Under the classical harmonic approximation, the **equipartition theorem** states that each normal mode carries an average thermal energy of $\frac{1}{2}k_B T$. The mean-square displacement amplitude of mode $k$ is therefore:

$$
\langle A_k^2 \rangle = \frac{k_B T}{\lambda_k}
$$

This has a profound implication: **low-frequency modes (small $\lambda_k$) contribute large-amplitude fluctuations**, while high-frequency modes (large $\lambda_k$) contribute small fluctuations. The total thermal fluctuation of the system is dominated by the handful of lowest-frequency modes, which is why ENM-based analyses of fluctuations and correlations are already quite accurate using only the first 10–20 non-trivial modes.

Furthermore, the **inverse of the Hessian** (its pseudo-inverse, excluding the null space) defines the **covariance matrix** of atomic displacements at thermal equilibrium:

$$
\langle u_{i\alpha}\, u_{j\beta} \rangle = k_B T\, [\mathbf{H}^+]_{i\alpha, j\beta}
$$

where $\mathbf{H}^+$ is the Moore-Penrose pseudo-inverse. This relationship is the foundation for the RMSF and DCCM calculations described below.

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

- **`*_mode_*.txt`**: Text files containing mode vectors for individual modes

### 3. Trajectory Files
- **`*_mode_*_traj.pdb`**: PDB trajectories visualizing mode motions, generated by displacing the equilibrium structure along each mode eigenvector with amplitudes scaled by $\sqrt{k_B T / \lambda_k}$

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

The **collectivity** metric $\kappa$ quantifies how many atoms participate significantly in a given normal mode. It is based on the **Shannon entropy** of the distribution of squared atomic displacement magnitudes across the mode:

$$
\kappa_k = \frac{1}{N} \exp\!\left(-\sum_{i=1}^{N} p_i^{(k)} \ln p_i^{(k)}\right)
$$

where the participation weights are defined as:

$$
p_i^{(k)} = \frac{|\mathbf{u}_i^{(k)}|^2}{\displaystyle\sum_{j=1}^N |\mathbf{u}_j^{(k)}|^2}
$$

and $\mathbf{u}_i^{(k)}$ is the 3D displacement vector of site $i$ in mode $k$. The normalization factor $1/N$ ensures that $\kappa_k \in (0, 1]$. When all atoms participate equally in the mode (a maximally delocalized motion), the entropy $-\sum p_i \ln p_i$ reaches its maximum value of $\ln N$, and $\kappa_k = 1$. When only a single atom moves (a maximally localized mode), the entropy is zero and $\kappa_k \to 0$.

In practice, low-frequency modes typically have high collectivity ($\kappa > 0.5$), reflecting their large-scale, global character. High-frequency modes tend to be localized ($\kappa \ll 1$) and correspond to local stiff deformations. Identifying modes with anomalously low collectivity for their frequency can reveal biologically relevant hinges or allosteric sites.

### Proportion of Variance Plot

Under the equipartition theorem, each mode $k$ contributes a variance proportional to $1/\lambda_k$ to the total mean-square displacement. The **fractional variance** explained by mode $k$ is:

$$
f_k = \frac{1/\lambda_k}{\displaystyle\sum_{m=7}^{M} 1/\lambda_m}
$$

where the sum runs over all $M - 6$ non-trivial modes (modes 7 through $M$, excluding the six rigid-body modes). This fraction is exactly analogous to the proportion of variance explained by a principal component in PCA.

The plot displays two complementary quantities: a bar chart of the individual fractional contributions $f_k$ for the first $N$ reported modes, and an overlaid line chart of the **cumulative variance** $\sum_{k=7}^{n} f_k$ as a function of the number of modes included. This allows the user to assess how many modes are needed to capture a given percentage of the total structural variance — a key diagnostic for understanding the dimensionality of the functional dynamics.

### RMSF Calculation

The **Root Mean Square Fluctuation** of site $i$ measures its expected thermal mobility, computed as an ensemble average over all non-trivial modes weighted by their thermal amplitudes:

$$
\text{RMSF}_i = \sqrt{\frac{k_B T}{m_i} \sum_{k=7}^{M} \frac{|\mathbf{u}_i^{(k)}|^2}{\lambda_k}}
$$

where $k_B$ is Boltzmann's constant, $T$ is the absolute temperature, $m_i$ is the mass of site $i$, $\mathbf{u}_i^{(k)}$ is the displacement vector of site $i$ in mode $k$, and $\lambda_k$ is the corresponding eigenvalue.

This expression follows directly from the equipartition-weighted covariance matrix: the mean-square displacement of site $i$ is $\langle |\mathbf{u}_i|^2 \rangle = k_B T \sum_k |\mathbf{u}_i^{(k)}|^2 / \lambda_k$ (with mass-weighting absorbed into the eigenvectors if $\tilde{\mathbf{H}}$ is used). Residues in flexible loops or terminal regions accumulate contributions from many modes and thus display large RMSF values, while residues in tight secondary structure elements or buried cores show small RMSF values. The RMSF computed from the ENM correlates well with experimental B-factors from crystallographic data ($B_i = 8\pi^2 \langle |\mathbf{u}_i|^2 \rangle / 3$), providing a useful validation of the model.

### Dynamical Cross-Correlation Matrix (DCCM)

The **Dynamical Cross-Correlation Matrix** (DCCM) captures the extent to which pairs of residues fluctuate in a correlated or anti-correlated manner. The cross-correlation coefficient between residues $i$ and $j$ is:

$$
C_{ij} = \frac{\langle \Delta\mathbf{r}_i \cdot \Delta\mathbf{r}_j \rangle}{\sqrt{\langle |\Delta\mathbf{r}_i|^2 \rangle \cdot \langle |\Delta\mathbf{r}_j|^2 \rangle}}
$$

where $\Delta\mathbf{r}_i$ is the displacement vector of residue $i$ from its equilibrium position. The numerator is the **cross-covariance** of the displacement vectors (their dot product, summed over the ensemble), and the denominator normalizes by the geometric mean of the individual mean-square displacements to yield a quantity in the range $[-1, +1]$.

In terms of the normal modes, the cross-covariance is:

$$
\langle \Delta\mathbf{r}_i \cdot \Delta\mathbf{r}_j \rangle = k_B T \sum_{k=7}^{M} \frac{\mathbf{u}_i^{(k)} \cdot \mathbf{u}_j^{(k)}}{\lambda_k}
$$

A value of $C_{ij} = +1$ means residues $i$ and $j$ always move in the same direction (correlated), while $C_{ij} = -1$ means they always move in opposite directions (anti-correlated). By construction, $C_{ii} = 1$ for all $i$. Off-diagonal positive correlations often indicate residues belonging to the same rigid domain or secondary structure element; negative correlations frequently appear between residues on opposite sides of a hinge, which move in opposing directions during domain motions. The DCCM is therefore a rich diagnostic tool for identifying allosteric pathways, domain boundaries, and concerted conformational changes.

The **Dynamical Cross-Correlation Matrix** (DCCM) captures the extent to which pairs of residues fluctuate in a correlated or anti-correlated manner. The cross-correlation coefficient between residues $i$ and $j$ is:

$$
C_{ij} = \frac{\langle \Delta\mathbf{r}_i \cdot \Delta\mathbf{r}_j \rangle}{\sqrt{\langle |\Delta\mathbf{r}_i|^2 \rangle \cdot \langle |\Delta\mathbf{r}_j|^2 \rangle}}
$$

where $\Delta\mathbf{r}_i$ is the displacement vector of residue $i$ from its equilibrium position. The numerator is the **cross-covariance** of the displacement vectors (their dot product, summed over the ensemble), and the denominator normalizes by the geometric mean of the individual mean-square displacements to yield a quantity in the range $[-1, +1]$.

In terms of the normal modes, the cross-covariance is:

$$
\langle \Delta\mathbf{r}_i \cdot \Delta\mathbf{r}_j \rangle = k_B T \sum_{k=7}^{M} \frac{\mathbf{u}_i^{(k)} \cdot \mathbf{u}_j^{(k)}}{\lambda_k}
$$

A value of $C_{ij} = +1$ means residues $i$ and $j$ always move in the same direction (correlated), while $C_{ij} = -1$ means they always move in opposite directions (anti-correlated). By construction, $C_{ii} = 1$ for all $i$. Off-diagonal positive correlations often indicate residues belonging to the same rigid domain or secondary structure element; negative correlations frequently appear between residues on opposite sides of a hinge, which move in opposing directions during domain motions. The DCCM is therefore a rich diagnostic tool for identifying allosteric pathways, domain boundaries, and concerted conformational changes.

[Back to top ↩](#)
* ****

## Usage Examples

### Basic Cα analysis:

```
python enm.py -i protein.pdb \
              -o results
```

### Heavy-atom analysis with custom parameters:

```
python enm.py -i protein.pdb \
              -o results \
              -t heavy \
              -c 10.0 \
              -k 0.5 \
              -n 20
```

### Minimal analysis (only essential outputs):

```
python enm.py -i protein.pdb \
              -o results \
              --no_nm_trj \
              --no_collectivity \
              --no_contributions \
              --no_rmsf \
              --no_dccm
```

### Force CPU-only computation:

```
python enm.py -i protein.pdb \
              -o results \
              --no_gpu
```

[Back to top ↩](#)
* ****

## Dependencies

- `Python >= 3.8`

- `NumPy >= 1.21`

- `SciPy >= 1.7`

- `OpenMM >= 7.7`

- `Numba >= 0.55`

- `Matplotlib >= 3.5`

- `CuPy >= 8.0` (optional, for GPU acceleration. **Note:** cupy requires matching CUDA toolkit.)

[Back to top ↩](#)
* ****

## Citing

Please cite the following paper if you are using ENM in your work:

[Pedro Túlio de Resende-Lara. (2025). pedro-tulio/enm: Elastic Network Model (v1.0.0). Zenodo. DOI: 10.5281/zenodo.17898499](https://zenodo.org/records/17898499)

[Back to top ↩](#)

## Contact

If you experience a bug or have any question or suggestion, feel free to contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
