# 3DFEM
## Python 3D finite element code

This python code allows for solving 3D structural problems using the finite element method.
New features will be added over time.

*This code has **NOT** been validated on reference cases yet.*

## Requirements and dependencies
* Python 3
* numpy, scipy

* Paraview is suggested for visualizing exported VTK files, especially since animations are generated as a series of VTK files as handled by this software.

## Current features:

### Meshes
* Tetrahedral mesh generation from a set of points using scipy.spatial.Delaunay
* 4-node tetrahedral (Tet4), 6-node prism (Prism6), 8-node brick (Brick8) elements support
* Support for meshes containing different types of elements, possibly of different orders

### Materials
* Linear isotropic elastic materials

### Structural matrices
* Consistent mass matrix
* Linear elastic stiffness matrix
* Rayleigh damping matrix (linear combination of mass and stiffness matrices)
* Support only for zero-Dirichlet boundary conditions
* Full Gauss quadrature scheme

### Reduced-Order Modeling
* Projection on linear elastic modes

### Forces
* Nodal forces
* Support for modulation of the initial force vector by a given function over time steps or frequency steps in dynamical analyses

### Solvers
* Modal analysis
* Linear static analysis
* Linear frequency-domain dynamics using a reduced-order model based on elastic modes
* Linear time-domain dynamics using the Newmark scheme and a reduced-order model based on elastic modes

### Post-processing
* Plotting of Frequency-Response Functions (FRF), time trajectories
* Export of a deformed mesh to VTK format (PolyData legacy format)
* Export of a series of deformed meshes to VTK format for animations (for visualizing elastic modes, or deformations in time-domain dynamical analyses)

----

## Examples of solutions

### Modal analysis

<img src="https://github.com/rcapillon/3DFEM/blob/main/visuals/animations/beam_mode1.gif" width=500>

*First linear elastic mode of a clamped-clamped beam displayed in Paraview*

### Frequency-domain dynamics
<img src="https://github.com/rcapillon/3DFEM/blob/main/visuals/images/plate_frequency_15000Hz.png" width=500>
<img src="https://github.com/rcapillon/3DFEM/blob/main/visuals/images/plate_frf.png" width=500>

*Linear elastic plate excited by a nodal force at the center of the upper face (deformed mesh and Frequency-Response Function calculated at the excitation point)*

### Time-domain dynamics

<img src="https://github.com/rcapillon/3DFEM/blob/main/visuals/animations/plate_dynamics.gif" width=500>
<img src="https://github.com/rcapillon/3DFEM/blob/main/visuals/images/plate_time_displacement.png" width=500>

*Linear elastic plate excited by a nodal force at the center of the upper face (deformed mesh and vertical displacement at the excitation point)*

<img src="https://github.com/rcapillon/3DFEM/blob/main/visuals/animations/beam_dynamics.gif" width=500>

*Linear elastic cantilever beam with Rayleigh damping pulled up by a constant force on its free end, then released*

----

## Intended future features: 

### Meshes
* Support for Tet10, Prism18, Brick27 elements

### Materials
* All anisotropy classes for elastic materials
* Linear viscoelastic materials using the Generalized Maxwell Model

### Structural matrices
* Arbitrary Dirichlet boundary conditions

### Reduced-Order Modeling
* Support for specifying an arbitrary pre-computed reduced-order basis
* Proper Orthogonal Decomposition (POD) for nonlinear problems

### Forces
* Body forces
* Surface forces

### Solvers
* Newton-Raphson method for geometrically nonlinear elastostatics and elastodynamics
* Arc-length method for geometrically nonlinear elastostatics and elastodynamics with strong nonlinearities (e.g. post-buckling analysis)

### Post-processing
* Plotting of modal coordinates at given step
* Support for including element stress and strain components in the VTK files

### Uncertainty Quantification
* Typical Maximum Entropy-probability distributions for elastic coefficients (Young's modulus, Poisson's ratio, anisotropic coefficients) for **parametric uncertainties**
* Typical Maximum Entropy-probability distributions for mass, linear elastic stiffness, Rayleigh damping and linear viscoelastic damping matrices for **nonparametric uncertainties**
* Monte Carlo method for uncertainty propagation with plotting of confidence intervals
* Gaussian Kernel Density Estimation (GKDE) for the estimation of probability density functions of observable quantities
* Causal nonparametric probabilistic models (linear viscoelasticity)
