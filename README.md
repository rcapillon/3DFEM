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
* Linear time-domain dynamics using the Newmark scheme and a reduced-order model based on elastic modes

### Post-processing
* Export of a deformed mesh to VTK format (PolyData legacy format)
* Export of a series of deformed meshes to VTK format for animations (for visualizing elastic modes, or deformations in time-domain dynamical analyses)

----

## Examples of solutions

### Modal analysis

![Beam Mode](https://github.com/rcapillon/3DFEM/blob/main/visuals/animations/beam_mode1.gif)

*First linear elastic mode of a clamped-clamped beam displayed in Paraview*

### Time-domain dynamics

![Beam Time Dynamics](https://github.com/rcapillon/3DFEM/blob/main/visuals/animations/beam_dynamics.gif)

*Linear elastic cantilever beam with Rayleigh damping pulled up by a constant force on its free end, then released*

![Plate Time Dynamics](https://github.com/rcapillon/3DFEM/blob/main/visuals/animations/plate_dynamics.gif)

*Linear elastic plate with Rayleigh damping excited by a nodal force, on the center of the upper face, varying with a bell curve envelope*

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
* Proper Orthogonal Decomposition (POD) for nonlinear problems

### Forces
* Body forces
* Surface forces

### Solvers
* Frequency-domain dynamical analysis
* Newton-Raphson method for geometrically nonlinear elastostatics and elastodynamics
* Arc-length method for geometrically nonlinear elastostatics and elastodynamics with strong nonlinearities (e.g. post-buckling analysis)

### Post-processing
* Plotting of Frequency-Response Functions (FRF)
* Support for including element stress and strain components in the VTK files

### Uncertainty Quantification
* Typical Maximum Entropy-probability distributions for elastic coefficients (Young's modulus, Poisson's ratio, anisotropic coefficients) for **parametric uncertainties**
* Algorithm for finding the Maximum Entropy distribution with arbitrary constraints for **parametric uncertainties**
* Typical Maximum Entropy-probability distributions for mass, linear elastic stiffness, Rayleigh damping and linear viscoelastic damping matrices for **nonparametric uncertainties**
* Monte Carlo method for uncertainty propagation with plotting of confidence intervals
* Gaussian Kernel Density Estimation (GKDE) for the estimation of probability density functions of observable quantities
* Causal nonparametric probabilistic models (linear viscoelasticity)
