# 3DFEM
## Python 3D finite element code

This python code allows for solving 3D structural problems with or without uncertainties using the finite element method.
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
* Uncertainty Quantification:
  - Parametric probabilistic model for the mass density and Young's modulus in dynamics solvers
  - Nonparametric probabilistic models for reduced matrices in dynamics solvers
  - Generalized probabilistic approach, including both parametric and nonparametric uncertainties, in dynamics solvers
  - Direct Monte Carlo method for uncertainty propagation

### Post-processing
* Plotting of Frequency-Response Functions (FRF), time trajectories
* Plotting of confidence intervals on FRF and time trajectories in stochastic simulations
* Plotting of probability density functions at a given step using gaussian kernel density estimation
* Export of a deformed mesh to VTK format (PolyData legacy format)
* Export of a series of deformed meshes to VTK format for animations (elastic modes, time-domain solutions)

----

## Examples of solutions

Visuals of results from the various provided sample scripts can be found [here](https://github.com/rcapillon/3DFEM/blob/main/results/README.md).

|||
|:----:|:----:|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/animation_time_beam.gif" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF5_3generalized.png" width="400"> |

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
* Uncertainty Quantification:
  - Parametric probabilistic models for all elastic coefficients
  - Causal nonparametric probabilistic models for linear viscoelastic materials
* Parallelization of stochastic solvers

### Post-processing
* Plotting of modal coordinates at given step
* Support for including element stress and strain components in the VTK files
