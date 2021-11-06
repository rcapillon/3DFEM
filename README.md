# 3DFEM
## Python 3D finite element code

This python code allows for solving 3D structural problems using the finite element method.
The code is intended to solve problems over unstructured meshes, no optimization is done by accounting for e.g. linear elements only.
New features will be added over time.

As of now, here is a summary of the code's capabilities:

### Meshes
* Tetrahedral mesh generation from a set of points using scipy.spatial.Delaunay
* 4-node tetrahedral (Tet4) elements support
* Support for unstructured meshes (useless for now, since a single element type is available)
* Full Gauss quadrature scheme

### Materials
* Linear isotropic elastic materials

### Structural matrices
* Consistent mass matrix
* Linear elastic stiffness matrix
* Rayleigh damping matrix (linear combination of mass and stiffness matrices)
* Support for zero-Dirichlet boundary conditions

### Reduced-Order Modeling
* Projection on linear elastic modes

### Forces
* Nodal forces
* Support for modulation of the initial force vector by a given function over timesteps in time-domain dynamical analyses

### Solvers
* Modal analysis
* Linear static analysis
* Newmark scheme for linear time-domain dynamics

### Post-processing
* Export of a deformed mesh to VTK format (PolyData legacy format)
* Export of a series of deformed meshes to VTK format for animations (for visualizing elastic modes, or deformations in time-domain dynamical analyses)

----

## Intended future features: 

### Meshes
* Tet10, Brick8, Brick27, Prism6, Prism18 elements support

### Materials
* All anisotropy classes for elastic materials
* Linear viscoelasticity using the Generalized Maxwell Model

### Structural matrices
* Arbitrary Dirichlet boundary conditions
* Lumped mass matrix

### Reduced-Order Modeling
* Proper Orthogonal Decomposition (POD) for nonlinear problems

### Forces
* Body forces
* Allowing for the definition of surface forces

### Solvers
* Frequency-domain dynamical analysis
* Newton-Raphson method for geometrically nonlinear statics and dynamics
* Arc-length method for geometrically nonlinear statics and dynamics with strong nonlinearities (e.g. post-buckling analysis)

### Post-processing
* Plotting of Frequency-Response Functions (FRF)

### Uncertainty Quantification




