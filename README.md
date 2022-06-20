# 3DFEM
## Python 3D finite element code for structural dynamics with uncertainty modeling

This python library allows for solving 3D structural problems with or without uncertainties using the finite element method.
New features will be added over time.
Documentation for the library will be added. For now, one should look at the examples (one for each available solver) to see how to run a specific simulation.

*Solvers have **NOT** been validated on reference cases yet.*

## Requirements and installation
Create a virtual environment with a Python3 distribution and install the requirements listed in the file 'requirements.txt'.
Clone this repository in the folder of your choosing and run 'setup.py' in that directory using a terminal:

```
python setup.py install
```

Then, you're good to go. You can try the sample study scripts in the 'examples' directory, for instance:

```
cd examples/
python linear_statics_example.py 
```

All examples will create a 'plots' and/or a 'vtk_files' directory where they are executed with the results files. 

Paraview is suggested for visualizing exported VTK files, especially since animations are generated as a series of VTK files as handled by this software.

----

## Examples of solutions

Visuals of results from each example script can be found [here](https://github.com/rcapillon/3DFEM/blob/main/examples_results.md).

| Time-domain deterministic simulation| Frequency-domain stochastic simulation |
|:----:|:----:|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_linear_newmark_example.gif" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/plot_U_rand_linear_frequency_UQ_nonparametric_example6673.png" width="400"> |

----

## Current features:

### Meshes
* Tetrahedral mesh generation from a set of points using scipy.spatial.Delaunay
* 4-node tetrahedral (Tet4), 6-node prism (Prism6), 8-node brick (Brick8) elements support
* Support for meshes containing different types of elements, possibly of different orders
* Merging of two meshes by fusing nodes whose coordinates are equal up to a certain tolerance
* Automatic mesh generation of basic shapes: for now, rectangular bricks and beams with a circular cross-section are available

### Materials
* Linear isotropic elastic materials

### Structural matrices
* Consistent mass matrix
* Linear elastic stiffness matrix
* Rayleigh damping matrix (linear combination of mass and stiffness matrices)
* Support only for zero-Dirichlet boundary conditions
* Full Gauss quadrature scheme

### Boundary conditions
* Only zero-valued Dirichlet conditions are supported for now
* Nodal forces as Neumann conditions
* Support for modulation of the initial force vector by a given function over time steps or frequency steps in dynamical analyses

### Reduced-Order Modeling
* Projection on linear elastic modes

### Solvers
* Modal analysis
* Linear statics analysis
* Linear frequency-domain dynamics using a reduced-order model based on elastic modes
* Linear time-domain dynamics using the Newmark scheme and a reduced-order model based on elastic modes
* Nonlinear statics analysis using the Newton-Raphson method
* Nonlinear statics analysis using the Arc-Length method
* Uncertainty Quantification:
  - Parametric probabilistic model for the Young's modulus in dynamics solvers
  - Nonparametric probabilistic models for reduced matrices in dynamics solvers
  - Generalized probabilistic approach, including both parametric and nonparametric uncertainties, in dynamics solvers
  - Direct Monte Carlo method for uncertainty propagation

### Post-processing
* Plotting of Frequency-Response Functions (FRF), time trajectories
* Plotting of confidence intervals on FRF and time trajectories in stochastic simulations
* Plotting of probability density functions at a given step using gaussian kernel density estimation
* Export of a deformed mesh to VTK format (PolyData legacy format)
* Export of a series of deformed meshes to VTK format for animations (elastic modes, time-domain, frequency-domain and nonlinear statics solutions)

----

## Features currently being worked on:
* Computation of Cauchy stresses and linearized strains in elements 
* Display of stresses and strains on faces of elements in VTK files
* Code commentary and documentation

----

## Intended future features: 

### Meshes
* Support for Tet10, Prism18, Brick27 elements (including splitting of existing meshes using Tet4, Prism6 and Brick8 elements

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
* Uncertainty Quantification:
  - Parametric probabilistic models for all material coefficients
  - Causal nonparametric probabilistic models for linear viscoelastic materials
* Parallelization of stochastic solvers

### Post-processing
* Plotting of modal coordinates at given step
* Computation of Cauchy stress tensor and linearized deformation tensor components on a deformed mesh.
* Support for including element stress and strain components in the VTK files
