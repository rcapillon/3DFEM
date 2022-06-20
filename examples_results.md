# Results from the example scripts (one for each available solver)

## linear_statics_example

Undeformed mesh | Deformed mesh |
:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/undeformedmesh_linear_statics_example.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/deformedmesh_linear_statics_example.png" width="400"> |

## modal_example

10th elastic mode |
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_modal_example.gif" width="400"> |

## nonlinear_statics_newtonraphson_example

Load-displacement curve | Mesh deformation |
:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_norm_U_nonlinear_statics_newtonraphson_example.gif" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_nonlinear_statics_newtonraphson_example.gif" width="400"> |

## nonlinear_statics_arclength_example

Load-displacement curve | Mesh deformation |
:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_norm_U_nonlinear_statics_arclength_example.gif" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_nonlinear_statics_arclength_example.gif" width="400"> |

## linear_frequency_example

Frequency response at certain DOFs |
:----:|:----:|
<img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/plot_linear_frequency_example6674.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/plot_linear_frequency_example12852.png" width="400"> |

Deformed mesh as the frequency spectrum is sweeped |
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/readme_files/animation_linear_frequency_example.gif" width="400"> |

## linear_frequency_UQ_example

## time_beam

*Time-domain dynamical analysis of a clamped-free linear elastic beam subjected to a nodal force at the center of its free end. The force linearly increases to a maximum value, stays constant a while and then vanishes.*

Deformed mesh
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/animation_time_beam.gif" width="400"> |

1st observed node, x-displacement | 1st observed node, y-displacement | 1st observed node, z-displacement
:----:|:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/U_DOF1.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/U_DOF2.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/U_DOF3.png" width="400"> |

2nd observed node, x-displacement | 2nd observed node, y-displacement | 2nd observed node, z-displacement
:----:|:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/U_DOF4.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/U_DOF5.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/time_beam/U_DOF6.png" width="400"> |

## UQ_time_beam

*Time-domain dynamical analysis of a clamped-free linear elastic beam subjected to a nodal force at the center of its free end. The force linearly increases to a maximum value, stays constant a while and then vanishes.
Probabilistic model for the Young's modulus (parametric uncertainty modeling) as well as the mass, stiffness and damping matrices (nonparametric uncertainty modeling).*

### 1st observed node, y-displacement

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF5_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF5_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF5_3generalized.png" width="400"> |

### Probability density function at 1st observed node, y-displacement, 0.1125 seconds

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_gkde_DOF5_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_gkde_DOF5_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_gkde_DOF5_3generalized.png" width="400"> |

### 2nd observed node, z-displacement

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF6_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF6_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_DOF6_3generalized.png" width="400"> |

### Probability density function at 2nd observed node, z-displacement, 0.1125 seconds

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_gkde_DOF6_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_gkde_DOF6_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_time_beam/U_gkde_DOF6_3generalized.png" width="400"> |
