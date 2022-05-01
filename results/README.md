# Results from sample script files

## static_beam

*Static analysis of a clamped-clamped linear elastic beam subjected to a downward nodal force at the center of the upper face.*

Deformed mesh
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/static_beam/image_static_beam.png" width="400"> |

## linear_elastic_stress_strain

*Static analysis of a cantilever linear elastic beam subjected to a surface force at its free end, along the beam's main axis. Axial stress and strain are computed.*

Stress-strain curve
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/linear_elastic_stress_strain/stress_strain_linear_static_beam_3004_0.png" width="400"> |

## modal_beam

*Computation of the first 10 elastic modes of a clamped-clamped linear elastic beam.*

10th elastic mode
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/modal_beam/animation_modal_beam.gif" width="400"> |

## freq_plate

*Frequency-domain dynamical analysis of a linear elastic plate subjected to a unit nodal force at the center of its upper face for all frequencies. All other faces are blocked.*

Deformed mesh at 16660 Hz
:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/image_mesh_16660Hz.png" width="400"> |

FRF at 1st observed node, x-displacement | FRF at 1st observed node, y-displacement | FRF at 1st observed node, z-displacement
:----:|:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/frf_DOF1.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/frf_DOF2.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/frf_DOF3.png" width="400"> |

FRF at 2nd observed node, x-displacement | FRF at 2nd observed node, y-displacement | FRF at 2nd observed node, z-displacement
:----:|:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/frf_DOF4.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/frf_DOF5.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/freq_plate/frf_DOF6.png" width="400"> |

## UQ_freq_plate

*Frequency-domain dynamical analysis of a linear elastic plate subjected to a unit nodal force at the center of its upper face for all frequencies. All other faces are blocked.
Probabilistic model for the Young's modulus (parametric uncertainty modeling) as well as the mass, stiffness and damping matrices (nonparametric uncertainty modeling).*

### FRF at 1st observed node, y-displacement

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/frf_DOF2_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/frf_DOF2_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/frf_DOF2_3generalized.png" width="400"> |

### Probability density function at 1st observed node, y-displacement, 14660 Hz

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/gkde_DOF2_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/gkde_DOF2_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/gkde_DOF2_3generalized.png" width="400"> |

### FRF at 2nd observed node, x-displacement

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
:----:|:----:|:----:
<img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/frf_DOF4_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/frf_DOF4_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/frf_DOF4_3generalized.png" width="400"> |

### Probability density function at 2nd observed node, x-displacement, 14660 Hz

| Parametric uncertainty modeling | Nonparametric uncertainty modeling | Generalized approach |
|----|----|----|
| <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/gkde_DOF4_1parametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/gkde_DOF4_2nonparametric.png" width="400"> | <img src="https://github.com/rcapillon/3DFEM/blob/main/results/UQ_freq_plate/gkde_DOF4_3generalized.png" width="400"> |

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
