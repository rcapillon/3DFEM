import numpy as np


def gamma(mean_value, dispersion_coefficient):
    if dispersion_coefficient != 0:
        gamma_shape = 1.0 / dispersion_coefficient ** 2
        gamma_scale = mean_value * dispersion_coefficient ** 2
        random_scalar = np.random.gamma(gamma_shape, gamma_scale)
    else:
        random_scalar = mean_value

    return random_scalar
