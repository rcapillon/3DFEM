import numpy as np
from random_generators.scalars import gamma

factorized_mat_C_1 = np.eye(6)
factorized_mat_C_2 = np.array([[-1, 1, 1, 0, 0, 0],
                               [1, -1, 1, 0, 0, 0],
                               [1, 1, -1, 0, 0, 0],
                               [0, 0, 0, -2, 0, 0],
                               [0, 0, 0, 0, -2, 0],
                               [0, 0, 0, 0, 0, -2]])

list_factorized_mat_C = [factorized_mat_C_1, factorized_mat_C_2]
n_factorized_mat_C = 2


def compute_mat_C(Y, nu):
    lame1 = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
    lame2 = Y / (2 * (1 + nu))

    repmat_lame1 = np.tile(lame1, (3, 3))
    mat_C = 2 * lame2 * np.eye(6)
    mat_C[:3, :3] += repmat_lame1

    return mat_C


class LinearElasticIsotropic:
    def __init__(self, id_number, rho, Y, nu):
        # rho: mass density (kg/m^3)
        # Y: Young's modulus (Pa)
        # nu: Poisson ratio

        self.id = id_number
        self.type = 'Linear Elastic Isotropic'

        self.rho = rho
        self.mean_rho = rho

        self.Y = Y
        self.mean_Y = Y
        self.dispersion_coefficient_Y = 0.0

        self.nu = nu
        self.mean_nu = nu

        self.dict_of_properties = {
            'Mass density (kg/m^3)': self.mean_rho,
            'Young''s modulus (Pa)': self.mean_Y,
            'Poisson ratio': self.mean_nu
        }

        self.mat_C = compute_mat_C(Y, nu)

        self.list_factorized_mat_C = list_factorized_mat_C
        self.n_factorized_mat_C = n_factorized_mat_C
        self.list_factorized_coeffs = self.compute_factorized_coeffs()

    def set_dispersion_coefficient_Y(self, dispersion_coefficient_Y):
        self.dispersion_coefficient_Y = dispersion_coefficient_Y

    def compute_factorized_coeffs(self):
        coeff = self.Y / ((1 + self.nu) * (1 - 2 * self.nu))
        list_factorized_coeffs = [coeff, coeff * self.nu]

        return list_factorized_coeffs

    def generate_random_material_parameters(self):
        if self.dispersion_coefficient_Y == 0.0:
            self.Y = self.mean_Y
        else:
            self.Y = gamma(self.mean_Y, self.dispersion_coefficient_Y)

        self.rho = self.mean_rho
        self.nu = self.mean_nu

    def restore_mean_material_parameters(self):
        self.rho = self.mean_rho
        self.Y = self.mean_Y
        self.nu = self.mean_nu
