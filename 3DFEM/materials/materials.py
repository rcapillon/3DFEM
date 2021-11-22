##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM code available at:                   #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import numpy as np

import importlib.util
spec1 = importlib.util.spec_from_file_location("random_generators", "../random_generators/random_generators.py")
rng = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(rng)

class Material:
    def __init__(self, mat_C, rho, id_number):
        self.__id = id_number
        
        self.__mean_mat_C = mat_C
        self.__mat_C = mat_C
        
        self.__mean_rho = rho
        self.__rho = rho
        
    def set_id(self, id_number):
        self.__id = id_number
        
    def get_id(self):
        return self.__id
    
    def set_mean_mat_C(self, mean_mat_C):
        self.__mean_mat_C = mean_mat_C
        
    def get_mean_mat_C(self):
        return self.__mean_mat_C
    
    def set_mat_C(self, mat_C):
        self.__mat_C = mat_C
        
    def get_mat_C(self):
        return self.__mat_C
    
    def set_mean_rho(self, mean_rho):
        self.__mean_rho = mean_rho
        
    def get_mean_rho(self):
        return self.__mean_rho
    
    def set_rho(self, rho):
        self.__rho = rho
        
    def get_rho(self):
        return self.__rho
    
####
# linear isotropic elastic material
####

factorized_mat_C_1 = np.eye(6)
factorized_mat_C_2 = np.array([[-1, 1, 1, 0, 0, 0],\
                               [1, -1, 1, 0, 0, 0],\
                               [1, 1, -1, 0, 0, 0],\
                               [0, 0, 0, -2, 0, 0],\
                               [0, 0, 0, 0, -2, 0],\
                               [0, 0, 0, 0, 0, -2]])

factorized_mats_C = [factorized_mat_C_1, factorized_mat_C_2]
n_factorized_mats_C = 2

class LinearIsotropicElasticMaterial(Material):
    def __init__(self, rho, Y, nu, id_number):
        # rho: mass density (kg/m^3)
        # Y: Young's modulus (Pa)
        # nu: Poisson coefficient
        
        self.__dispersion_coefficient_rho = 0
        
        self.__Y = Y
        self.__mean_Y = Y
        self.__dispersion_coefficient_Y = 0
        
        self.__nu = nu
        self.__mean_nu = nu
        
        self.__lame1 = (self.__Y * self.__nu) / ((1 + self.__nu) * (1 - 2*self.__nu))
        self.__mean_lame1 = self.__lame1
        self.__lame2 = self.__Y / (2 * (1 + self.__nu))
        self.__mean_lame2 = self.__lame2
        
        mat_C = self.compute_mat_C(self.__Y, self.__nu)
        
        self.__factorized_mats_C = factorized_mats_C
        self.__n_factorized_mats_C = n_factorized_mats_C
        
        super(LinearIsotropicElasticMaterial, self).__init__(mat_C, rho, id_number)
        
    def get_factorized_mat_C(self):
        return self.__factorized_mats_C
    
    def get_n_factorized_mat_C(self):
        return self.__n_factorized_mats_C
    
    def compute_factorized_coeffs(self):
        coeff = self.__Y / ((1 + self.__nu) * (1 - 2 * self.__nu))
        vec_coeffs = np.array([coeff, coeff * self.__nu])
        
        return vec_coeffs
        
    def compute_mat_C(self, Y, nu):
        lame1 = (Y * nu) / ((1 + nu) * (1 - 2*nu))
        lame2 = Y / (2 * (1 + nu))
        
        repmat_lame1 = np.tile(lame1, (3, 3))
        mat_C = 2 * lame2 * np.eye(6)
        mat_C[:3,:3] += repmat_lame1
        
        return mat_C
    
    def get_Y(self):
        return self.__Y
    
    def get_nu(self):
        return self.__nu
    
    def set_dispersion_coefficient_rho(self, dispersion_coefficient_rho):
        self.__dispersion_coefficient_rho = dispersion_coefficient_rho
        
    def set_dispersion_coefficient_Y(self, dispersion_coefficient_Y):
        self.__dispersion_coefficient_Y = dispersion_coefficient_Y
        
    def compute_random_elastic_coeffs(self):
        if self.__dispersion_coefficient_Y == 0:
            self.__Y_rand = self.__Y * np.ones((self.__n_samples,))
        else:
            self.__Y_rand = rng.scalars_gamma(self.__n_samples, self.__Y, self.__dispersion_coefficient_Y)
            
        self.__nu_rand = self.__nu * np.ones((self.__n_samples,))
    
    def get_Y_rand(self):
        return self.__Y_rand
    
    def get_nu_rand(self):
        return self.__nu_rand
    
    def set_random_elastic_coeffs(self, index):
        Y_rand = self.__Y_rand[index]
        nu_rand = self.__nu_rand[index]
        self.__Y = Y_rand
        self._nu = nu_rand
        
    def restore_elastic_coeffs(self):
        self.__Y = self.__mean_Y
        self.__nu = self.__mean_nu
        
    def compute_random_rho(self):
        if self.__dispersion_coefficient_rho == 0:
            self.__rho_rand = self.__rho * np.ones((self.__n_samples,))
        else:
            self.__rho_rand = rng.scalars_gamma(self.__n_samples, self.get_rho(), self.__dispersion_coefficient_rho)
        
    def get_rho_rand(self):
        return self.__rho_rand
    
    def set_random_rho(self, index):
        self.set_rho(self.__rho_rand[index])
        
    def restore_rho(self):
        self.set_rho(self.get_mean_rho())
        
    def compute_random_material(self):
        self.compute_random_elastic_coeffs()
        self.compute_random_rho()
        
    def set_random_material(self, index):
        self.set_random_elastic_coeffs(index)
        self.set_random_rho(index)
        
    def restore_material(self):
        self.restore_elastic_coeffs()
        self.restore_rho()
        
    def set_n_samples(self, n_samples):
        self.__n_samples = n_samples
            
