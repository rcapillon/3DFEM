##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM library available at:                #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import numpy as np

class IsotropicElasticMaterial:
    def __init__(self, rho, Y, nu):
        self.set_properties(rho, Y, nu)
        
    def set_properties(self, rho, Y, nu):
        # rho: mass density (kg/m^3)
        # Y: Young's modulus (Pa)
        # nu: Poisson coefficient
        
        self.__rho = rho
        self.__Y = Y
        self.__nu = nu
        
        self.__lame1 = (self.__Y * self.__nu) / ((1 + self.__nu) * (1 - 2*self.__nu))
        self.__lame2 = self.__Y / (2 * (1 + self.__nu))
        
        repmat_lame1 = np.tile(self.__lame1, (3, 3))
        self.__mat_C = 2 * self.__lame2 * np.eye(6)
        self.__mat_C[:3,:3] += repmat_lame1
        
    def get_rho(self):
        return self.__rho
    
    def get_Y(self):
        return self.__Y
    
    def get_nu(self):
        return self.__nu
        
    def get_mat_C(self):
        return self.__mat_C