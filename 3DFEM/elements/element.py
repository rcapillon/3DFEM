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
spec1 = importlib.util.spec_from_file_location("materials", "../materials/materials.py")
materials = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(materials)

class Element(materials.LinearIsotropicElasticMaterial):
    def __init__(self, rho, Y, nu):
        super(Element, self).__init__(rho, Y, nu)
    
    def set_element_number(self, num):
        self.__num = num
        
    def get_element_number(self):
        return self.__num
    
    def set_nodes_dofs(self, nodes_nums):
        self.__nodes_nums = nodes_nums
        self.__dofs_nums = np.array([], dtype=np.int32)
        for node in self.__nodes_nums:
            self.__dofs_nums = np.append(self.__dofs_nums,[node * 3, node * 3 + 1, node * 3 + 2])
        
    def get_nodes_nums(self):
        return self.__nodes_nums
    
    def get_dofs_nums(self):
        return self.__dofs_nums