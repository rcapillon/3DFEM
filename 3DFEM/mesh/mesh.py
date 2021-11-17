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
import scipy.spatial

import importlib.util
spec1 = importlib.util.spec_from_file_location("tet4", "../elements/tet4.py")
tet4 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(tet4)

class Mesh:
    def __init__(self, points):
        self.__n_points = points.shape[0]
        self.__n_total_dofs = self.__n_points * 3
        self.__points = points
        self.__ls_dofs_dir = [] # zero-dirichlet condition
        self.__ls_dofs_free = [ii for ii in range(self.__n_total_dofs)]
        self.__ls_dofs_observed = []
        self.__n_dofs_observed = 0
    
    def set_elements_list(self, elem_list):
        self.__n_elements = len(elem_list)
        self.__elements_list = elem_list
        
    def delaunay3D_from_points(self, rho, Y, nu):
        tri = scipy.spatial.Delaunay(self.__points)
        self.__points = tri.points
        self.__n_elements = tri.simplices.shape[0]
                
        self.__elements_list = []
        
        elem_counter = 0
        for ii in range(self.__n_elements):
            nodes_ii = tri.simplices[ii, :]
            nodes_coords = self.__points[nodes_ii, :]
            
            P1 = nodes_coords[0,:]
            P2 = nodes_coords[1,:]
            P3 = nodes_coords[2,:]
            P4 = nodes_coords[3,:]
            
            mixed_product = np.dot(np.cross((P2-P1), (P3-P1)), (P4-P1))
            
            if mixed_product < 0:
                nodes_ii[[0, 1]] = nodes_ii[[1, 0]]
                nodes_coords = self.__points[nodes_ii,:]
                
            if mixed_product != 0:
                element_ii = tet4.Tet4(rho, Y, nu, nodes_coords)
                element_ii.set_element_number(elem_counter)
                element_ii.set_nodes_dofs(nodes_ii)
                self.__elements_list.append(element_ii)
                
    def add_UL_to_points(self, UL):    
        U = np.zeros((self.__n_total_dofs,))
        U[self.__ls_dofs_free] = UL
        
        self.__points[:,0] += U[::3]
        self.__points[:,1] += U[1::3]
        self.__points[:,2] += U[2::3]
        
    def add_U_to_points(self, U):    
        self.__points[:,0] += U[::3]
        self.__points[:,1] += U[1::3]
        self.__points[:,2] += U[2::3]
                
    def get_n_points(self):
        return self.__n_points
    
    def get_n_total_dofs(self):
        return self.__n_total_dofs
    
    def get_points(self):
        return self.__points
    
    def get_n_elements(self):
        return self.__n_elements
    
    def get_elements_list(self):
        return self.__elements_list
    
    def set_dirichlet(self, ls_dofs_dir):
        self.__ls_dofs_dir = ls_dofs_dir
        self.__ls_dofs_free = list(set(range(self.__n_total_dofs)) - set(ls_dofs_dir))
        
    def get_dirichlet_dofs(self):
        return self.__ls_dofs_dir
    
    def get_free_dofs(self):
        return self.__ls_dofs_free
    
    def set_observed_dofs(self, ls_dofs_observed):
        self.__ls_dofs_observed = ls_dofs_observed
        self.__n_dofs_observed = len(ls_dofs_observed)
        
    def add_observed_dofs(self, dof_observed):
        self.__ls_dofs_observed.append(dof_observed)
        self.__n_dofs_observed += 1
        
    def get_observed_dofs(self):
        return self.__ls_dofs_observed
    
    def get_n_observed_dofs(self):
        return self.__n_dofs_observed