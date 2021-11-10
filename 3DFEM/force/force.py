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
spec1 = importlib.util.spec_from_file_location("mesh", "./mesh/mesh.py")
mesh = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(mesh)

class Force:
    def __init__(self, mesh):
        self.__mesh = mesh
        self.__ls_nodal_forces_t0 = []
        self.__vec_variation = np.array([0])
        
    def get_mesh(self):
        return self.__mesh
    
    def add_nodal_forces_t0(self, ls_nodes, nodal_force_vector):
        self.__ls_nodal_forces_t0.append((ls_nodes, nodal_force_vector))
        
    def get_nodal_forces_t0(self):
        return self.__ls_nodal_forces_t0
    
    def compute_F0(self):
        n_total_dofs = self.__mesh.get_n_total_dofs()
        
        self.__vec_F0 = np.zeros((n_total_dofs,))
        
        for group in self.get_nodal_forces_t0():
            nodes = group[0]
            force_vector = group[1]
            
            ls_dofs_x = [node * 3 for node in nodes]
            ls_dofs_y = [node * 3 + 1 for node in nodes]
            ls_dofs_z = [node * 3 + 2 for node in nodes]
            
            self.__vec_F0[ls_dofs_x] += np.repeat(force_vector[0], len(ls_dofs_x))
            self.__vec_F0[ls_dofs_y] += np.repeat(force_vector[1], len(ls_dofs_y))
            self.__vec_F0[ls_dofs_z] += np.repeat(force_vector[2], len(ls_dofs_z))
            
    def compute_constant_F(self, n_timesteps):        
        self.__mat_F = np.tile(self.__vec_F0, n_timesteps)
        
    def set_F_variation(self, vec_variation):
        self.__vec_variation = vec_variation
        
    def compute_varying_F(self):
        n_timesteps = len(self.__vec_variation)
        
        self.__mat_F = np.zeros((self.__vec_F0.shape[0], n_timesteps))
        
        for ii in range(n_timesteps):
            vec_F_ii = self.__vec_F0 * self.__vec_variation[ii]
            self.__mat_F[:, ii] = vec_F_ii
            
    def apply_dirichlet_F0(self):
        self.__vec_F0L = self.__vec_F0[self.__mesh.get_free_dofs()]
        self.__vec_F0D = self.__vec_F0[self.__mesh.get_dirichlet_dofs()]
        
    def get_F0L(self):
        return self.__vec_F0L
    
    def get_F0D(self):
        return self.__vec_F0D
    
    def apply_dirichlet_F(self):
        self.__mat_FL = self.__mat_F[self.__mesh.get_free_dofs(), :]
        self.__mat_FD = self.__mat_F[self.__mesh.get_dirichlet_dofs(), :]
        
    def get_FL(self):
        return self.__mat_FL
    
    def get_FD(self):
        return self.__mat_FD