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
import scipy.sparse

import importlib.util
spec1 = importlib.util.spec_from_file_location("mesh", "../mesh/mesh.py")
mesh = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(mesh)

class Structure:
    def __init__(self, mesh):
        self.__mesh = mesh
        self.__n_total_dofs = self.__mesh.get_n_total_dofs()
        self.__n_free_dofs = len(self.__mesh.get_free_dofs())
        self.__n_dir_dofs = len(self.__mesh.get_dirichlet_dofs())
        self.__alphaM = 0
        self.__alphaK = 0
        
    def get_mesh(self):
        return self.__mesh
    
    def compute_M(self, symmetrization=False):    
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        
        for element in self.get_mesh().get_elements_list():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Me()
            
            vec_dataM = np.append(vec_dataM, element.get_mat_Me().flatten(order='F'))
            
        vec_rows = np.array(vec_rows)
        vec_cols = np.array(vec_cols)
        
        self.__mat_M = scipy.sparse.csr_matrix((vec_dataM, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
        
    def compute_K(self, symmetrization=False):    
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataK = np.array([])
        
        for element in self.get_mesh().get_elements_list():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Ke()
            
            vec_dataK = np.append(vec_dataK, element.get_mat_Ke().flatten(order='F'))
            
        vec_rows = np.array(vec_rows)
        vec_cols = np.array(vec_cols)
        
        self.__mat_K = scipy.sparse.csr_matrix((vec_dataK, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
                
    def compute_M_K(self, symmetrization=False):
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        vec_dataK = np.array([])
        
        for element in self.get_mesh().get_elements_list():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Me_mat_Ke()
            
            vec_dataM = np.append(vec_dataM, element.get_mat_Me().flatten(order='F'))
            vec_dataK = np.append(vec_dataK, element.get_mat_Ke().flatten(order='F'))
            
        self.__mat_M = scipy.sparse.csr_matrix((vec_dataM, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
        
        self.__mat_K = scipy.sparse.csr_matrix((vec_dataK, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
        
    def set_rayleigh(self, alphaM, alphaK):
        self.__alphaM = alphaM
        self.__alphaK = alphaK
        
    def get_alphaM(self):
        return self.__alphaM
    
    def get_alphaK(self):
        return self.__alphaK
        
    def compute_D(self):
        self.__mat_D = self.__alphaM * self.__mat_M + self.__alphaK * self.__mat_K
        
    def get_M(self):
        return self.__mat_M
    
    def get_K(self):
        return self.__mat_K
    
    def get_D(self):
        return self.__mat_D
    
    def apply_dirichlet_M(self):
        self.__mat_MLL = self.__mat_M[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_free_dofs()]
        self.__mat_MLD = self.__mat_M[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_dirichlet_dofs()]
        
    def apply_dirichlet_K(self):
        self.__mat_KLL = self.__mat_K[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_free_dofs()]
        self.__mat_KLD = self.__mat_K[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_dirichlet_dofs()]
        
    def apply_dirichlet_D(self):
        self.__mat_DLL = self.__mat_D[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_free_dofs()]
        self.__mat_DLD = self.__mat_D[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_dirichlet_dofs()]
                
    def get_MLL(self):
        return self.__mat_MLL
    
    def get_MLD(self):
        return self.__mat_MLD
    
    def get_KLL(self):
        return self.__mat_KLL
    
    def get_KLD(self):
        return self.__mat_KLD
    
    def get_DLL(self):
        return self.__mat_DLL
    
    def get_DLD(self):
        return self.__mat_DLD
    
    def set_U0L(self, vec_U0L=None):
        if vec_U0L == None:
            self.__vec_U0L = np.zeros((self.__n_free_dofs,))
        else:
            self.__vec_U0L = vec_U0L
        
    def get_U0L(self):
        return self.__vec_U0L
    
    def set_V0L(self, vec_V0L=None):
        if vec_V0L == None:
            self.__vec_V0L = np.zeros((self.__n_free_dofs,))
        else:
            self.__vec_V0L = vec_V0L
        
    def get_V0L(self):
        return self.__vec_V0L
    
    def set_A0L(self, vec_A0L=None):
        if vec_A0L == None:
            self.__vec_A0L = np.zeros((self.__n_free_dofs,))
        else:
            self.__vec_A0L = vec_A0L
        
    def get_A0L(self):
        return self.__vec_A0L