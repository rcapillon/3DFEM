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
from scipy.sparse.linalg import eigsh

import importlib.util
spec1 = importlib.util.spec_from_file_location("mesh", "../mesh/mesh.py")
mesh = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(mesh)

import importlib.util
spec2 = importlib.util.spec_from_file_location("random_generators", "../random_generators/random_generators.py")
rng = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rng)

class Structure:
    def __init__(self, mesh):
        self.__mesh = mesh
        self.__n_total_dofs = self.__mesh.get_n_total_dofs()
        self.__n_free_dofs = len(self.__mesh.get_free_dofs())
        self.__n_dir_dofs = len(self.__mesh.get_dirichlet_dofs())
        self.__alphaM = 0
        self.__alphaK = 0
        self.__dispersion_coefficient_M = 0
        self.__dispersion_coefficient_K = 0
        self.__n_samples = 0
        
    def get_n_total_dofs(self):
        return self.__n_total_dofs
    
    def get_n_free_dofs(self):
        return self.__n_free_dofs
    
    def get_n_dir_dofs(self):
        return self.__n_dir_dofs
        
    def get_mesh(self):
        return self.__mesh
    
    def compute_M(self, symmetrization=False):    
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        
        for element in self.__mesh.get_elements_list():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Me()
            
            vec_dataM = np.append(vec_dataM, element.get_mat_Me().flatten(order='F'))
        
        self.__mat_M = scipy.sparse.csr_matrix((vec_dataM, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
        
    def compute_K(self, symmetrization=False):    
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataK = np.array([])
        
        for element in self.__mesh.get_elements_list():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Ke()
            
            vec_dataK = np.append(vec_dataK, element.get_mat_Ke().flatten(order='F'))
        
        self.__mat_K = scipy.sparse.csr_matrix((vec_dataK, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
                
    def compute_M_K(self, symmetrization=False):
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        vec_dataK = np.array([])
        
        for element in self.__mesh.get_elements_list():
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
        self.__mat_K = scipy.sparse.csr_matrix((vec_dataK, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
            
    def compute_factorized_M_vectors(self):
        self.__mesh.compute_sub_elements_lists()
        
        self.__list_factorized_M_vectors = []
        self.__list_factorized_K_vectors = []
                
        for ls in self.__mesh.get_list_of_elements_lists():
            vec_rows = np.array([])
            vec_cols = np.array([])
            mat_M_vec_data = np.array([])
                                    
            for element in ls:
                n_dofs = element.get_n_dofs()
                
                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)
                                    
                vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
                vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
                
                element.compute_factorized_mat_Me()
                
                mat_M_vec_data = np.append(mat_M_vec_data, element.get_factorized_mat_Me().flatten(order='F'))
                
            self.__list_factorized_M_vectors.append((vec_rows, vec_cols, mat_M_vec_data))
        
    def compute_factorized_K_vectors(self):
        self.__mesh.compute_sub_elements_lists()
        
        self.__list_factorized_K_vectors = []
                
        for ls in self.__mesh.get_list_of_elements_lists():
            vec_rows = np.array([])
            vec_cols = np.array([])
            mat_K_vec_data = np.array([])
            
            n_coeffs = ls[0].get_material().get_n_factorized_mat_C()
            
            list_factorized_mat_K_vec_data = [np.array([]) for jj in range(n_coeffs)]
            
            for element in ls:
                n_dofs = element.get_n_dofs()
                
                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)
                                    
                vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
                vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
                
                element.compute_factorized_mat_Ke()
                
                list_factorized_mat_Ke = element.get_list_factorized_mat_Ke()
                
                print("list_factorized_mat_Ke")
                print(len(list_factorized_mat_Ke))
                
                for ii in range(n_coeffs):
                    mat_Ke = list_factorized_mat_Ke[ii]
                    mat_K_vec_data = mat_Ke.flatten(order='F')
                    list_factorized_mat_K_vec_data[ii] = np.append(list_factorized_mat_K_vec_data[ii], mat_K_vec_data)
                    
            self.__list_factorized_K_vectors.append((vec_rows, vec_cols, list_factorized_mat_K_vec_data))
        
    def compute_factorized_M_K_vectors(self):
        self.__mesh.compute_sub_elements_lists()
        
        self.__list_factorized_M_vectors = []
        self.__list_factorized_K_vectors = []
                
        for ls in self.__mesh.get_list_of_elements_lists():
            vec_rows = np.array([])
            vec_cols = np.array([])
            mat_M_vec_data = np.array([])
            mat_K_vec_data = np.array([])
            
            n_coeffs = ls[0].get_material().get_n_factorized_mat_C()
            
            list_factorized_mat_K_vec_data = [np.array([]) for jj in range(n_coeffs)]
            
            for element in ls:
                n_dofs = element.get_n_dofs()
                
                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)
                                    
                vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
                vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
                
                element.compute_factorized_mat_Me_mat_Ke()
                
                list_factorized_mat_Ke = element.get_list_factorized_mat_Ke()
                
                for ii in range(n_coeffs):
                    mat_Ke = list_factorized_mat_Ke[ii]
                    mat_K_vec_data = mat_Ke.flatten(order='F')
                    list_factorized_mat_K_vec_data[ii] = np.append(list_factorized_mat_K_vec_data[ii], mat_K_vec_data)
                                
                mat_M_vec_data = np.append(mat_M_vec_data, element.get_factorized_mat_Me().flatten(order='F'))
                
            self.__list_factorized_M_vectors.append((vec_rows, vec_cols, mat_M_vec_data))
            self.__list_factorized_K_vectors.append((vec_rows, vec_cols, list_factorized_mat_K_vec_data))
        
    def compute_factorized_M(self, symmetrization=False):
        vec_rows = np.array([])
        vec_cols = np.array([])
        mat_M_vec_data = np.array([])
        
        list_of_elements_lists = self.__mesh.get_list_of_elements_lists()
        n_materials = len(list_of_elements_lists)
        
        for ii in range(n_materials):
            rho_ii = list_of_elements_lists[ii][0].get_material().get_rho()
            
            (vec_rows_ii, vec_cols_ii, mat_M_vec_data_ii) = self.__list_factorized_M_vectors[ii]
            vec_rows = np.append(vec_rows, vec_rows_ii)
            vec_cols = np.append(vec_cols, vec_cols_ii)
            mat_M_vec_data = np.append(mat_M_vec_data, rho_ii * mat_M_vec_data_ii)
        
        self.__mat_M = scipy.sparse.csr_matrix((mat_M_vec_data, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
        
    def compute_factorized_K(self, symmetrization=False):        
        vec_rows = np.array([])
        vec_cols = np.array([])
        mat_K_vec_data = np.array([])
        
        list_of_elements_lists = self.__mesh.get_list_of_elements_lists()
        n_materials = len(list_of_elements_lists)
        
        for ii in range(n_materials):
            coeffs_ii = list_of_elements_lists[ii][0].get_material().compute_factorized_coeffs()
            
            (vec_rows_ii, vec_cols_ii, list_factorized_mat_K_vec_data_ii) = self.__list_factorized_K_vectors[ii]
            vec_rows = np.append(vec_rows, vec_rows_ii)
            vec_cols = np.append(vec_cols, vec_cols_ii)
            mat_K_vec_data_ii = np.zeros(vec_rows_ii.shape)
            
            for jj in range(len(list_factorized_mat_K_vec_data_ii)):
                mat_K_vec_data_ii += coeffs_ii[jj] * list_factorized_mat_K_vec_data_ii[jj]
                
            mat_K_vec_data = np.append(mat_K_vec_data, mat_K_vec_data_ii)
        
        self.__mat_K = scipy.sparse.csr_matrix((mat_K_vec_data, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
        
    def compute_factorized_M_K(self, symmetrization=False):
        vec_rows = np.array([])
        vec_cols = np.array([])
        mat_M_vec_data = np.array([])
        mat_K_vec_data = np.array([])
        
        list_of_elements_lists = self.__mesh.get_list_of_elements_lists()
        n_materials = len(list_of_elements_lists)
        
        for ii in range(n_materials):
            rho_ii = list_of_elements_lists[ii][0].get_material().get_rho()
            coeffs_ii = list_of_elements_lists[ii][0].get_material().compute_factorized_coeffs()
                        
            (vec_rows_ii, vec_cols_ii, mat_M_vec_data_ii) = self.__list_factorized_M_vectors[ii]
            mat_M_vec_data = np.append(mat_M_vec_data, rho_ii * mat_M_vec_data_ii)
            
            (vec_rows_ii, vec_cols_ii, list_factorized_mat_K_vec_data_ii) = self.__list_factorized_K_vectors[ii]
                        
            mat_K_vec_data_ii = np.zeros(vec_rows_ii.shape)
            
            for jj in range(len(list_factorized_mat_K_vec_data_ii)):
                mat_K_vec_data_ii += coeffs_ii[jj] * list_factorized_mat_K_vec_data_ii[jj]
                
            mat_K_vec_data = np.append(mat_K_vec_data, mat_K_vec_data_ii)
            
            vec_rows = np.append(vec_rows, vec_rows_ii)
            vec_cols = np.append(vec_cols, vec_cols_ii)
        
        self.__mat_M = scipy.sparse.csr_matrix((mat_M_vec_data, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        self.__mat_K = scipy.sparse.csr_matrix((mat_K_vec_data, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
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
    
    def compute_modes(self, n_modes):
        # Eigenvectors are mass-normalized
        
        self.__n_modes = n_modes
        self.compute_factorized_M_K_vectors()
        self.compute_factorized_M_K()
        # self.compute_M_K()
        self.apply_dirichlet_M()
        self.apply_dirichlet_K()
        
        (eigvals, eigvects) = eigsh(self.__mat_KLL, n_modes, self.__mat_MLL, which='SM')
        sort_indices = np.argsort(eigvals)
        eigvals = eigvals[sort_indices]
        eigvects = eigvects[:, sort_indices]
        
        self.__eigenfreqs = np.sqrt(eigvals) / (2 * np.pi)
        self.__modesL = eigvects
        self.__modes = np.zeros((self.__n_total_dofs, n_modes))
        self.__modes[self.__mesh.get_free_dofs(), :] = eigvects
        
    def get_eigenfreqs(self):
        return self.__eigenfreqs
    
    def get_modesL(self):
        return self.__modesL
    
    def get_modes(self):
        return self.__modes
    
    def get_n_modes(self):
        return self.__n_modes
        
    def compute_linear_ROM(self):        
        self.__Mrom = np.dot(self.__modesL.transpose(), self.__mat_MLL.dot(self.__modesL))
        self.__Krom = np.dot(self.__modesL.transpose(), self.__mat_KLL.dot(self.__modesL))
        self.__Drom = self.__alphaM * self.__Mrom + self.__alphaK * self.__Krom
        
        if self.__dispersion_coefficient_M > 0:
            self.__cholesky_Mrom = np.linalg.cholesky(self.__Mrom)
        if self.__dispersion_coefficient_K > 0:
            self.__cholesky_Krom = np.linalg.cholesky(self.__Krom)
        if self.__dispersion_coefficient_M > 0 or self.__dispersion_coefficient_K > 0:
            self.__cholesky_Drom = np.linalg.cholesky(self.__Drom)
        
    def compute_linear_diagonal_ROM(self):        
        self.__Mrom = np.ones((self.__n_modes,))
        self.__Krom = np.power(2 * np.pi * self.__eigenfreqs, 2)
        self.__Drom = self.__alphaM * self.__Mrom + self.__alphaK * self.__Krom
        
    def get_Mrom(self):
        return self.__Mrom
    
    def get_Krom(self):
        return self.__Krom
    
    def get_Drom(self):
        return self.__Drom
    
    def set_n_samples(self, n_samples):
        self.__n_samples = n_samples
        
    def get_n_samples(self):
        return self.__n_samples
    
    def set_dispersion_coefficient_M(self, dispersion_coefficient_M):
        self.__dispersion_coefficient_M = dispersion_coefficient_M
        
    def set_dispersion_coefficient_K(self, dispersion_coefficient_K):
        self.__dispersion_coefficient_K = dispersion_coefficient_K
        
    def get_dispersion_coefficient_M(self):
        return self.__dispersion_coefficient_M
    
    def get_dispersion_coefficient_K(self):
        return self.__dispersion_coefficient_K
        
    def generate_random_M(self, n_samples):
        Mrom_rand = rng.matrices_SEplus(n_samples, self.__cholesky_Mrom, self.__dispersion_coefficient_M)
        
        return Mrom_rand
        
    def generate_random_K(self, n_samples):
        Krom_rand = rng.matrices_SEplus(n_samples, self.__cholesky_Krom, self.__dispersion_coefficient_K)
        
        return Krom_rand
        
    def compute_Drom(self, Mrom, Krom):
        Drom_rand = self.__alphaM * Mrom + self.__alphaK * Krom
        
        return Drom_rand
        
    def generate_random_matrices(self, uncertainty_type="nonparametric"):
        if uncertainty_type == "parametric":
            self.__mesh.compute_random_materials()
            self.__Mrom_rand = np.zeros(self.__Mrom.shape + (self.__n_samples,))
            self.__Krom_rand = np.zeros(self.__Krom.shape + (self.__n_samples,))
            self.__Drom_rand = np.zeros(self.__Drom.shape + (self.__n_samples,))
                        
            for ii in range(self.__n_samples):
                self.__mesh.set_random_materials(ii)
                self.compute_factorized_M_K()
                self.apply_dirichlet_M()
                self.apply_dirichlet_K()
                self.compute_linear_ROM()
                
                Mrom_rand_ii = self.get_Mrom()
                Krom_rand_ii = self.get_Krom()
                Drom_rand_ii = self.compute_Drom(Mrom_rand_ii, Krom_rand_ii)
                
                self.__Mrom_rand[:, :, ii] = Mrom_rand_ii
                self.__Krom_rand[:, :, ii] = Krom_rand_ii
                self.__Drom_rand[:, :, ii] = Drom_rand_ii
                
            self.__mesh.restore_materials()
        
        elif uncertainty_type == "nonparametric":
            if self.__dispersion_coefficient_M > 0:
                self.__Mrom_rand = self.generate_random_M(self.__n_samples)
            else:
                self.__Mrom_rand = np.tile(self.__Mrom, (1, 1, self.__n_samples))
                
            if self.__dispersion_coefficient_K > 0:
                self.__Krom_rand = self.generate_random_K(self.__n_samples)
            else:
                self.__Krom_rand = np.tile(self.__Krom, (1, 1, self.__n_samples))
                
            if self.__dispersion_coefficient_M > 0 or self.__dispersion_coefficient_K > 0:
                self.__Drom_rand = self.compute_Drom(self.__Mrom_rand, self.__Krom_rand)
            else:
                self.__Drom_rand = np.tile(self.__Drom, (1, 1, self.__n_samples))
                
        elif uncertainty_type == "generalized":
            self.__mesh.compute_random_materials()
            self.__Mrom_rand = np.zeros(self.__Mrom.shape + (self.__n_samples,))
            self.__Krom_rand = np.zeros(self.__Krom.shape + (self.__n_samples,))
            self.__Drom_rand = np.zeros(self.__Drom.shape + (self.__n_samples,))
            
            for ii in range(self.__n_samples):
                self.__mesh.set_random_materials(ii)
                self.compute_factorized_M_K()
                self.apply_dirichlet_M()
                self.apply_dirichlet_K()
                self.compute_linear_ROM()
                
                Mrom_rand_ii = self.generate_random_M(1)[:, :, 0]
                Krom_rand_ii = self.generate_random_K(1)[:, :, 0]
                Drom_rand_ii = self.compute_Drom(Mrom_rand_ii, Krom_rand_ii)
                
                self.__Mrom_rand[:, :, ii] = Mrom_rand_ii
                self.__Krom_rand[:, :, ii] = Krom_rand_ii
                self.__Drom_rand[:, :, ii] = Drom_rand_ii
                
            self.__mesh.restore_materials()
            
    def get_Mrom_rand(self):
        return self.__Mrom_rand
    
    def get_Krom_rand(self):
        return self.__Krom_rand
    
    def get_Drom_rand(self):
        return self.__Drom_rand