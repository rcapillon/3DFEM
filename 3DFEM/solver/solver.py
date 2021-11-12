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
from scipy.sparse.linalg import eigsh, spsolve

import importlib.util
spec1 = importlib.util.spec_from_file_location("force", "../force/force.py")
force = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(force)

import importlib.util
spec2 = importlib.util.spec_from_file_location("structure", "../structure/structure.py")
structure = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(structure)

class Solver:
    def __init__(self, structure, force=None):
        self.__structure = structure
        self.__force = force
        self.__n_total_dofs = self.__structure.get_mesh().get_n_total_dofs()
        
    def get_structure(self):
        return self.__structure
    
    def get_force(self):
        return self.__force

    def modal_solver(self, n_modes):
        # Eigenvectors are mass-normalized
        
        self.__structure.compute_M_K()
        self.__structure.apply_dirichlet_M()
        self.__structure.apply_dirichlet_K()
        
        (eigvals, eigvects) = eigsh(self.__structure.get_KLL(), n_modes, self.__structure.get_MLL(), which='SM')
        sort_indices = np.argsort(eigvals)
        eigvals = eigvals[sort_indices]
        eigvects = eigvects[:, sort_indices]
        
        self.__eigenfreqs = np.sqrt(eigvals) / (2 * np.pi)
        self.__modesL = eigvects
        self.__modes = np.zeros((self.__n_total_dofs, n_modes))
        self.__modes[self.__structure.get_mesh().get_free_dofs(), :] = eigvects
        
    def get_eigenfreqs(self):
        return self.__eigenfreqs
    
    def get_modesL(self):
        return self.__modesL
    
    def get_modes(self):
        return self.__modes
            
    def linear_static_solver(self):
        self.__structure.compute_K()
        self.__structure.apply_dirichlet_K()
        
        self.__force.compute_F0()
        self.__force.apply_dirichlet_F0()
        
        vec_UL = spsolve(self.__structure.get_KLL(), self.__force.get_F0L())
                
        self.__vec_U = np.zeros((self.__n_total_dofs,))
        self.__vec_U[self.__structure.get_mesh().get_free_dofs()] = vec_UL
        
    def get_vec_U(self):
        return self.__vec_U
    
    def linear_newmark_solver(self, beta1, beta2, vec_t, n_modes, verbose=True):
        self.__x_axis = vec_t
        
        t0 = self.__x_axis[0]
        n_timesteps = len(self.__x_axis)
        
        print("Computing reduced-order model...")
        
        self.modal_solver(n_modes)
        
        self.__structure.compute_D()
        self.__structure.apply_dirichlet_D()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F()
        self.__force.apply_dirichlet_F()
        
        Mrom = np.dot(self.__modesL.transpose(), self.__structure.get_MLL().dot(self.__modesL))
        Krom = np.dot(self.__modesL.transpose(), self.__structure.get_KLL().dot(self.__modesL))
        Drom = np.dot(self.__modesL.transpose(), self.__structure.get_DLL().dot(self.__modesL))
        
        print(np.linalg.norm(Mrom))
        print(np.linalg.norm(Krom))
        print(np.linalg.norm(Drom))
        
        From = np.dot(self.__modesL.transpose(), self.__force.get_FL())
        
        print("Applying initial conditions...")
        
        qU0 = np.dot(self.__modesL.transpose(), self.__structure.get_U0L())
        qV0 = np.dot(self.__modesL.transpose(), self.__structure.get_V0L())
        qA0 = np.dot(self.__modesL.transpose(), self.__structure.get_A0L())
        
        self.__qU = np.zeros((n_modes, n_timesteps))
        self.__qV = np.zeros((n_modes, n_timesteps))
        self.__qA = np.zeros((n_modes, n_timesteps))
        
        self.__qU[:, 0] = qU0
        self.__qV[:, 0] = qV0
        self.__qA[:, 0] = qA0
        
        # resolution
        
        print("Starting time-domain resolution...")
        
        prev_t = self.__x_axis[0]
        
        for ii in range(1, n_timesteps):
            
            t_ii = self.__x_axis[ii]
            
            dt = t_ii - prev_t
            
            if verbose == True:
                print("Timestep n° ", ii, " , time = ", t0 + ii * dt)
            
            matrix_ii = Mrom + beta2 * dt**2 * Krom / 2 + dt * Drom / 2
                        
            vector1_ii = From[:, ii - 1]
            vector2_ii = np.dot(Krom, self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + 0.5 * (1 - beta2) * dt**2 * self.__qA[:, ii - 1])
            vector3_ii = np.dot(Drom, self.__qV[:, ii - 1] + dt * self.__qA[:, ii - 1] / 2)
                        
            qA_ii = np.linalg.solve(matrix_ii, vector1_ii - vector2_ii - vector3_ii)
            qV_ii = self.__qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * self.__qA[:, ii - 1])
            qU_ii = self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + dt**2 * (beta2 * qA_ii + (1 - beta2) * self.__qA[:, ii - 1]) / 2
        
            self.__qU[:, ii] = qU_ii
            self.__qV[:, ii] = qV_ii
            self.__qA[:, ii] = qA_ii
            
        print("End of time-domain resolution.")
        
        self.__mat_U_observed = np.dot(self.__modes[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU)
        self.__mat_V_observed = np.dot(self.__modes[self.__structure.get_mesh().get_observed_dofs(), :], self.__qV)
        self.__mat_A_observed = np.dot(self.__modes[self.__structure.get_mesh().get_observed_dofs(), :], self.__qA)
    
    def linear_frequency_solver(self, vec_f, n_modes, verbose=True):
        self.__x_axis = vec_f
        vec_w = 2 * np.pi * vec_f
        n_freqsteps = len(self.__x_axis)
        
        print("Computing reduced-order model...")
        
        self.modal_solver(n_modes)
                
        self.__structure.compute_D()
        self.__structure.apply_dirichlet_D()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_freqsteps)
        
        self.__force.apply_dirichlet_F()
                
        Mrom = np.dot(self.__modesL.transpose(), self.__structure.get_MLL().dot(self.__modesL))
        Krom = np.dot(self.__modesL.transpose(), self.__structure.get_KLL().dot(self.__modesL))
        Drom = np.dot(self.__modesL.transpose(), self.__structure.get_DLL().dot(self.__modesL))
        
        From = np.dot(self.__modesL.transpose(), self.__force.get_FL())
        
        # resolution
        
        print("Starting frequency-domain resolution...")
        
        self.__qU = np.zeros((n_modes, n_freqsteps), dtype=np.csingle)
                
        for ii in range(n_freqsteps):
            
            w_ii = vec_w[ii]
            
            if verbose == True:
                print("Frequency step n° ", ii, " / frequency = ", w_ii / (2 * np.pi), " Hz")
                
            matrix_ii = -(w_ii**2) * Mrom + 1j * w_ii * Drom + Krom
                        
            vector_ii = From[:, ii]
                        
            qU_ii = np.linalg.solve(matrix_ii, vector_ii)
            
            self.__qU[:, ii] = qU_ii
            
        print("End of time-domain resolution.")
        
        self.__mat_U_observed = np.abs(np.dot(self.__modes[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU))
    
    def get_x_axis(self):
        return self.__x_axis
    
    def get_qU(self):
        return self.__qU
    
    def get_qV(self):
        return self.__qV
    
    def get_qA(self):
        return self.__qA
    
    def get_sub_U(self, ls_dofs):
        sub_U = np.dot(self.__modes[ls_dofs, :], self.__qU)
        
        return sub_U
    
    def get_sub_V(self, ls_dofs):
        sub_V = np.dot(self.__modes[ls_dofs, :], self.__qV)
        
        return sub_V
    
    def get_sub_A(self, ls_dofs):
        sub_A = np.dot(self.__modes[ls_dofs, :], self.__qA)
        
        return sub_A
    
    def get_mat_U_observed(self):
        return self.__mat_U_observed
    
    def get_mat_V_observed(self):
        return self.__mat_V_observed
    
    def get_mat_A_observed(self):
        return self.__mat_A_observed
        
    def get_mat_U(self):
        mat_U = np.dot(self.__modes, self.__qU)
        
        return mat_U
    
    def get_mat_V(self):
        mat_V = np.dot(self.__modes, self.__qV)
        
        return mat_V
    
    def get_mat_A(self):
        mat_A = np.dot(self.__modes, self.__qA)
        
        return mat_A
    
    def get_vec_U_step(self, index_step):
        vec_U = np.dot(self.__modes, self.__qU[:, index_step])
        
        return vec_U
    
    def get_vec_V_step(self, index_step):
        vec_V = np.dot(self.__modes, self.__qV[:, index_step])
        
        return vec_V
    
    def get_vec_A_step(self, index_step):
        vec_A = np.dot(self.__modes, self.__qA[:, index_step])
        
        return vec_A
    
    def get_vec_absU_step(self, index_step):
        vec_U = np.abs(np.dot(self.__modes, self.__qU[:, index_step]))
        
        return vec_U