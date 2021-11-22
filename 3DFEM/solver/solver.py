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
        
        self.__structure.compute_factorized_M_K_vectors()
        self.__structure.compute_factorized_M_K()
        # self.compute_M_K()
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
        
        self.__structure.get_mesh().compute_sub_elements_lists()
        self.__structure.compute_modes(n_modes)
        self.__structure.compute_linear_ROM()
        
        Mrom = self.__structure.get_Mrom()
        Krom = self.__structure.get_Krom()
        Drom = self.__structure.get_Drom()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_timesteps)
        self.__force.apply_dirichlet_F()
        
        From = np.dot(self.__structure.get_modesL().transpose(), self.__force.get_FL())
        
        print("Applying initial conditions...")
        
        qU0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_U0L())
        qV0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_V0L())
        qA0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_A0L())
        
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
                        
            vector1_ii = From[:, ii]
            vector2_ii = np.dot(Krom, self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + 0.5 * (1 - beta2) * dt**2 * self.__qA[:, ii - 1])
            vector3_ii = np.dot(Drom, self.__qV[:, ii - 1] + dt * self.__qA[:, ii - 1] / 2)
                        
            qA_ii = np.linalg.solve(matrix_ii, vector1_ii - vector2_ii - vector3_ii)
            qV_ii = self.__qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * self.__qA[:, ii - 1])
            qU_ii = self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + dt**2 * (beta2 * qA_ii + (1 - beta2) * self.__qA[:, ii - 1]) / 2
        
            self.__qU[:, ii] = qU_ii
            self.__qV[:, ii] = qV_ii
            self.__qA[:, ii] = qA_ii
            
            prev_t = t_ii
            
        print("End of time-domain resolution.")
        
        self.__mat_U_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU)
        self.__mat_V_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qV)
        self.__mat_A_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qA)
    
    def linear_diagonal_newmark_solver(self, beta1, beta2, vec_t, n_modes, verbose=True):
        self.__x_axis = vec_t
        
        t0 = self.__x_axis[0]
        n_timesteps = len(self.__x_axis)
        
        print("Computing reduced-order model...")
        
        self.__structure.get_mesh().compute_sub_elements_lists()
        self.__structure.compute_modes(n_modes)
        self.__structure.compute_linear_diagonal_ROM()
        
        Mrom = self.__structure.get_Mrom()
        Krom = self.__structure.get_Krom()
        Drom = self.__structure.get_Drom()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_timesteps)
        self.__force.apply_dirichlet_F()
        
        From = np.dot(self.__structure.get_modesL().transpose(), self.__force.get_FL())
        
        print("Applying initial conditions...")
        
        qU0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_U0L())
        qV0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_V0L())
        qA0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_A0L())
        
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
                        
            vector1_ii = From[:, ii]
            vector2_ii = np.multiply(Krom, self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + 0.5 * (1 - beta2) * dt**2 * self.__qA[:, ii - 1])
            vector3_ii = np.multiply(Drom, self.__qV[:, ii - 1] + dt * self.__qA[:, ii - 1] / 2)
                        
            qA_ii = np.divide(vector1_ii - vector2_ii - vector3_ii, matrix_ii)
            qV_ii = self.__qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * self.__qA[:, ii - 1])
            qU_ii = self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + dt**2 * (beta2 * qA_ii + (1 - beta2) * self.__qA[:, ii - 1]) / 2
        
            self.__qU[:, ii] = qU_ii
            self.__qV[:, ii] = qV_ii
            self.__qA[:, ii] = qA_ii
            
            prev_t = t_ii
            
        print("End of time-domain resolution.")
        
        self.__mat_U_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU)
        self.__mat_V_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qV)
        self.__mat_A_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qA)
    
    def linear_frequency_solver(self, vec_f, n_modes, verbose=True):
        self.__x_axis = vec_f
        vec_w = 2 * np.pi * vec_f
        n_freqsteps = len(self.__x_axis)
        
        print("Computing reduced-order model...")
                
        self.__structure.get_mesh().compute_sub_elements_lists()
        self.__structure.compute_modes(n_modes)
        self.__structure.compute_linear_ROM()
        
        Mrom = self.__structure.get_Mrom()
        Krom = self.__structure.get_Krom()
        Drom = self.__structure.get_Drom()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_freqsteps)
        self.__force.apply_dirichlet_F()
        
        From = np.dot(self.__structure.get_modesL().transpose(), self.__force.get_FL())
        
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
            
        print("End of frequency-domain resolution.")
        
        self.__mat_U_observed = np.abs(np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU))
    
    def linear_diagonal_frequency_solver(self, vec_f, n_modes, verbose=True):
        self.__x_axis = vec_f
        vec_w = 2 * np.pi * vec_f
        n_freqsteps = len(self.__x_axis)
        
        print("Computing reduced-order model...")
                
        self.__structure.get_mesh().compute_sub_elements_lists()
        self.__structure.compute_modes(n_modes)
        self.__structure.compute_linear_diagonal_ROM()
        
        Mrom = self.__structure.get_Mrom()
        Krom = self.__structure.get_Krom()
        Drom = self.__structure.get_Drom()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_freqsteps)
        self.__force.apply_dirichlet_F()
        
        From = np.dot(self.__structure.get_modesL().transpose(), self.__force.get_FL())
        
        # resolution
        
        print("Starting frequency-domain resolution...")
        
        self.__qU = np.zeros((n_modes, n_freqsteps), dtype=np.csingle)
                
        for ii in range(n_freqsteps):
            
            w_ii = vec_w[ii]
            
            if verbose == True:
                print("Frequency step n° ", ii, " / frequency = ", w_ii / (2 * np.pi), " Hz")
                
            matrix_ii = -(w_ii**2) * Mrom + 1j * w_ii * Drom + Krom
                        
            vector_ii = From[:, ii]
                        
            qU_ii = np.divide(vector_ii, matrix_ii)
            
            self.__qU[:, ii] = qU_ii
            
        print("End of frequency-domain resolution.")
        
        self.__mat_U_observed = np.abs(np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU))
    
    def get_x_axis(self):
        return self.__x_axis
    
    def get_qU(self):
        return self.__qU
    
    def get_qV(self):
        return self.__qV
    
    def get_qA(self):
        return self.__qA
    
    def get_sub_U(self, ls_dofs):
        sub_U = np.dot(self.__structure.get_modes()[ls_dofs, :], self.__qU)
        
        return sub_U
    
    def get_sub_V(self, ls_dofs):
        sub_V = np.dot(self.__structure.get_modes()[ls_dofs, :], self.__qV)
        
        return sub_V
    
    def get_sub_A(self, ls_dofs):
        sub_A = np.dot(self.__structure.get_modes()[ls_dofs, :], self.__qA)
        
        return sub_A
    
    def get_mat_U_observed(self):
        return self.__mat_U_observed
    
    def get_mat_V_observed(self):
        return self.__mat_V_observed
    
    def get_mat_A_observed(self):
        return self.__mat_A_observed
        
    def get_mat_U(self):
        mat_U = np.dot(self.__structure.get_modes(), self.__qU)
        
        return mat_U
    
    def get_mat_V(self):
        mat_V = np.dot(self.__structure.get_modes(), self.__qV)
        
        return mat_V
    
    def get_mat_A(self):
        mat_A = np.dot(self.__structure.get_modes(), self.__qA)
        
        return mat_A
    
    def get_vec_U_step(self, index_step):
        vec_U = np.dot(self.__structure.get_modes(), self.__qU[:, index_step])
        
        return vec_U
    
    def get_vec_V_step(self, index_step):
        vec_V = np.dot(self.__structure.get_modes(), self.__qV[:, index_step])
        
        return vec_V
    
    def get_vec_A_step(self, index_step):
        vec_A = np.dot(self.__structure.get_modes(), self.__qA[:, index_step])
        
        return vec_A
    
    def get_vec_absU_step(self, index_step):
        vec_U = np.abs(np.dot(self.__structure.get_modes(), self.__qU[:, index_step]))
        
        return vec_U
    
    def linear_newmark_solver_UQ(self, beta1, beta2, vec_t, n_modes, uncertainty_type="nonparametric", add_deterministic=False, verbose=True):
        self.__x_axis = vec_t
        
        n_timesteps = len(self.__x_axis)
                
        print("Computing reduced-order model...")
        
        self.__structure.get_mesh().compute_sub_elements_lists()
        self.__structure.compute_modes(n_modes)
        self.__structure.compute_linear_ROM()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_timesteps)
        self.__force.apply_dirichlet_F()
        
        From = np.dot(self.__structure.get_modesL().transpose(), self.__force.get_FL())
        
        print("Generating random matrices...")
        
        self.__structure.generate_random_matrices(uncertainty_type)
        
        Mrom_rand = self.__structure.get_Mrom_rand()
        Krom_rand = self.__structure.get_Krom_rand()
        Drom_rand = self.__structure.get_Drom_rand()
        
        print("Applying initial conditions...")
        
        qU0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_U0L())
        qU0 = np.reshape(qU0, (n_modes, 1))
        qV0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_V0L())
        qV0 = np.reshape(qV0, (n_modes, 1))
        qA0 = np.dot(self.__structure.get_modesL().transpose(), self.__structure.get_A0L())
        qA0 = np.reshape(qA0, (n_modes, 1))
        
        self.__qU_rand = np.zeros((n_modes, n_timesteps, self.__structure.get_n_samples()))
        self.__qV_rand = np.zeros((n_modes, n_timesteps, self.__structure.get_n_samples()))
        self.__qA_rand = np.zeros((n_modes, n_timesteps, self.__structure.get_n_samples()))
        
        self.__qU_rand[:, 0, :] = np.tile(qU0, (1, self.__structure.get_n_samples()))
        self.__qV_rand[:, 0, :] = np.tile(qV0, (1, self.__structure.get_n_samples()))
        self.__qA_rand[:, 0, :] = np.tile(qA0, (1, self.__structure.get_n_samples()))
        
        self.__array_U_rand_observed = np.zeros((self.__structure.get_mesh().get_n_observed_dofs(), n_timesteps, self.__structure.get_n_samples()))
        self.__array_V_rand_observed = np.zeros((self.__structure.get_mesh().get_n_observed_dofs(), n_timesteps, self.__structure.get_n_samples()))
        self.__array_A_rand_observed = np.zeros((self.__structure.get_mesh().get_n_observed_dofs(), n_timesteps, self.__structure.get_n_samples()))
        
        U0L = self.__structure.get_U0L()
        U0 = np.zeros((self.__structure.get_n_total_dofs(),))
        U0[self.__structure.get_mesh().get_free_dofs()] = U0L
        U0_obs = U0[self.__structure.get_mesh().get_observed_dofs()]
        U0_obs = np.reshape(U0_obs, (self.__structure.get_mesh().get_n_observed_dofs(), 1))
        
        V0L = self.__structure.get_V0L()
        V0 = np.zeros((self.__structure.get_n_total_dofs(),))
        V0[self.__structure.get_mesh().get_free_dofs()] = V0L
        V0_obs = V0[self.__structure.get_mesh().get_observed_dofs()]
        V0_obs = np.reshape(V0_obs, (self.__structure.get_mesh().get_n_observed_dofs(), 1))
        
        A0L = self.__structure.get_A0L()
        A0 = np.zeros((self.__structure.get_n_total_dofs(),))
        A0[self.__structure.get_mesh().get_free_dofs()] = A0L
        A0_obs = A0[self.__structure.get_mesh().get_observed_dofs()]
        A0_obs = np.reshape(A0_obs, (self.__structure.get_mesh().get_n_observed_dofs(), 1))
                
        self.__array_U_rand_observed[:, 0, :] = np.tile(U0_obs, (1, self.__structure.get_n_samples()))
        self.__array_V_rand_observed[:, 0, :] = np.tile(V0_obs, (1, self.__structure.get_n_samples()))
        self.__array_A_rand_observed[:, 0, :] = np.tile(A0_obs, (1, self.__structure.get_n_samples()))
        
        # resolution
        
        print("Starting stochastic time-domain resolution...")
            
        for jj in range(self.__structure.get_n_samples()):
            
            if verbose == True:
                print("Sample n° ", jj)
                
            prev_t = self.__x_axis[0]
                
            Mrom_ii = np.squeeze(Mrom_rand[:, :, jj])
            Krom_ii = np.squeeze(Krom_rand[:, :, jj])
            Drom_ii = np.squeeze(Drom_rand[:, :, jj])
        
            for ii in range(1, n_timesteps):
                
                t_ii = self.__x_axis[ii]
            
                dt = t_ii - prev_t
                
                matrix_ii = Mrom_ii + beta2 * dt**2 * Krom_ii / 2 + dt * Drom_ii / 2
                            
                vector1_ii = From[:, ii]
                vector2_ii = np.dot(Krom_ii, self.__qU_rand[:, ii - 1, jj] + dt * self.__qV_rand[:, ii - 1, jj] + 0.5 * (1 - beta2) * dt**2 * self.__qA_rand[:, ii - 1, jj])
                vector3_ii = np.dot(Drom_ii, self.__qV_rand[:, ii - 1, jj] + dt * self.__qA_rand[:, ii - 1, jj] / 2)
                            
                qA_ii = np.linalg.solve(matrix_ii, vector1_ii - vector2_ii - vector3_ii)
                qV_ii = self.__qV_rand[:, ii - 1, jj] + dt * (beta1 * qA_ii + (1 - beta1) * self.__qA_rand[:, ii - 1, jj])
                qU_ii = self.__qU_rand[:, ii - 1, jj] + dt * self.__qV_rand[:, ii - 1, jj] + dt**2 * (beta2 * qA_ii + (1 - beta2) * self.__qA_rand[:, ii - 1, jj]) / 2
            
                self.__qU_rand[:, ii, jj] = qU_ii
                self.__qV_rand[:, ii, jj] = qV_ii
                self.__qA_rand[:, ii, jj] = qA_ii
                
                prev_t = t_ii
                                        
            self.__mat_U_rand_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], np.squeeze(self.__qU_rand[:, :, jj]))
            self.__mat_V_rand_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], np.squeeze(self.__qV_rand[:, :, jj]))
            self.__mat_A_rand_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], np.squeeze(self.__qA_rand[:, :, jj]))
            
            self.__array_U_rand_observed[:, :, jj] = self.__mat_U_rand_observed
            self.__array_V_rand_observed[:, :, jj] = self.__mat_V_rand_observed
            self.__array_A_rand_observed[:, :, jj] = self.__mat_A_rand_observed
            
        if add_deterministic == True:
            
            print("Deterministic case...")
            
            if uncertainty_type == "parametric" or uncertainty_type == "generalized":
                self.__structure.compute_factorized_M_K()
                self.__structure.apply_dirichlet_M()
                self.__structure.apply_dirichlet_K()
                self.__structure.compute_linear_ROM()
            
            Mrom = self.__structure.get_Mrom()
            Krom = self.__structure.get_Krom()
            Drom = self.__structure.get_Drom()
            
            self.__qU = np.zeros((n_modes, n_timesteps))
            self.__qV = np.zeros((n_modes, n_timesteps))
            self.__qA = np.zeros((n_modes, n_timesteps))
                
            for ii in range(1, n_timesteps):
            
                t_ii = self.__x_axis[ii]
                
                dt = t_ii - prev_t
                
                matrix_ii = Mrom + beta2 * dt**2 * Krom / 2 + dt * Drom / 2
                            
                vector1_ii = From[:, ii]
                vector2_ii = np.dot(Krom, self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + 0.5 * (1 - beta2) * dt**2 * self.__qA[:, ii - 1])
                vector3_ii = np.dot(Drom, self.__qV[:, ii - 1] + dt * self.__qA[:, ii - 1] / 2)
                            
                qA_ii = np.linalg.solve(matrix_ii, vector1_ii - vector2_ii - vector3_ii)
                qV_ii = self.__qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * self.__qA[:, ii - 1])
                qU_ii = self.__qU[:, ii - 1] + dt * self.__qV[:, ii - 1] + dt**2 * (beta2 * qA_ii + (1 - beta2) * self.__qA[:, ii - 1]) / 2
            
                self.__qU[:, ii] = qU_ii
                self.__qV[:, ii] = qV_ii
                self.__qA[:, ii] = qA_ii
                
                prev_t = t_ii
                            
            self.__mat_U_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU)
            self.__mat_V_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qV)
            self.__mat_A_observed = np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qA)
            
        print("End of stochastic time-domain resolution.")
    
    def linear_frequency_solver_UQ(self, vec_f, n_modes, uncertainty_type="nonparametric", add_deterministic=False, verbose=True):
        self.__x_axis = vec_f
        vec_w = 2 * np.pi * vec_f
        n_freqsteps = len(self.__x_axis)
                
        print("Computing reduced-order model...")
        
        self.__structure.get_mesh().compute_sub_elements_lists()
        self.__structure.compute_modes(n_modes)
        self.__structure.compute_linear_ROM()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F(n_freqsteps)
        self.__force.apply_dirichlet_F()
        
        From = np.dot(self.__structure.get_modesL().transpose(), self.__force.get_FL())
        
        print("Generating random matrices...")
        
        self.__structure.generate_random_matrices(uncertainty_type)
        
        Mrom_rand = self.__structure.get_Mrom_rand()
        Krom_rand = self.__structure.get_Krom_rand()
        Drom_rand = self.__structure.get_Drom_rand()
        
        # resolution
        
        print("Starting stochastic frequency-domain resolution...")
        
        self.__qU_rand = np.zeros((n_modes, n_freqsteps, self.__structure.get_n_samples()), dtype=np.csingle)
        self.__array_U_rand_observed = np.zeros((self.__structure.get_mesh().get_n_observed_dofs(), n_freqsteps, self.__structure.get_n_samples()))
           
        for jj in range(self.__structure.get_n_samples()):
            
            if verbose == True:
                print("Sample n° ", jj)
                
            Mrom_ii = np.squeeze(Mrom_rand[:, :, jj])
            Krom_ii = np.squeeze(Krom_rand[:, :, jj])
            Drom_ii = np.squeeze(Drom_rand[:, :, jj])
        
            for ii in range(n_freqsteps):
                
                w_ii = vec_w[ii]
                    
                matrix_ii = -(w_ii**2) * Mrom_ii + 1j * w_ii * Drom_ii + Krom_ii
                            
                vector_ii = From[:, ii]
                            
                qU_ii = np.linalg.solve(matrix_ii, vector_ii)
                
                self.__qU_rand[:, ii, jj] = qU_ii
                    
            self.__mat_U_rand_observed = np.abs(np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], np.squeeze(self.__qU_rand[:, :, jj])))
            self.__array_U_rand_observed[:, :, jj] = self.__mat_U_rand_observed
            
        if add_deterministic == True:
            
            print("Deterministic case...")
            
            if uncertainty_type == "parametric" or uncertainty_type == "generalized":
                self.__structure.compute_factorized_M_K()
                self.__structure.apply_dirichlet_M()
                self.__structure.apply_dirichlet_K()
                self.__structure.compute_linear_ROM()
            
            Mrom = self.__structure.get_Mrom()
            Krom = self.__structure.get_Krom()
            Drom = self.__structure.get_Drom()
            
            self.__qU = np.zeros((n_modes, n_freqsteps), dtype=np.csingle)
                
            for ii in range(n_freqsteps):
                
                w_ii = vec_w[ii]
                
                matrix_ii = -(w_ii**2) * Mrom + 1j * w_ii * Drom + Krom
                            
                vector_ii = From[:, ii]
                            
                qU_ii = np.linalg.solve(matrix_ii, vector_ii)
                
                self.__qU[:, ii] = qU_ii
                            
            self.__mat_U_observed = np.abs(np.dot(self.__structure.get_modes()[self.__structure.get_mesh().get_observed_dofs(), :], self.__qU))
            
        print("End of stochastic frequency-domain resolution.")
        
    def get_qU_rand(self):
        return self.__qU_rand
    
    def get_array_U_rand_observed(self):
        return self.__array_U_rand_observed