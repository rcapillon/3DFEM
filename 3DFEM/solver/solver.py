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
spec1 = importlib.util.spec_from_file_location("force", "./force/force.py")
force = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(force)

import importlib.util
spec2 = importlib.util.spec_from_file_location("structure", "./structure/structure.py")
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
    
    def linear_newmark_solver(self, beta1, beta2, t0, tmax, n_timesteps, n_modes, verbose=True):
        dt = (tmax - t0) / (n_timesteps - 1)
        
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
        
        From = np.dot(self.__modesL.transpose(), self.__force.get_FL())
        
        print("Applying initial conditions...")
        
        qU0 = np.dot(self.__modesL.transpose(), self.__structure.get_U0L())
        qV0 = np.dot(self.__modesL.transpose(), self.__structure.get_V0L())
        qA0 = np.dot(self.__modesL.transpose(), self.__structure.get_A0L())
        
        qU = np.zeros((n_modes, n_timesteps + 1))
        qV = np.zeros((n_modes, n_timesteps + 1))
        qA = np.zeros((n_modes, n_timesteps + 1))
        
        qU[:, 0] = qU0
        qV[:, 0] = qV0
        qA[:, 0] = qA0
        
        # resolution
        
        print("Starting time-domain resolution...")
        
        for ii in range(1, n_timesteps + 1):
            
            if verbose == True:
                print("Timestep n° ", ii)
            
            spmatrix_ii = Mrom + beta2 * dt**2 * Krom / 2 + dt * Drom / 2
                        
            vector1_ii = From[:, ii - 1]
            vector2_ii = np.dot(Krom, qU[:, ii - 1] + dt * qV[:, ii - 1] + 0.5 * (1 - beta2) * dt**2 * qA[:, ii - 1])
            vector3_ii = np.dot(Drom, qV[:, ii - 1] + dt * qA[:, ii - 1] / 2)
                        
            qA_ii = np.linalg.solve(spmatrix_ii, vector1_ii - vector2_ii - vector3_ii)
            qV_ii = qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * qA[:, ii - 1])
            qU_ii = qU[:, ii - 1] + dt * qV[:, ii - 1] + dt**2 * (beta2 * qA_ii + (1 - beta2) * qA[:, ii - 1]) / 2
        
            qU[:, ii] = qU_ii
            qV[:, ii] = qV_ii
            qA[:, ii] = qA_ii
            
        print("End of time-domain resolution.")
        
        self.__mat_U = np.zeros((self.__n_total_dofs, n_timesteps + 1))
        self.__mat_V = np.zeros((self.__n_total_dofs, n_timesteps + 1))
        self.__mat_A = np.zeros((self.__n_total_dofs, n_timesteps + 1))
        
        mat_UL = np.dot(self.__modesL, qU)
        mat_VL = np.dot(self.__modesL, qV)
        mat_AL = np.dot(self.__modesL, qA)
        
        self.__mat_U[self.__structure.get_mesh().get_free_dofs(), :] = mat_UL
        self.__mat_V[self.__structure.get_mesh().get_free_dofs(), :] = mat_VL
        self.__mat_A[self.__structure.get_mesh().get_free_dofs(), :] = mat_AL
        
    def get_mat_U(self):
        return self.__mat_U
    
    def get_mat_V(self):
        return self.__mat_V
    
    def get_mat_A(self):
        return self.__mat_A