import numpy as np
from scipy.sparse.linalg import eigsh
from solvers.solver import Solver


class ModalSolver(Solver):
    def __init__(self, structure):
        super(ModalSolver, self).__init__(structure)

        self.vec_eigenfreqs = None
        self.mat_modes = None

    def run(self, n_modes=10):
        self.vec_eigenfreqs = np.zeros((n_modes,))
        self.mat_modes = np.zeros((self.n_total_dofs, n_modes))

        self.structure.compute_factorized_M_K_vectors()
        self.structure.compute_factorized_M_K()
        self.structure.apply_dirichlet_M()
        self.structure.apply_dirichlet_K()

        (eigvals, eigvects) = eigsh(self.structure.mat_Kff, n_modes, self.structure.mat_Mff, which='SM')
        sort_indices = np.argsort(eigvals)
        eigvals = eigvals[sort_indices]
        eigvects = eigvects[:, sort_indices]

        self.vec_eigenfreqs = np.sqrt(eigvals) / (2 * np.pi)
        self.mat_modes = np.zeros((self.n_total_dofs, n_modes))
        self.mat_modes[self.free_dofs, :] = eigvects
