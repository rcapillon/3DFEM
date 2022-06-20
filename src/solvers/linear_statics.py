import numpy as np
from scipy.sparse.linalg import spsolve
from solvers.solver import Solver


class LinearStaticsSolver(Solver):
    def __init__(self, structure, neumann_BC):
        super(LinearStaticsSolver, self).__init__(structure, neumann_BC=neumann_BC)

        self.vec_U = None
        self.vec_U_observed = None

    def run(self):
        self.structure.compute_factorized_K_vectors()
        self.structure.compute_factorized_K()
        self.structure.apply_dirichlet_K()

        self.neumann_BC.compute_F0(self.n_total_dofs)
        vec_F0_f = self.neumann_BC.vec_F0[self.free_dofs]

        vec_UL = spsolve(self.structure.mat_Kff, vec_F0_f)

        self.vec_U = np.zeros((self.n_total_dofs,))
        self.vec_U[self.free_dofs] = vec_UL

        self.vec_U_observed = self.vec_U[self.observed_dofs]
