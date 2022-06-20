import numpy as np
from scipy.sparse.linalg import spsolve
from solvers.solver import Solver


class NonlinearStaticsNewtonRaphsonSolver(Solver):
    def __init__(self, structure, neumann_BC):
        super(NonlinearStaticsNewtonRaphsonSolver, self).__init__(structure, neumann_BC=neumann_BC)

        self.y_axis = None

        self.mat_U = None

        self.mat_U_observed = None

    def run(self, lambda_max=1.0, n_load_increments=10, n_iter_max=10, tol=1e-3, verbose=True):
        self.y_axis = np.linspace(0, lambda_max, n_load_increments + 1)

        self.mat_U = np.zeros((self.n_total_dofs, n_load_increments + 1))

        if verbose:
            print("Computing forces...")

        self.neumann_BC.compute_F0(self.n_total_dofs)
        vec_F_f = self.neumann_BC.vec_F0[self.free_dofs]

        if verbose:
            print("Starting Newton-Raphson resolution...")

        vec_U = np.zeros((self.n_total_dofs,))

        self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
        self.structure.compute_factorized_KT_Fint(vec_U)
        self.structure.apply_dirichlet_KT_Fint()

        for ll, load_factor in enumerate(self.y_axis[1:]):
            if verbose:
                print("\nLoad increment n°", ll + 1, " , load factor =", load_factor)

            vec_Delta_U = spsolve(self.structure.mat_KTff, load_factor * vec_F_f - self.structure.vec_Fint_f)

            vec_U[self.free_dofs] += vec_Delta_U

            self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
            self.structure.compute_factorized_KT_Fint(vec_U)
            self.structure.apply_dirichlet_KT_Fint()

            error = np.linalg.norm(load_factor * vec_F_f - self.structure.vec_Fint_f) \
                    / np.linalg.norm(load_factor * vec_F_f)

            if verbose:
                print("Predictor error:", error)

            counter_iter = 0

            if np.isnan(error):
                error = tol + 1
                counter_iter = n_iter_max

            while counter_iter < n_iter_max and error > tol:
                vec_delta_U = spsolve(self.structure.mat_KTff, load_factor * vec_F_f - self.structure.vec_Fint_f)

                vec_U[self.free_dofs] += vec_delta_U

                self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
                self.structure.compute_factorized_KT_Fint(vec_U)
                self.structure.apply_dirichlet_KT_Fint()

                error = np.linalg.norm(load_factor * vec_F_f - self.structure.vec_Fint_f) \
                        / np.linalg.norm(load_factor * vec_F_f)

                if verbose:
                    print("Correction", counter_iter, ", error:", error)

                if np.isnan(error):
                    error = tol + 1
                    counter_iter = n_iter_max

                counter_iter += 1

            if counter_iter >= n_iter_max and error > tol:
                print("Newton-Raphson algorithm failed to converge.")
                self.mat_U[:, (ll + 1):] = float('NaN')
            else:
                self.mat_U[self.free_dofs, ll + 1] = vec_U[self.free_dofs]

        self.mat_U_observed = self.mat_U[self.observed_dofs, :]
