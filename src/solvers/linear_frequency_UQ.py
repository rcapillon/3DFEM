import numpy as np
from solvers.solver import Solver


class LinearFrequencyUQSolver(Solver):
    def __init__(self, structure, neumann_BC):
        super(LinearFrequencyUQSolver, self).__init__(structure, neumann_BC=neumann_BC)

        self.x_axis = None

        self.array_qU_rand = None
        self.array_U_rand_observed = None

        self.mat_qU = None
        self.mat_U_observed = None

    def run(self, vec_f, n_modes=10,
            n_samples=100, uncertainty_type="nonparametric", add_deterministic=False,
            verbose=True):
        self.x_axis = vec_f
        vec_w = 2 * np.pi * vec_f
        n_freqsteps = len(vec_w)

        if verbose:
            print("Computing reduced-order model...")

        self.structure.compute_modes(n_modes)
        self.structure.compute_linear_ROM()

        if add_deterministic:
            Mrom_deter = self.structure.Mrom
            Krom_deter = self.structure.Krom
            Drom_deter = self.structure.Drom

        self.neumann_BC.compute_F0(self.n_total_dofs)
        self.neumann_BC.compute_varying_F()
        mat_F_f = self.neumann_BC.mat_F[self.free_dofs, :]

        From = np.dot(self.structure.mat_modes_f.transpose(), mat_F_f)

        # resolution

        if verbose:
            print("Starting stochastic frequency-domain resolution...")

        self.array_qU_rand = np.zeros((n_modes, n_freqsteps, n_samples), dtype=np.csingle)
        self.array_U_rand_observed = np.zeros((self.n_observed_dofs, n_freqsteps, n_samples))

        for jj in range(n_samples):

            if verbose:
                print("Sample n° ", jj)

            self.structure.generate_random_matrices(uncertainty_type)

            Mrom = self.structure.Mrom_rand
            Krom = self.structure.Krom_rand
            Drom = self.structure.Drom_rand

            for ii in range(n_freqsteps):
                w_ii = vec_w[ii]

                matrix_ii = -(w_ii ** 2) * Mrom + 1j * w_ii * Drom + Krom

                vector_ii = From[:, ii]

                qU_ii = np.linalg.solve(matrix_ii, vector_ii)

                self.array_qU_rand[:, ii, jj] = qU_ii

            mat_U_rand_observed_jj = np.abs(np.dot(self.structure.mat_modes[self.observed_dofs, :],
                                                   np.squeeze(self.array_qU_rand[:, :, jj])))
            self.array_U_rand_observed[:, :, jj] = mat_U_rand_observed_jj

        if uncertainty_type == "parametric" or uncertainty_type == "generalized":
            for material in self.structure.mesh.materials:
                material.restore_mean_material_parameters()

        if add_deterministic:

            if verbose:
                print("Deterministic case...")

            self.mat_qU = np.zeros((n_modes, n_freqsteps), dtype=np.csingle)

            for ii in range(n_freqsteps):
                w_ii = vec_w[ii]

                matrix_ii = -(w_ii ** 2) * Mrom_deter + 1j * w_ii * Drom_deter + Krom_deter

                vector_ii = From[:, ii]

                qU_ii = np.linalg.solve(matrix_ii, vector_ii)

                self.mat_qU[:, ii] = qU_ii

            self.mat_U_observed = np.abs(
                np.dot(self.structure.mat_modes[self.observed_dofs, :], self.mat_qU))

        if verbose:
            print("End of stochastic frequency-domain resolution.")
