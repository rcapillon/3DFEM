import numpy as np
from solvers.solver import Solver


class LinearDiagonalFrequencySolver(Solver):
    def __init__(self, structure, neumann_BC):
        super(LinearDiagonalFrequencySolver, self).__init__(structure, neumann_BC=neumann_BC)

        self.x_axis = None

        self.mat_qU = None
        self.mat_U_observed = None

    def run(self, vec_f, n_modes=10, verbose=True):
        self.x_axis = vec_f
        vec_w = 2 * np.pi * vec_f
        n_freqsteps = len(vec_w)

        if verbose:
            print("Computing reduced-order model...")

        self.structure.compute_modes(n_modes)
        self.structure.compute_linear_diagonal_ROM()

        Mrom = self.structure.Mrom
        Krom = self.structure.Krom
        Drom = self.structure.Drom

        self.neumann_BC.compute_F0(self.n_total_dofs)
        self.neumann_BC.compute_varying_F()
        mat_F_f = self.neumann_BC.mat_F[self.free_dofs, :]

        From = np.dot(self.structure.mat_modes_f.transpose(), mat_F_f)

        # resolution

        if verbose:
            print("Starting frequency-domain resolution...")

        self.mat_qU = np.zeros((n_modes, n_freqsteps), dtype=np.csingle)

        for ii in range(n_freqsteps):

            w_ii = vec_w[ii]

            if verbose:
                print("Frequency step n° ", ii, " / frequency = ", w_ii / (2 * np.pi), " Hz")

            matrix_ii = -(w_ii ** 2) * Mrom + 1j * w_ii * Drom + Krom

            vector_ii = From[:, ii]

            qU_ii = np.divide(vector_ii, matrix_ii)

            self.mat_qU[:, ii] = qU_ii

        print("End of frequency-domain resolution.")

        self.mat_U_observed = np.abs(np.dot(self.structure.mat_modes[self.observed_dofs, :], self.mat_qU))
