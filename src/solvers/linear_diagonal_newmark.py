import numpy as np
from solvers.solver import Solver


class LinearDiagonalNewmarkSolver(Solver):
    def __init__(self, structure, neumann_BC, initial_conditions):
        super(LinearDiagonalNewmarkSolver, self).__init__(structure, neumann_BC=neumann_BC, initial_conditions=initial_conditions)

        self.x_axis = None

        self.mat_qU = None
        self.mat_qV = None
        self.mat_qA = None
        self.mat_U_observed = None
        self.mat_V_observed = None
        self.mat_A_observed = None

    def run(self, beta1=1.0 / 2, beta2=1.0 / 2, t0=0, dt=1e-3, n_timesteps=1000, n_modes=10, verbose=True):
        self.x_axis = np.linspace(t0, t0 + dt * n_timesteps, n_timesteps + 1)

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

        if verbose:
            print("Applying initial conditions...")

        self.initial_conditions.apply_dirichlet(self.free_dofs)

        qU0 = np.dot(self.structure.mat_modes_f.transpose(), self.initial_conditions.U0_f)
        qV0 = np.dot(self.structure.mat_modes_f.transpose(), self.initial_conditions.V0_f)
        qA0 = np.dot(self.structure.mat_modes_f.transpose(), self.initial_conditions.A0_f)

        self.mat_qU = np.zeros((n_modes, n_timesteps + 1))
        self.mat_qV = np.zeros((n_modes, n_timesteps + 1))
        self.mat_qA = np.zeros((n_modes, n_timesteps + 1))

        self.mat_qU[:, 0] = qU0
        self.mat_qV[:, 0] = qV0
        self.mat_qA[:, 0] = qA0

        # resolution

        if verbose:
            print("Starting time-domain resolution...")

        prev_t = self.x_axis[0]

        for ii in range(1, n_timesteps + 1):

            t_ii = self.x_axis[ii]

            dt = t_ii - prev_t

            if verbose:
                print("Timestep n° ", ii, " , time = ", t0 + ii * dt)

            matrix_ii = Mrom + beta2 * dt ** 2 * Krom / 2 + dt * Drom / 2

            vector1_ii = From[:, ii]
            vector2_ii = np.multiply(Krom, self.mat_qU[:, ii - 1] + dt * self.mat_qV[:, ii - 1]
                                     + 0.5 * (1 - beta2) * dt ** 2 * self.mat_qA[:, ii - 1])
            vector3_ii = np.multiply(Drom, self.mat_qV[:, ii - 1] + dt * self.mat_qA[:, ii - 1] / 2)

            qA_ii = np.divide(vector1_ii - vector2_ii - vector3_ii, matrix_ii)
            qV_ii = self.mat_qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * self.mat_qA[:, ii - 1])
            qU_ii = self.mat_qU[:, ii - 1] + dt * self.mat_qV[:, ii - 1] + dt ** 2 * (
                        beta2 * qA_ii + (1 - beta2) * self.mat_qA[:, ii - 1]) / 2

            self.mat_qU[:, ii] = qU_ii
            self.mat_qV[:, ii] = qV_ii
            self.mat_qA[:, ii] = qA_ii

            prev_t = t_ii

        if verbose:
            print("End of time-domain resolution.")

        self.mat_U_observed = np.dot(self.structure.mat_modes[self.structure.mesh.observed_dofs, :],
                                     self.mat_qU)
        self.mat_V_observed = np.dot(self.structure.mat_modes[self.structure.mesh.observed_dofs, :],
                                     self.mat_qV)
        self.mat_A_observed = np.dot(self.structure.mat_modes[self.structure.mesh.observed_dofs, :],
                                     self.mat_qA)
