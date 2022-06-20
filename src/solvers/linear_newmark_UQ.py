import numpy as np
from solvers.solver import Solver


class LinearNewmarkUQSolver(Solver):
    def __init__(self, structure, neumann_BC, initial_conditions):
        super(LinearNewmarkUQSolver, self).__init__(structure, neumann_BC=neumann_BC, initial_conditions=initial_conditions)

        self.x_axis = None

        self.array_qU_rand = None
        self.array_qV_rand = None
        self.array_qA_rand = None
        self.array_U_rand_observed = None
        self.array_V_rand_observed = None
        self.array_A_rand_observed = None

        self.mat_qU = None
        self.mat_qV = None
        self.mat_qA = None
        self.mat_U_observed = None
        self.mat_V_observed = None
        self.mat_A_observed = None

    def run(self, beta1=1.0/2, beta2=1.0/2, t0=0, dt=1e-3, n_timesteps=1000, n_modes=10,
            n_samples=100, uncertainty_type="nonparametric", add_deterministic=False,
            verbose=True):
        self.x_axis = np.linspace(t0, t0 + dt * n_timesteps, n_timesteps + 1)

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

        if verbose:
            print("Applying initial conditions...")

        self.initial_conditions.apply_dirichlet(self.free_dofs)

        qU0 = np.dot(self.structure.mat_modes_f.transpose(), self.initial_conditions.U0_f)
        qU0 = np.reshape(qU0, (n_modes, 1))
        qV0 = np.dot(self.structure.mat_modes_f.transpose(), self.initial_conditions.V0_f)
        qV0 = np.reshape(qV0, (n_modes, 1))
        qA0 = np.dot(self.structure.mat_modes_f.transpose(), self.initial_conditions.A0_f)
        qA0 = np.reshape(qA0, (n_modes, 1))

        self.array_qU_rand = np.zeros((n_modes, n_timesteps + 1, n_samples))
        self.array_qV_rand = np.zeros((n_modes, n_timesteps + 1, n_samples))
        self.array_qA_rand = np.zeros((n_modes, n_timesteps + 1, n_samples))

        self.array_qU_rand[:, 0, :] = np.tile(qU0, (1, n_samples))
        self.array_qV_rand[:, 0, :] = np.tile(qV0, (1, n_samples))
        self.array_qA_rand[:, 0, :] = np.tile(qA0, (1, n_samples))

        self.array_U_rand_observed = np.zeros((self.n_observed_dofs, n_timesteps + 1, n_samples))
        self.array_V_rand_observed = np.zeros((self.n_observed_dofs, n_timesteps + 1, n_samples))
        self.array_A_rand_observed = np.zeros((self.n_observed_dofs, n_timesteps + 1, n_samples))

        U0_f = self.initial_conditions.U0_f
        U0 = np.zeros((self.n_total_dofs,))
        U0[self.free_dofs] = U0_f
        U0_obs = U0[self.observed_dofs]
        U0_obs = np.reshape(U0_obs, (self.n_observed_dofs, 1))

        V0_f = self.initial_conditions.V0_f
        V0 = np.zeros((self.n_total_dofs,))
        V0[self.free_dofs] = V0_f
        V0_obs = V0[self.observed_dofs]
        V0_obs = np.reshape(V0_obs, (self.n_observed_dofs, 1))

        A0_f = self.initial_conditions.A0_f
        A0 = np.zeros((self.n_total_dofs,))
        A0[self.free_dofs] = A0_f
        A0_obs = A0[self.observed_dofs]
        A0_obs = np.reshape(A0_obs, (self.n_observed_dofs, 1))

        self.array_U_rand_observed[:, 0, :] = np.tile(U0_obs, (1, n_samples))
        self.array_V_rand_observed[:, 0, :] = np.tile(V0_obs, (1, n_samples))
        self.array_A_rand_observed[:, 0, :] = np.tile(A0_obs, (1, n_samples))

        # resolution

        if verbose:
            print("Starting stochastic time-domain resolution...")

        for jj in range(n_samples):

            if verbose:
                print("Sample n° ", jj)

            self.structure.generate_random_matrices(uncertainty_type)

            Mrom = self.structure.Mrom_rand
            Krom = self.structure.Krom_rand
            Drom = self.structure.Drom_rand

            prev_t = self.x_axis[0]

            for ii in range(1, n_timesteps + 1):
                t_ii = self.x_axis[ii]

                dt = t_ii - prev_t

                matrix_ii = Mrom + beta2 * dt ** 2 * Krom / 2 + dt * Drom / 2

                vector1_ii = From[:, ii]
                vector2_ii = np.dot(Krom,
                                    self.array_qU_rand[:, ii - 1, jj] + dt * self.array_qV_rand[:, ii - 1, jj]
                                    + 0.5 * (1 - beta2) * dt ** 2 * self.array_qA_rand[:, ii - 1, jj])
                vector3_ii = np.dot(Drom, self.array_qV_rand[:, ii - 1, jj]
                                          + dt * self.array_qA_rand[:, ii - 1, jj] / 2)

                qA_ii = np.linalg.solve(matrix_ii, vector1_ii - vector2_ii - vector3_ii)
                qV_ii = self.array_qV_rand[:, ii - 1, jj] \
                        + dt * (beta1 * qA_ii + (1 - beta1) * self.array_qA_rand[:, ii - 1, jj])
                qU_ii = self.array_qU_rand[:, ii - 1, jj] + dt * self.array_qV_rand[:, ii - 1, jj] \
                        + dt ** 2 * (beta2 * qA_ii + (1 - beta2) * self.array_qA_rand[:, ii - 1, jj]) / 2

                self.array_qU_rand[:, ii, jj] = qU_ii
                self.array_qV_rand[:, ii, jj] = qV_ii
                self.array_qA_rand[:, ii, jj] = qA_ii

                prev_t = t_ii

            mat_U_rand_observed_jj = np.dot(self.structure.mat_modes[self.observed_dofs, :],
                                            np.squeeze(self.array_qU_rand[:, :, jj]))
            mat_V_rand_observed_jj = np.dot(self.structure.mat_modes[self.observed_dofs, :],
                                            np.squeeze(self.array_qV_rand[:, :, jj]))
            mat_A_rand_observed_jj = np.dot(self.structure.mat_modes[self.observed_dofs, :],
                                            np.squeeze(self.array_qA_rand[:, :, jj]))

            self.array_U_rand_observed[:, :, jj] = mat_U_rand_observed_jj
            self.array_V_rand_observed[:, :, jj] = mat_V_rand_observed_jj
            self.array_A_rand_observed[:, :, jj] = mat_A_rand_observed_jj

        if uncertainty_type == "parametric" or uncertainty_type == "generalized":
            for material in self.structure.mesh.materials:
                material.restore_mean_material_parameters()

        if add_deterministic:
            if verbose:
                print("Deterministic case...")

            self.mat_qU = np.zeros((n_modes, n_timesteps + 1))
            self.mat_qV = np.zeros((n_modes, n_timesteps + 1))
            self.mat_qA = np.zeros((n_modes, n_timesteps + 1))

            prev_t = self.x_axis[0]

            for ii in range(1, n_timesteps + 1):

                t_ii = self.x_axis[ii]

                dt = t_ii - prev_t

                matrix_ii = Mrom_deter + beta2 * dt ** 2 * Krom_deter / 2 + dt * Drom_deter / 2

                vector1_ii = From[:, ii]
                vector2_ii = np.dot(Krom_deter,
                                    self.mat_qU[:, ii - 1] + dt * self.mat_qV[:, ii - 1]
                                    + 0.5 * (1 - beta2) * dt ** 2 * self.mat_qA[:, ii - 1])
                vector3_ii = np.dot(Drom_deter, self.mat_qV[:, ii - 1] + dt * self.mat_qA[:, ii - 1] / 2)

                qA_ii = np.linalg.solve(matrix_ii, vector1_ii - vector2_ii - vector3_ii)
                qV_ii = self.mat_qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * self.mat_qA[:, ii - 1])
                qU_ii = self.mat_qU[:, ii - 1] + dt * self.mat_qV[:, ii - 1] \
                        + dt ** 2 * (beta2 * qA_ii + (1 - beta2) * self.mat_qA[:, ii - 1]) / 2

                self.mat_qU[:, ii] = qU_ii
                self.mat_qV[:, ii] = qV_ii
                self.mat_qA[:, ii] = qA_ii

                prev_t = t_ii

            self.mat_U_observed = np.dot(self.structure.mat_modes[self.observed_dofs, :], self.mat_qU)
            self.mat_V_observed = np.dot(self.structure.mat_modes[self.observed_dofs, :], self.mat_qV)
            self.mat_A_observed = np.dot(self.structure.mat_modes[self.observed_dofs, :], self.mat_qA)

        if verbose:
            print("End of stochastic time-domain resolution.")
