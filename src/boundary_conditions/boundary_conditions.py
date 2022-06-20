import numpy as np


class DirichletBC:
    def __init__(self):
        self.dirichlet_dofs = []
        self.n_dirichlet_dofs = 0

    def add_list_of_dirichlet_dofs(self, list_dofs):
        self.dirichlet_dofs.extend(list_dofs)

    def compute_list_of_free_dofs(self, n_total_dofs):
        free_dofs = list(set(range(n_total_dofs)) - set(self.dirichlet_dofs))
        return free_dofs


class NeumannBC:
    def __init__(self):
        self.list_nodal_forces_t0 = []
        self.vec_F0 = None
        self.vec_variation = None
        self.mat_F = None

    def add_nodal_forces_t0(self, list_nodes, nodal_force_vector):
        self.list_nodal_forces_t0.append((list_nodes, nodal_force_vector))

    def compute_F0(self, n_total_dofs):
        self.vec_F0 = np.zeros((n_total_dofs,))

        for group in self.list_nodal_forces_t0:
            nodes = group[0]
            force_vector = group[1]

            ls_dofs_x = [node * 3 for node in nodes]
            ls_dofs_y = [node * 3 + 1 for node in nodes]
            ls_dofs_z = [node * 3 + 2 for node in nodes]

            self.vec_F0[ls_dofs_x] += np.repeat(force_vector[0], len(ls_dofs_x))
            self.vec_F0[ls_dofs_y] += np.repeat(force_vector[1], len(ls_dofs_y))
            self.vec_F0[ls_dofs_z] += np.repeat(force_vector[2], len(ls_dofs_z))

    def set_variation(self, vec_variation):
        self.vec_variation = vec_variation

    def compute_constant_F(self, n_steps):
        self.mat_F = np.transpose([np.transpose(self.vec_F0)] * n_steps)

    def compute_varying_F(self):
        n_steps = len(self.vec_variation)

        self.mat_F = np.zeros((self.vec_F0.shape[0], n_steps))

        for ii in range(n_steps):
            vec_F_ii = self.vec_F0 * self.vec_variation[ii]
            self.mat_F[:, ii] = vec_F_ii
