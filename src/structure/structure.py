import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from random_generators.matrices import SEplus


class Structure:
    def __init__(self, mesh, dirichlet_BC):
        self.mesh = mesh
        self.mesh.sort_elements_by_material()

        self.dirichlet_BC = dirichlet_BC

        self.n_total_dofs = mesh.n_total_dofs
        self.dirichlet_dofs = dirichlet_BC.n_dirichlet_dofs
        self.n_dirichlet_dofs = dirichlet_BC.n_dirichlet_dofs
        self.free_dofs = dirichlet_BC.compute_list_of_free_dofs(self.n_total_dofs)
        self.n_free_dofs = len(self.free_dofs)
        self.observed_dofs = mesh.observed_dofs
        self.n_observed_dofs = mesh.n_observed_dofs

        # Rayleigh damping parameters
        self.alpha_M = 0.0
        self.alpha_K = 0.0

        # Stochastic parameters
        self.dispersion_coefficient_M = 0.0
        self.dispersion_coefficient_K = 0.0

        ####
        # Structural matrices and vectors
        ####
        # Linear structural dynamics
        self.mat_M = None
        self.list_factorized_M_vectors = None

        self.mat_K = None
        self.list_factorized_K_vectors = None

        self.mat_D = None

        self.mat_Mff = None
        self.mat_Kff = None
        self.mat_Dff = None

        # Reduced-order models
        self.mat_modes = None
        self.mat_modes_f = None
        self.vec_eigenfreqs = None

        self.Mrom = None
        self.cholesky_Mrom = None

        self.Krom = None
        self.cholesky_Krom = None

        self.Drom = None
        self.cholesky_Drom = None

        self.Mrom_rand = None
        self.Krom_rand = None
        self.Drom_rand = None

        # Nonlinear statics
        self.list_factorized_KT_vectors = []
        self.mat_K_tangential = None
        self.mat_KTff = None

        self.list_factorized_Ktot_vectors = []
        self.vec_F_internal = None
        self.vec_Fint_f = None

    def set_rayleigh_parameters(self, alpha_M=0.0, alpha_K=0.0):
        self.alpha_M = alpha_M
        self.alpha_K = alpha_K

    def compute_M(self):
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.dofs_nums[ind_I])
                vec_cols = np.append(vec_cols, element.dofs_nums[ind_J])

                mat_Me = element.compute_mat_Me(material)

                vec_dataM = np.append(vec_dataM, mat_Me.flatten(order='F'))

        self.mat_M = csc_matrix((vec_dataM, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))

    def compute_K(self):
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataK = np.array([])

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.dofs_nums[ind_I])
                vec_cols = np.append(vec_cols, element.dofs_nums[ind_J])

                mat_Ke = element.compute_mat_Ke(material)

                vec_dataK = np.append(vec_dataK, mat_Ke.flatten(order='F'))

        self.mat_K = csc_matrix((vec_dataK, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))

    def compute_M_K(self):
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        vec_dataK = np.array([])

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
                vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])

                mat_Me, mat_Ke = element.compute_mat_Me_mat_Ke(material)

                vec_dataM = np.append(vec_dataM, mat_Me.flatten(order='F'))
                vec_dataK = np.append(vec_dataK, mat_Ke.flatten(order='F'))

        self.mat_M = csc_matrix((vec_dataM, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))
        self.mat_K = csc_matrix((vec_dataK, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))

    def compute_factorized_M_vectors(self):
        self.list_factorized_M_vectors = []

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            vec_rows = np.array([])
            vec_cols = np.array([])
            mat_M_vec_data = np.array([])

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.dofs_nums[ind_I])
                vec_cols = np.append(vec_cols, element.dofs_nums[ind_J])

                factorized_mat_Me = element.compute_factorized_mat_Me(material)

                mat_M_vec_data = np.append(mat_M_vec_data, factorized_mat_Me.flatten(order='F'))

            self.list_factorized_M_vectors.append((vec_rows, vec_cols, mat_M_vec_data))

    def compute_factorized_M(self):
        vec_rows = np.array([])
        vec_cols = np.array([])
        mat_M_vec_data = np.array([])

        for mm, material in enumerate(self.mesh.materials):
            rho = material.rho

            (vec_rows_mm, vec_cols_mm, mat_M_vec_data_mm) = self.list_factorized_M_vectors[mm]
            vec_rows = np.append(vec_rows, vec_rows_mm)
            vec_cols = np.append(vec_cols, vec_cols_mm)
            mat_M_vec_data = np.append(mat_M_vec_data, rho * mat_M_vec_data_mm)

        self.mat_M = csc_matrix((mat_M_vec_data, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))

    def compute_factorized_K_vectors(self):
        self.list_factorized_K_vectors = []

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            vec_rows = np.array([])
            vec_cols = np.array([])

            n_coeffs = material.n_factorized_mat_C

            list_factorized_mat_K_vec_data = [np.array([]) for _ in range(n_coeffs)]

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.dofs_nums[ind_I])
                vec_cols = np.append(vec_cols, element.dofs_nums[ind_J])

                list_factorized_mat_Ke = element.compute_list_factorized_mat_Ke(material)

                for ii, mat_Ke in enumerate(list_factorized_mat_Ke):
                    mat_K_vec_data = mat_Ke.flatten(order='F')
                    list_factorized_mat_K_vec_data[ii] = np.append(list_factorized_mat_K_vec_data[ii], mat_K_vec_data)

            self.list_factorized_K_vectors.append((vec_rows, vec_cols, list_factorized_mat_K_vec_data))

    def compute_factorized_K(self):
        vec_rows = np.array([])
        vec_cols = np.array([])
        mat_K_vec_data = np.array([])

        for mm, material in enumerate(self.mesh.materials):
            coeffs_mm = material.compute_factorized_coeffs()

            (vec_rows_mm, vec_cols_mm, list_factorized_mat_K_vec_data_mm) = self.list_factorized_K_vectors[mm]
            vec_rows = np.append(vec_rows, vec_rows_mm)
            vec_cols = np.append(vec_cols, vec_cols_mm)
            mat_K_vec_data_mm = np.zeros(vec_rows_mm.shape)

            for cc, coeff in enumerate(coeffs_mm):
                mat_K_vec_data_mm += coeff * list_factorized_mat_K_vec_data_mm[cc]

            mat_K_vec_data = np.append(mat_K_vec_data, mat_K_vec_data_mm)

        self.mat_K = csc_matrix((mat_K_vec_data, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))

    def compute_factorized_M_K_vectors(self):
        self.list_factorized_M_vectors = []
        self.list_factorized_K_vectors = []

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            vec_rows = np.array([])
            vec_cols = np.array([])
            mat_M_vec_data = np.array([])

            n_coeffs = material.n_factorized_mat_C

            list_factorized_mat_K_vec_data = [np.array([]) for _ in range(n_coeffs)]

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.dofs_nums[ind_I])
                vec_cols = np.append(vec_cols, element.dofs_nums[ind_J])

                factorized_mat_Me = element.compute_factorized_mat_Me(material)

                mat_M_vec_data = np.append(mat_M_vec_data, factorized_mat_Me.flatten(order='F'))

                list_factorized_mat_Ke = element.compute_list_factorized_mat_Ke(material)

                for ii, mat_Ke in enumerate(list_factorized_mat_Ke):
                    mat_K_vec_data = mat_Ke.flatten(order='F')
                    list_factorized_mat_K_vec_data[ii] = np.append(list_factorized_mat_K_vec_data[ii], mat_K_vec_data)

            self.list_factorized_M_vectors.append((vec_rows, vec_cols, mat_M_vec_data))
            self.list_factorized_K_vectors.append((vec_rows, vec_cols, list_factorized_mat_K_vec_data))

    def compute_factorized_M_K(self):
        vec_rows = np.array([])
        vec_cols = np.array([])
        mat_M_vec_data = np.array([])
        mat_K_vec_data = np.array([])

        for mm, material in enumerate(self.mesh.materials):
            rho = material.rho

            (_, _, mat_M_vec_data_mm) = self.list_factorized_M_vectors[mm]
            mat_M_vec_data = np.append(mat_M_vec_data, rho * mat_M_vec_data_mm)

            coeffs_mm = material.compute_factorized_coeffs()

            (vec_rows_mm, vec_cols_mm, list_factorized_mat_K_vec_data_mm) = self.list_factorized_K_vectors[mm]
            vec_rows = np.append(vec_rows, vec_rows_mm)
            vec_cols = np.append(vec_cols, vec_cols_mm)
            mat_K_vec_data_mm = np.zeros(vec_rows_mm.shape)

            for cc, coeff in enumerate(coeffs_mm):
                mat_K_vec_data_mm += coeff * list_factorized_mat_K_vec_data_mm[cc]

            mat_K_vec_data = np.append(mat_K_vec_data, mat_K_vec_data_mm)

        self.mat_M = csc_matrix((mat_M_vec_data, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))
        self.mat_K = csc_matrix((mat_K_vec_data, (vec_rows, vec_cols)),
                                shape=(self.n_total_dofs, self.n_total_dofs))

    def compute_D(self):
        if self.alpha_M != 0:
            self.mat_D = self.alpha_M * self.mat_M
            if self.alpha_K != 0:
                self.mat_D += self.alpha_K * self.mat_K
        else:
            if self.alpha_K != 0:
                self.mat_D = self.alpha_K * self.mat_K
            else:
                self.mat_D = np.zeros_like(self.mat_K)

    def apply_dirichlet_M(self):
        self.mat_Mff = self.mat_M[self.free_dofs, :][:, self.free_dofs]

    def apply_dirichlet_K(self):
        self.mat_Kff = self.mat_K[self.free_dofs, :][:, self.free_dofs]

    def apply_dirichlet_D(self):
        self.mat_Dff = self.mat_D[self.free_dofs, :][:, self.free_dofs]

    def compute_modes(self, n_modes):
        # Eigenvectors are mass-normalized

        self.compute_factorized_M_K_vectors()
        self.compute_factorized_M_K()
        self.apply_dirichlet_M()
        self.apply_dirichlet_K()

        (eigvals, eigvects) = eigsh(self.mat_Kff, n_modes, self.mat_Mff, which='SM')
        sort_indices = np.argsort(eigvals)
        eigvals = eigvals[sort_indices]
        eigvects = eigvects[:, sort_indices]

        self.vec_eigenfreqs = np.sqrt(eigvals) / (2 * np.pi)
        self.mat_modes_f = eigvects
        self.mat_modes = np.zeros((self.n_total_dofs, n_modes))
        self.mat_modes[self.free_dofs, :] = eigvects

    def compute_Drom(self):
        if self.alpha_M != 0:
            self.Drom = self.alpha_M * self.Mrom
            if self.alpha_K != 0:
                self.Drom += self.alpha_K * self.Krom
        else:
            if self.alpha_K != 0:
                self.Drom = self.alpha_K * self.Krom
            else:
                self.Drom = np.zeros_like(self.Krom)

    def compute_linear_ROM(self):
        self.Mrom = np.dot(self.mat_modes_f.transpose(), self.mat_Mff.dot(self.mat_modes_f))
        self.Krom = np.dot(self.mat_modes_f.transpose(), self.mat_Kff.dot(self.mat_modes_f))
        self.compute_Drom()

        if self.dispersion_coefficient_M > 0:
            self.cholesky_Mrom = np.linalg.cholesky(self.Mrom)
        if self.dispersion_coefficient_K > 0:
            self.cholesky_Krom = np.linalg.cholesky(self.Krom)

    def compute_linear_diagonal_ROM(self):
        self.Mrom = np.ones((self.vec_eigenfreqs.shape[0],))
        self.Krom = np.power(2 * np.pi * self.vec_eigenfreqs, 2)
        self.compute_Drom()

    def generate_random_Mrom(self):
        self.Mrom_rand = SEplus(self.cholesky_Mrom, self.dispersion_coefficient_M)

    def generate_random_Krom(self):
        self.Krom_rand = SEplus(self.cholesky_Krom, self.dispersion_coefficient_K)

    def compute_random_Drom(self):
        if self.alpha_M != 0:
            self.Drom_rand = self.alpha_M * self.Mrom_rand
            if self.alpha_K != 0:
                self.Drom_rand += self.alpha_K * self.Krom_rand
        else:
            if self.alpha_K != 0:
                self.Drom_rand = self.alpha_K * self.Krom_rand
            else:
                self.Drom_rand = np.zeros_like(self.Krom_rand)

    def generate_random_matrices(self, uncertainty_type):
        if uncertainty_type == "parametric":
            for material in self.mesh.materials:
                material.generate_random_material_parameters()

            self.compute_factorized_M_K()
            self.apply_dirichlet_M()
            self.apply_dirichlet_K()
            self.compute_linear_ROM()

            self.Mrom_rand = self.Mrom
            self.Krom_rand = self.Krom
            self.compute_random_Drom()

            for material in self.mesh.materials:
                material.restore_mean_material_parameters()

        elif uncertainty_type == "nonparametric":
            if self.dispersion_coefficient_M > 0:
                self.generate_random_Mrom()
            else:
                self.Mrom_rand = self.Mrom

            if self.dispersion_coefficient_K > 0:
                self.generate_random_Krom()
            else:
                self.Krom_rand = self.Krom

            if self.dispersion_coefficient_M > 0 or self.dispersion_coefficient_K > 0:
                self.compute_random_Drom()
            else:
                self.Drom_rand = self.Drom

        elif uncertainty_type == "generalized":
            for material in self.mesh.materials:
                material.generate_random_material_parameters()

            self.compute_factorized_M_K()
            self.apply_dirichlet_M()
            self.apply_dirichlet_K()
            self.compute_linear_ROM()

            if self.dispersion_coefficient_M > 0:
                self.generate_random_Mrom()
            else:
                self.Mrom_rand = self.Mrom

            if self.dispersion_coefficient_K > 0:
                self.generate_random_Krom()
            else:
                self.Krom_rand = self.Krom

            if self.dispersion_coefficient_M > 0 or self.dispersion_coefficient_K > 0:
                self.compute_random_Drom()
            else:
                self.Drom_rand = self.Drom

            for material in self.mesh.materials:
                material.restore_mean_material_parameters()

    def compute_factorized_KT_Ktot_vectors(self, vec_U):
        self.list_factorized_Ktot_vectors = []
        self.list_factorized_KT_vectors = []

        for mm, element_list in enumerate(self.mesh.elements_by_material):
            material = self.mesh.materials[mm]

            vec_rows = np.array([])
            vec_cols = np.array([])

            n_coeffs = material.n_factorized_mat_C

            list_factorized_mat_Ktot_vec_data = [np.array([]) for _ in range(n_coeffs)]
            list_factorized_mat_KT_vec_data = [np.array([]) for _ in range(n_coeffs)]

            for element in element_list:
                n_dofs = element.n_dofs

                ind_I = list(range(n_dofs)) * n_dofs
                ind_J = []
                for ii in range(n_dofs):
                    ind_J.extend([ii] * n_dofs)

                vec_rows = np.append(vec_rows, element.dofs_nums[ind_I])
                vec_cols = np.append(vec_cols, element.dofs_nums[ind_J])

                vec_Ue = vec_U[element.dofs_nums]

                list_factorized_mat_KT1e, \
                list_factorized_mat_KT2e, \
                list_factorized_mat_KT3e, \
                list_factorized_mat_Ktote = element.compute_list_factorized_mat_KTe(vec_Ue, material)

                for ii in range(n_coeffs):
                    mat_Ktote = list_factorized_mat_Ktote[ii]

                    mat_Ktot_vec_data = mat_Ktote.flatten(order='F')

                    mat_KT1e = list_factorized_mat_KT1e[ii]
                    mat_KT2e = list_factorized_mat_KT2e[ii]
                    mat_KT3e = list_factorized_mat_KT3e[ii]

                    mat_KT1_vec_data = mat_KT1e.flatten(order='F')
                    mat_KT2_vec_data = mat_KT2e.flatten(order='F')
                    mat_KT3_vec_data = mat_KT3e.flatten(order='F')

                    list_factorized_mat_Ktot_vec_data[ii] = np.append(list_factorized_mat_Ktot_vec_data[ii],
                                                                      mat_Ktot_vec_data)

                    list_factorized_mat_KT_vec_data[ii] = np.append(list_factorized_mat_KT_vec_data[ii],
                                                                    mat_KT1_vec_data
                                                                    + mat_KT2_vec_data
                                                                    + mat_KT3_vec_data)

            self.list_factorized_Ktot_vectors.append((vec_rows, vec_cols, list_factorized_mat_Ktot_vec_data))
            self.list_factorized_KT_vectors.append((vec_rows, vec_cols, list_factorized_mat_KT_vec_data))

    def compute_factorized_KT_Fint(self, vec_U):
        vec_rows_KT = np.array([])
        vec_cols_KT = np.array([])

        mat_Ktot_vec_data = np.array([])
        mat_KT_vec_data = np.array([])

        for mm, material in enumerate(self.mesh.materials):
            coeffs_mm = material.compute_factorized_coeffs()

            (vec_rows_KT_mm, vec_cols_KT_mm, list_factorized_mat_Ktot_vec_data_mm) = \
            self.list_factorized_Ktot_vectors[mm]

            mat_Ktot_vec_data_mm = np.zeros(vec_rows_KT_mm.shape)

            (_, _, list_factorized_mat_KT_vec_data_mm) = self.list_factorized_KT_vectors[mm]

            vec_rows_KT = np.append(vec_rows_KT, vec_rows_KT_mm)
            vec_cols_KT = np.append(vec_cols_KT, vec_cols_KT_mm)
            mat_KT_vec_data_mm = np.zeros(vec_rows_KT_mm.shape)

            for cc, coeff in enumerate(coeffs_mm):
                mat_Ktot_vec_data_mm += coeff * list_factorized_mat_Ktot_vec_data_mm[cc]
                mat_KT_vec_data_mm += coeff * list_factorized_mat_KT_vec_data_mm[cc]

            mat_Ktot_vec_data = np.append(mat_Ktot_vec_data, mat_Ktot_vec_data_mm)
            mat_KT_vec_data = np.append(mat_KT_vec_data, mat_KT_vec_data_mm)

        vec_rows = vec_rows_KT
        vec_cols = vec_cols_KT
        vec_data_KT = mat_KT_vec_data
        vec_data_Ktot = mat_Ktot_vec_data

        self.mat_K_tangential = csc_matrix((vec_data_KT, (vec_rows, vec_cols)),
                                           shape=(self.n_total_dofs, self.n_total_dofs))
        mat_Ktot = csc_matrix((vec_data_Ktot, (vec_rows, vec_cols)),
                              shape=(self.n_total_dofs, self.n_total_dofs))

        self.vec_F_internal = mat_Ktot.dot(vec_U)

    def apply_dirichlet_KT(self):
        self.mat_KTff = self.mat_K_tangential[self.free_dofs, :][:, self.free_dofs]

    def apply_dirichlet_Fint(self):
        self.vec_Fint_f = self.vec_F_internal[self.free_dofs]

    def apply_dirichlet_KT_Fint(self):
        self.mat_KTff = self.mat_K_tangential[self.free_dofs, :][:, self.free_dofs]
        self.vec_Fint_f = self.vec_F_internal[self.free_dofs]
