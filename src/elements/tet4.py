import numpy as np
from elements.element import Element

# Constant finite element matrices

s = np.sqrt(2)/2
mat_G = np.zeros((6, 9))
mat_G[0, 0] = 1
mat_G[1, 4] = 1
mat_G[2, 8] = 1
mat_G[3, 2] = s
mat_G[3, 6] = s
mat_G[4, 1] = s
mat_G[4, 3] = s
mat_G[5, 5] = s
mat_G[5, 7] = s

mat_P = np.zeros((9, 9))
mat_P[0, 0] = 1
mat_P[1, 3] = 1
mat_P[2, 6] = 1
mat_P[3, 1] = 1
mat_P[4, 4] = 1
mat_P[5, 7] = 1
mat_P[6, 2] = 1
mat_P[7, 5] = 1
mat_P[8, 8] = 1

# Reference coordinates of the nodes

nodes_reference_coords = np.array([[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])

# Faces defined by reference nodes

faces = [[0, 1, 2],
         [0, 1, 3],
         [0, 2, 3],
         [1, 2, 3]]

# Coefficients for the shape functions on the reference element

x0 = nodes_reference_coords[0, 0]
y0 = nodes_reference_coords[0, 1]
z0 = nodes_reference_coords[0, 2]
x1 = nodes_reference_coords[1, 0]
y1 = nodes_reference_coords[1, 1]
z1 = nodes_reference_coords[1, 2]
x2 = nodes_reference_coords[2, 0]
y2 = nodes_reference_coords[2, 1]
z2 = nodes_reference_coords[2, 2]
x3 = nodes_reference_coords[3, 0]
y3 = nodes_reference_coords[3, 1]
z3 = nodes_reference_coords[3, 2]

mat_A = np.array([[1, x0, y0, z0],
                  [1, x1, y1, z1],
                  [1, x2, y2, z2],
                  [1, x3, y3, z3]])

shapefun_coeffs = np.linalg.solve(mat_A, np.eye(4))

# Gauss integration points and weights

n_gauss = 4
gauss = [(np.array([0.5854101966, 0.1381966011, 0.1381966011]), 0.0416666667),
         (np.array([0.1381966011, 0.5854101966, 0.1381966011]), 0.0416666667),
         (np.array([0.1381966011, 0.1381966011, 0.5854101966]), 0.0416666667),
         (np.array([0.1381966011, 0.1381966011, 0.1381966011]), 0.0416666667)]


####
# Functions used for calculating element matrices at gauss points

def shapefun_value(node_idx, reference_coords):
    # N_i(x, y, z) = a + b*x + c*y + d*z

    x = reference_coords[0]
    y = reference_coords[1]
    z = reference_coords[2]

    value = shapefun_coeffs[0, node_idx] \
            + shapefun_coeffs[1, node_idx] * x \
            + shapefun_coeffs[2, node_idx] * y \
            + shapefun_coeffs[3, node_idx] * z

    return value


def derivative_shapefun_value(shapefun_idx, derivative_coord_idx):
    # dNdx_i(x, y, z) = b
    # dNdy_i(x, y, z) = c
    # dNdz_i(x, y, z) = d

    # derivative_coord_idx: 1 -> derivative with respect to x
    # derivative_coord_idx: 2 -> derivative with respect to y
    # derivative_coord_idx: 3 -> derivative with respect to z

    value = shapefun_coeffs[derivative_coord_idx, shapefun_idx]

    return value


def compute_mat_Ee(reference_coords):
    mat_I = np.eye(3)
    mat_E0 = shapefun_value(0, reference_coords) * mat_I
    mat_E1 = shapefun_value(1, reference_coords) * mat_I
    mat_E2 = shapefun_value(2, reference_coords) * mat_I
    mat_E3 = shapefun_value(3, reference_coords) * mat_I

    mat_Ee = np.concatenate((mat_E0, mat_E1, mat_E2, mat_E3), axis=1)

    return mat_Ee


def compute_mat_De():
    mat_I = np.eye(3)

    mat_D0dx = derivative_shapefun_value(0, 1) * mat_I
    mat_D1dx = derivative_shapefun_value(1, 1) * mat_I
    mat_D2dx = derivative_shapefun_value(2, 1) * mat_I
    mat_D3dx = derivative_shapefun_value(3, 1) * mat_I

    mat_Ddx = np.concatenate((mat_D0dx, mat_D1dx, mat_D2dx, mat_D3dx), axis=1)

    mat_D0dy = derivative_shapefun_value(0, 2) * mat_I
    mat_D1dy = derivative_shapefun_value(1, 2) * mat_I
    mat_D2dy = derivative_shapefun_value(2, 2) * mat_I
    mat_D3dy = derivative_shapefun_value(3, 2) * mat_I

    mat_Ddy = np.concatenate((mat_D0dy, mat_D1dy, mat_D2dy, mat_D3dy), axis=1)

    mat_D0dz = derivative_shapefun_value(0, 3) * mat_I
    mat_D1dz = derivative_shapefun_value(1, 3) * mat_I
    mat_D2dz = derivative_shapefun_value(2, 3) * mat_I
    mat_D3dz = derivative_shapefun_value(3, 3) * mat_I

    mat_Ddz = np.concatenate((mat_D0dz, mat_D1dz, mat_D2dz, mat_D3dz), axis=1)

    mat_De = np.concatenate((mat_Ddx, mat_Ddy, mat_Ddz), axis=0)

    return mat_De


####
# Element matrices at gauss points

list_mat_EeTEe_gauss = []
for ii in range(n_gauss):
    gauss_point_ii = gauss[ii][0]
    mat_Ee_ii = compute_mat_Ee(gauss_point_ii)
    list_mat_EeTEe_gauss.append(np.dot(mat_Ee_ii.transpose(), mat_Ee_ii))

mat_De_gauss = compute_mat_De()


####
# Tet4 element class

class Tet4(Element):
    def __init__(self, number, material_id, nodes_nums, nodes_coords):
        super(Tet4, self).__init__(number, material_id, nodes_nums, nodes_coords)

        self.nodes_reference_coords = nodes_reference_coords
        self.faces = faces

        self.det_J = None
        self.mat_invJJJ = None

    def compute_jacobian_at_gauss_point(self):
        mat_J1 = np.dot(mat_De_gauss[:3, :],  self.vec_nodes_coords)
        mat_J2 = np.dot(mat_De_gauss[3:6, :], self.vec_nodes_coords)
        mat_J3 = np.dot(mat_De_gauss[6:, :],  self.vec_nodes_coords)

        mat_J = np.vstack((mat_J1, mat_J2, mat_J3))
        det_J = np.linalg.det(mat_J)

        if det_J < 0:
            print('Element ', self.number)
            raise ValueError('Element has negative jacobian.')
        elif det_J == 0:
            print('Element ', self.number)
            raise ValueError('Element has zero jacobian.')

        mat_invJ = np.linalg.inv(mat_J)
        mat_invJJJ = np.zeros((9, 9))
        mat_invJJJ[0:3, 0:3] = mat_invJ
        mat_invJJJ[3:6, 3:6] = mat_invJ
        mat_invJJJ[6:9, 6:9] = mat_invJ

        return det_J, mat_invJJJ

    def compute_jacobian_at_reference_coords(self):
        mat_J1 = np.dot(mat_De_gauss[:3, :],  self.vec_nodes_coords)
        mat_J2 = np.dot(mat_De_gauss[3:6, :], self.vec_nodes_coords)
        mat_J3 = np.dot(mat_De_gauss[6:, :],  self.vec_nodes_coords)

        mat_J = np.vstack((mat_J1, mat_J2, mat_J3))
        det_J = np.linalg.det(mat_J)

        mat_invJ = np.linalg.inv(mat_J)
        mat_invJJJ = np.zeros((9, 9))
        mat_invJJJ[0:3, 0:3] = mat_invJ
        mat_invJJJ[3:6, 3:6] = mat_invJ
        mat_invJJJ[6:9, 6:9] = mat_invJ

        return det_J, mat_invJJJ, mat_De_gauss

    def compute_mat_Me(self, material):
        mat_Me = np.zeros((self.n_dofs, self.n_dofs))

        det_J, _ = self.compute_jacobian_at_gauss_point()

        for gg, (_, gauss_weight) in enumerate(gauss):
            mat_Me += gauss_weight * material.rho * det_J * list_mat_EeTEe_gauss[gg]

        return mat_Me

    def compute_mat_Ke(self, material):
        mat_Ke = np.zeros((self.n_dofs, self.n_dofs))

        det_J, mat_invJJJ = self.compute_jacobian_at_gauss_point()
        mat_B = np.dot(mat_G, np.dot(mat_invJJJ, np.dot(mat_P, mat_De_gauss)))

        for gg, (_, gauss_weight) in enumerate(gauss):
            mat_Ke += gauss_weight * det_J * np.dot(mat_B.transpose(), np.dot(material.mat_C, mat_B))

        return mat_Ke

    def compute_mat_Me_mat_Ke(self, material):
        mat_Me = np.zeros((self.n_dofs, self.n_dofs))
        mat_Ke = np.zeros((self.n_dofs, self.n_dofs))

        det_J, mat_invJJJ = self.compute_jacobian_at_gauss_point()
        mat_B = np.dot(mat_G, np.dot(mat_invJJJ, np.dot(mat_P, mat_De_gauss)))

        for gg, (_, gauss_weight) in enumerate(gauss):
            mat_Me += gauss_weight * material.rho * det_J * list_mat_EeTEe_gauss[gg]
            mat_Ke += gauss_weight * det_J * np.dot(mat_B.transpose(), np.dot(material.mat_C, mat_B))

        return mat_Me, mat_Ke

    def compute_factorized_mat_Me(self, _):
        factorized_mat_Me = np.zeros((self.n_dofs, self.n_dofs))

        det_J, _ = self.compute_jacobian_at_gauss_point()

        for gg, (_, gauss_weight) in enumerate(gauss):
            factorized_mat_Me += gauss_weight * det_J * list_mat_EeTEe_gauss[gg]

        return factorized_mat_Me

    def compute_list_factorized_mat_Ke(self, material):
        n_factors = material.n_factorized_mat_C
        list_factorized_mat_C = material.list_factorized_mat_C

        list_factorized_mat_Ke = [np.zeros((self.n_dofs, self.n_dofs)) for _ in range(n_factors)]

        det_J, mat_invJJJ = self.compute_jacobian_at_gauss_point()
        mat_B = np.dot(mat_G, np.dot(mat_invJJJ, np.dot(mat_P, mat_De_gauss)))

        for gg, (_, gauss_weight) in enumerate(gauss):
            for kk, factorized_mat_C in enumerate(list_factorized_mat_C):
                list_factorized_mat_Ke[kk] += gauss_weight * det_J \
                                              * np.dot(mat_B.transpose(), np.dot(factorized_mat_C, mat_B))

        return list_factorized_mat_Ke

    def compute_factorized_mat_Me_mat_Ke(self, material):
        n_factors_K = material.n_factorized_mat_C
        list_factorized_mat_C = material.list_factorized_mat_C

        factorized_mat_Me = np.zeros((self.n_dofs, self.n_dofs))
        list_factorized_mat_Ke = [np.zeros((self.n_dofs, self.n_dofs)) for _ in range(n_factors_K)]

        det_J, mat_invJJJ = self.compute_jacobian_at_gauss_point()
        mat_B = np.dot(mat_G, np.dot(mat_invJJJ, np.dot(mat_P, mat_De_gauss)))

        for gg, (_, gauss_weight) in enumerate(gauss):
            factorized_mat_Me += gauss_weight * det_J * list_mat_EeTEe_gauss[gg]
            for kk, factorized_mat_C in enumerate(list_factorized_mat_C):
                list_factorized_mat_Ke[kk] += gauss_weight * det_J \
                                              * np.dot(mat_B.transpose(), np.dot(factorized_mat_C, mat_B))

        return factorized_mat_Me, list_factorized_mat_Ke

    def compute_list_factorized_mat_KTe(self, vec_Ue, material):
        n_factors = material.n_factorized_mat_C
        list_factorized_mat_C = material.list_factorized_mat_C

        list_factorized_mat_Ktote = [np.zeros((self.n_dofs, self.n_dofs)) for _ in range(n_factors)]
        list_factorized_mat_KT1e = [np.zeros((self.n_dofs, self.n_dofs)) for _ in range(n_factors)]
        list_factorized_mat_KT2e = [np.zeros((self.n_dofs, self.n_dofs)) for _ in range(n_factors)]
        list_factorized_mat_KT3e = [np.zeros((self.n_dofs, self.n_dofs)) for _ in range(n_factors)]

        det_J, mat_invJJJ = self.compute_jacobian_at_gauss_point()

        mat_grad = np.dot(mat_invJJJ, np.dot(mat_P, mat_De_gauss))
        mat_B = np.dot(mat_G, mat_grad)

        vec_grad_Ue = np.dot(mat_grad, vec_Ue)
        vec_D_Ue = np.dot(mat_P, vec_grad_Ue)
        mat_D_Ue = np.zeros((6, 9))
        mat_D_Ue[0, 0:3] = vec_D_Ue[0:3]
        mat_D_Ue[1, 3:6] = vec_D_Ue[3:6]
        mat_D_Ue[2, 6:] = vec_D_Ue[6:]
        mat_D_Ue[3, 0:3] = vec_D_Ue[3:6] / np.sqrt(2)
        mat_D_Ue[3, 3:6] = vec_D_Ue[0:3] / np.sqrt(2)
        mat_D_Ue[4, 0:3] = vec_D_Ue[6:] / np.sqrt(2)
        mat_D_Ue[4, 6:] = vec_D_Ue[0:3] / np.sqrt(2)
        mat_D_Ue[5, 3:6] = vec_D_Ue[6:] / np.sqrt(2)
        mat_D_Ue[5, 6:] = vec_D_Ue[3:6] / np.sqrt(2)

        mat_H = np.dot(mat_D_Ue, np.dot(mat_P, mat_grad))
        mat_F = np.eye(3) + np.reshape(vec_grad_Ue, (3, 3))
        mat_E = 0.5 * (np.dot(mat_F.transpose(), mat_F) - np.eye(3))

        vec_E = np.zeros((6,))
        vec_E[0] = mat_E[0, 0]
        vec_E[1] = mat_E[1, 1]
        vec_E[2] = mat_E[2, 2]
        vec_E[3] = np.sqrt(2) * mat_E[1, 2]
        vec_E[4] = np.sqrt(2) * mat_E[0, 2]
        vec_E[5] = np.sqrt(2) * mat_E[0, 1]

        for gg, (_, gauss_weight) in enumerate(gauss):
            for kk, factorized_mat_C in enumerate(list_factorized_mat_C):
                mat_KT1e_kk = gauss_weight * det_J * np.dot(mat_B.transpose(), np.dot(factorized_mat_C, mat_B))

                list_factorized_mat_KT1e[kk] += mat_KT1e_kk

                list_factorized_mat_KT2e[kk] += gauss_weight * det_J \
                                                * (np.dot(mat_B.transpose(), np.dot(factorized_mat_C, mat_H))
                                                   + np.dot(mat_H.transpose(), np.dot(factorized_mat_C, mat_B)))

                vec_piola2 = np.dot(factorized_mat_C, vec_E)
                mat_piola2 = np.zeros((3, 3))
                mat_piola2[0, 0] = vec_piola2[0]
                mat_piola2[1, 1] = vec_piola2[1]
                mat_piola2[2, 2] = vec_piola2[2]
                mat_piola2[1, 2] = vec_piola2[3]
                mat_piola2[0, 2] = vec_piola2[4]
                mat_piola2[0, 1] = vec_piola2[5]

                mat_pi = np.zeros((9, 9))
                mat_pi[0:3, 0:3] = mat_piola2
                mat_pi[3:6, 3:6] = mat_piola2
                mat_pi[6:, 6:] = mat_piola2

                list_factorized_mat_KT3e[kk] += gauss_weight * det_J \
                                                * (np.dot(mat_grad.transpose(), np.dot(mat_pi, mat_grad))
                                                   + np.dot(mat_H.transpose(), np.dot(factorized_mat_C, mat_H)))

                list_factorized_mat_Ktote[kk] += mat_KT1e_kk
                list_factorized_mat_Ktote[kk] += gauss_weight * det_J \
                                                 * (0.5 * np.dot(mat_B.transpose(), np.dot(factorized_mat_C, mat_H))
                                                    + np.dot(mat_H.transpose(), np.dot(factorized_mat_C, mat_B)))
                list_factorized_mat_Ktote[kk] += gauss_weight * det_J \
                                                 * 0.5 * np.dot(mat_H.transpose(), np.dot(factorized_mat_C, mat_H))

        return list_factorized_mat_KT1e, list_factorized_mat_KT2e, list_factorized_mat_KT3e, list_factorized_mat_Ktote

    def compute_element_strain_stress_at_reference_coords(self, _, vec_Ue, material):
        _, mat_invJJJ, mat_De_coords = self.compute_jacobian_at_reference_coords()
        mat_B = np.dot(mat_G, np.dot(mat_invJJJ, np.dot(mat_P, mat_De_coords)))

        vec_strain = np.dot(mat_B, vec_Ue)
        vec_stress = np.dot(material.mat_C, vec_strain)

        return vec_strain, vec_stress
