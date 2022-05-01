##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM code available at:                   #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import numpy as np

import importlib.util
spec1 = importlib.util.spec_from_file_location("element", "../elements/element.py")
element = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(element)

# functions


def tet4_shapefun_value(index, reference_coords):
    # N_i(x, y, z) = a + b*x + c*y + d*z
    
    x = reference_coords[0]
    y = reference_coords[1]
    z = reference_coords[2]
    
    value = tet4_shapefun_coeffs[0, index]\
            + tet4_shapefun_coeffs[1, index] * x\
            + tet4_shapefun_coeffs[2, index] * y\
            + tet4_shapefun_coeffs[3, index] * z
    
    return value


def tet4_derivative_shapefun_value(index_shapefun, index_coord):
    # dNdx_i(x, y, z) = b
    # dNdy_i(x, y, z) = c
    # dNdz_i(x, y, z) = d
    
    # index_coord: 1 -> derivative with respect to x
    # index_coord: 2 -> derivative with respect to y
    # index_coord: 3 -> derivative with respect to z
    
    value = tet4_shapefun_coeffs[index_coord, index_shapefun]
    
    return value


def tet4_compute_mat_Ee(reference_coords):
    mat_I = np.eye(3)
    mat_E0 = tet4_shapefun_value(0, reference_coords) * mat_I
    mat_E1 = tet4_shapefun_value(1, reference_coords) * mat_I
    mat_E2 = tet4_shapefun_value(2, reference_coords) * mat_I
    mat_E3 = tet4_shapefun_value(3, reference_coords) * mat_I
    
    tet4_mat_Ee = np.concatenate((mat_E0, mat_E1, mat_E2, mat_E3), axis=1)
    
    return tet4_mat_Ee


def tet4_compute_mat_De():
    mat_I = np.eye(3)
    
    mat_D0dx = tet4_derivative_shapefun_value(0, 1) * mat_I
    mat_D1dx = tet4_derivative_shapefun_value(1, 1) * mat_I
    mat_D2dx = tet4_derivative_shapefun_value(2, 1) * mat_I
    mat_D3dx = tet4_derivative_shapefun_value(3, 1) * mat_I
    
    mat_Ddx = np.concatenate((mat_D0dx, mat_D1dx, mat_D2dx, mat_D3dx), axis=1)
    
    mat_D0dy = tet4_derivative_shapefun_value(0, 2) * mat_I
    mat_D1dy = tet4_derivative_shapefun_value(1, 2) * mat_I
    mat_D2dy = tet4_derivative_shapefun_value(2, 2) * mat_I
    mat_D3dy = tet4_derivative_shapefun_value(3, 2) * mat_I
    
    mat_Ddy = np.concatenate((mat_D0dy, mat_D1dy, mat_D2dy, mat_D3dy), axis=1)
    
    mat_D0dz = tet4_derivative_shapefun_value(0, 3) * mat_I
    mat_D1dz = tet4_derivative_shapefun_value(1, 3) * mat_I
    mat_D2dz = tet4_derivative_shapefun_value(2, 3) * mat_I
    mat_D3dz = tet4_derivative_shapefun_value(3, 3) * mat_I
    
    mat_Ddz = np.concatenate((mat_D0dz, mat_D1dz, mat_D2dz, mat_D3dz), axis=1)
            
    tet4_mat_De = np.concatenate((mat_Ddx, mat_Ddy, mat_Ddz), axis=0) 
    
    return tet4_mat_De


# constant matrices

s = np.sqrt(2)/2
mat_G = np.zeros((6, 9))
mat_G[0,0] = 1
mat_G[1,4] = 1
mat_G[2,8] = 1
mat_G[3,2] = s
mat_G[3,6] = s
mat_G[4,1] = s
mat_G[4,3] = s
mat_G[5,5] = s
mat_G[5,7] = s

mat_P = np.zeros((9, 9))
mat_P[0,0] = 1
mat_P[1,3] = 1
mat_P[2,6] = 1
mat_P[3,1] = 1
mat_P[4,4] = 1
mat_P[5,7] = 1
mat_P[6,2] = 1
mat_P[7,5] = 1
mat_P[8,8] = 1

# Tet4 parameters and matrices

tet4_n_nodes = 4
tet4_n_dofs = tet4_n_nodes * 3
tet4_nodes_reference_coords =\
    np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1]])
tet4_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

tet4_n_gauss = 4
tet4_gauss = [(np.array([0.5854101966, 0.1381966011, 0.1381966011]), 0.0416666667),\
              (np.array([0.1381966011, 0.5854101966, 0.1381966011]), 0.0416666667),\
              (np.array([0.1381966011, 0.1381966011, 0.5854101966]), 0.0416666667),\
              (np.array([0.1381966011, 0.1381966011, 0.1381966011]), 0.0416666667)]

# tet4_n_gauss = 1
# tet4_gauss = [(np.array([1.0/4, 1.0/4, 1.0/4]), 1.0/6)]

x0 = tet4_nodes_reference_coords[0,0]
y0 = tet4_nodes_reference_coords[0,1]
z0 = tet4_nodes_reference_coords[0,2]
x1 = tet4_nodes_reference_coords[1,0]
y1 = tet4_nodes_reference_coords[1,1]
z1 = tet4_nodes_reference_coords[1,2]
x2 = tet4_nodes_reference_coords[2,0]
y2 = tet4_nodes_reference_coords[2,1]
z2 = tet4_nodes_reference_coords[2,2]
x3 = tet4_nodes_reference_coords[3,0]
y3 = tet4_nodes_reference_coords[3,1]
z3 = tet4_nodes_reference_coords[3,2]

tet4_mat_A = np.array([[1, x0, y0, z0],\
                       [1, x1, y1, z1],\
                       [1, x2, y2, z2],\
                       [1, x3, y3, z3]])
tet4_mat_I = np.eye(4)

tet4_shapefun_coeffs = np.linalg.solve(tet4_mat_A, tet4_mat_I)
    
tet4_mats_EeTEe_gauss = []
for ii in range(tet4_n_gauss):
    gauss_point_ii = tet4_gauss[ii][0]
    mat_Ee_ii = tet4_compute_mat_Ee(gauss_point_ii)
    tet4_mats_EeTEe_gauss.append(np.dot(mat_Ee_ii.transpose(), mat_Ee_ii))

tet4_mat_De_gauss = tet4_compute_mat_De()

##############################################################################


def get_gauss():
    return tet4_gauss


class Tet4(element.Element):
    def __init__(self, material, nodes_coords):
        super(Tet4, self).__init__(material)

        self.__list_factorized_mat_KT3e = None
        self.__list_factorized_mat_KT2e = None
        self.__list_factorized_mat_KT1e = None
        self.__list_factorized_mat_Ktote = None
        self.__list_factorized_mat_Ke = None
        self.__factorized_mat_Me = None
        self.__mat_Ke = None
        self.__mat_Me = None
        self.__nodes_coords = nodes_coords
        self.__vec_nodes_coords = np.reshape(nodes_coords, tet4_n_dofs)
        self.__nodes_reference_coords = tet4_nodes_reference_coords

    def get_nodes_reference_coords(self):
        return self.__nodes_reference_coords

    def get_n_nodes(self):
        return tet4_n_nodes
    
    def get_n_dofs(self):
        return tet4_n_dofs
    
    def get_nodes_coords(self):
        return self.__nodes_coords
    
    def get_vec_nodes_coords(self):
        return self.__vec_nodes_coords
    
    def set_nodes_coords(self, nodes_coords):
        self.__nodes_coords = nodes_coords
        self.__vec_nodes_coords = np.reshape(nodes_coords, tet4_n_dofs)
    
    def get_nodes_reference_coords(self):
        return tet4_nodes_reference_coords
    
    def get_faces(self):
        return tet4_faces
    
    def get_n_gauss(self):
        return tet4_n_gauss

    def __compute_jacobian(self):
        mat_J1 = np.dot(tet4_mat_De_gauss[:3 , :], self.__vec_nodes_coords)
        mat_J2 = np.dot(tet4_mat_De_gauss[3:6, :], self.__vec_nodes_coords)
        mat_J3 = np.dot(tet4_mat_De_gauss[6: , :], self.__vec_nodes_coords)
                
        self.__mat_J = np.vstack((mat_J1, mat_J2, mat_J3))
                        
        self.__det_J = np.linalg.det(self.__mat_J)
        
        if self.__det_J < 0:
            print('Element ', self.get_element_number())
            raise ValueError('Element has negative jacobian.')
        elif self.__det_J == 0:
            print('Element ', self.get_element_number())
            raise ValueError('Element has zero jacobian.')
        
        mat_invJ = np.linalg.inv(self.__mat_J)
        self.__mat_invJJJ = np.zeros((9, 9))
        self.__mat_invJJJ[0:3, 0:3] = mat_invJ
        self.__mat_invJJJ[3:6, 3:6] = mat_invJ
        self.__mat_invJJJ[6:9, 6:9] = mat_invJ

    def __compute_invJJJ_at_node(self):
        mat_J1 = np.dot(tet4_mat_De_gauss[:3, :], self.__vec_nodes_coords)
        mat_J2 = np.dot(tet4_mat_De_gauss[3:6, :], self.__vec_nodes_coords)
        mat_J3 = np.dot(tet4_mat_De_gauss[6:, :], self.__vec_nodes_coords)

        mat_J = np.vstack((mat_J1, mat_J2, mat_J3))

        mat_invJ = np.linalg.inv(mat_J)
        mat_invJJJ = np.zeros((9, 9))
        mat_invJJJ[0:3, 0:3] = mat_invJ
        mat_invJJJ[3:6, 3:6] = mat_invJ
        mat_invJJJ[6:9, 6:9] = mat_invJ

        return mat_invJJJ
        
    def get_mat_J(self):
        return self.__mat_J
    
    def get_det_J(self):
        return self.__det_J
    
    def get_mat_invJJJ(self):
        return self.__mat_invJJJ
    
    def compute_mat_Me(self):
        self.__mat_Me = np.zeros((tet4_n_dofs, tet4_n_dofs))
        
        self.__compute_jacobian()
        
        counter = 0
        for g in get_gauss():
            gauss_weight = g[1]
            self.__mat_Me += gauss_weight * self.get_material().get_rho() \
                             * self.__det_J * tet4_mats_EeTEe_gauss[counter]
            counter += 1
            
    def compute_mat_Ke(self):
        self.__mat_Ke = np.zeros((tet4_n_dofs, tet4_n_dofs))
        
        self.__compute_jacobian()
        mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, tet4_mat_De_gauss)))
        
        for g in get_gauss():
            gauss_weight = g[1]
            self.__mat_Ke += gauss_weight * self.__det_J \
                             * np.dot(mat_B.transpose(), np.dot(self.get_material().get_mat_C(), mat_B))
    
    def compute_mat_Me_mat_Ke(self):
        self.__mat_Me = np.zeros((tet4_n_dofs, tet4_n_dofs))
        self.__mat_Ke = np.zeros((tet4_n_dofs, tet4_n_dofs))
        
        self.__compute_jacobian()
        mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, tet4_mat_De_gauss)))
        
        counter = 0
        for g in get_gauss():
            gauss_weight = g[1]
            self.__mat_Me += gauss_weight * self.get_material().get_rho() \
                             * self.__det_J * tet4_mats_EeTEe_gauss[counter]
            self.__mat_Ke += gauss_weight * self.__det_J \
                             * np.dot(mat_B.transpose(), np.dot(self.get_material().get_mat_C(), mat_B))
            counter += 1
    
    def get_mat_Me(self):
        return self.__mat_Me
    
    def get_mat_Ke(self):
        return self.__mat_Ke
    
    def compute_factorized_mat_Me(self):                        
        self.__factorized_mat_Me = np.zeros((tet4_n_dofs, tet4_n_dofs))
        
        self.__compute_jacobian()
        
        counter = 0
        for g in get_gauss():
            gauss_weight = g[1]
            self.__factorized_mat_Me += gauss_weight * self.__det_J * tet4_mats_EeTEe_gauss[counter]
            
            counter += 1
            
    def compute_list_factorized_mat_Ke(self):
        list_factorized_mat_C = self.get_material().get_factorized_mat_C()
        
        n_materials = self.get_material().get_n_factorized_mat_C()
        
        self.__list_factorized_mat_Ke = [np.zeros((tet4_n_dofs, tet4_n_dofs)) for _ in range(n_materials)]
                
        self.__compute_jacobian()
        mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, tet4_mat_De_gauss)))
        
        counter = 0
        for g in get_gauss():
            gauss_weight = g[1]
            
            for kk in range(n_materials):
                mat_C = list_factorized_mat_C[kk]
                self.__list_factorized_mat_Ke[kk] += gauss_weight * self.__det_J \
                                                     * np.dot(mat_B.transpose(), np.dot(mat_C, mat_B))
            
            counter += 1
            
    def compute_factorized_mat_Me_mat_Ke(self):
        list_factorized_mat_C = self.get_material().get_factorized_mat_C()
        
        n_materials = self.get_material().get_n_factorized_mat_C()
        
        self.__list_factorized_mat_Ke = [np.zeros((tet4_n_dofs, tet4_n_dofs)) for _ in range(n_materials)]
        
        self.__factorized_mat_Me = np.zeros((tet4_n_dofs, tet4_n_dofs))
        
        self.__compute_jacobian()
        mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, tet4_mat_De_gauss)))
        
        counter = 0
        for g in get_gauss():
            gauss_weight = g[1]
            self.__factorized_mat_Me += gauss_weight * self.__det_J * tet4_mats_EeTEe_gauss[counter]
            
            for kk in range(n_materials):
                mat_C = list_factorized_mat_C[kk]
                self.__list_factorized_mat_Ke[kk] += gauss_weight * self.__det_J \
                                                     * np.dot(mat_B.transpose(), np.dot(mat_C, mat_B))
            
            counter += 1
                    
    def get_factorized_mat_Me(self):
        return self.__factorized_mat_Me
    
    def get_list_factorized_mat_Ke(self):
        return self.__list_factorized_mat_Ke
    
    def compute_list_factorized_mat_KTe(self, vec_Ue):
        list_factorized_mat_C = self.get_material().get_factorized_mat_C()
        
        n_materials = self.get_material().get_n_factorized_mat_C()
        
        self.__list_factorized_mat_Ktote = [np.zeros((tet4_n_dofs, tet4_n_dofs)) for _ in range(n_materials)]
        
        self.__list_factorized_mat_KT1e = [np.zeros((tet4_n_dofs, tet4_n_dofs)) for _ in range(n_materials)]
        self.__list_factorized_mat_KT2e = [np.zeros((tet4_n_dofs, tet4_n_dofs)) for _ in range(n_materials)]
        self.__list_factorized_mat_KT3e = [np.zeros((tet4_n_dofs, tet4_n_dofs)) for _ in range(n_materials)]
                
        self.__compute_jacobian()
        mat_grad = np.dot(self.__mat_invJJJ, np.dot(mat_P, tet4_mat_De_gauss))
        
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
        
        for g in get_gauss():
            gauss_weight = g[1]
            
            for kk in range(n_materials):
                mat_C = list_factorized_mat_C[kk]
                
                mat_KT1e_ii = gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(mat_C, mat_B))
                
                self.__list_factorized_mat_KT1e[kk] += mat_KT1e_ii
                
                self.__list_factorized_mat_KT2e[kk] += gauss_weight * self.__det_J \
                                                       * (np.dot(mat_B.transpose(), np.dot(mat_C, mat_H))
                                                          + np.dot(mat_H.transpose(), np.dot(mat_C, mat_B)))
                
                vec_piola2 = np.dot(mat_C, vec_E)
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
                
                self.__list_factorized_mat_KT3e[kk] += gauss_weight * self.__det_J \
                                                       * (np.dot(mat_grad.transpose(), np.dot(mat_pi, mat_grad))
                                                          + np.dot(mat_H.transpose(), np.dot(mat_C, mat_H)))
                    
                self.__list_factorized_mat_Ktote[kk] += mat_KT1e_ii
                self.__list_factorized_mat_Ktote[kk] += gauss_weight * self.__det_J \
                                                        * (0.5 * np.dot(mat_B.transpose(), np.dot(mat_C, mat_H))
                                                           + np.dot(mat_H.transpose(), np.dot(mat_C, mat_B)))
                self.__list_factorized_mat_Ktote[kk] += gauss_weight * self.__det_J \
                                                        * 0.5 * np.dot(mat_H.transpose(), np.dot(mat_C, mat_H))
                    
    def get_list_factorized_mat_Ktote(self):
        return self.__list_factorized_mat_Ktote
                
    def get_list_factorized_mat_KT1e(self):
        return self.__list_factorized_mat_KT1e
                
    def get_list_factorized_mat_KT2e(self):
        return self.__list_factorized_mat_KT2e
    
    def get_list_factorized_mat_KT3e(self):
        return self.__list_factorized_mat_KT3e

    def compute_element_stress_at_nodes(self, vec_Ue, return_strain=True):
        Ue_nodes = np.reshape(vec_Ue, (tet4_n_nodes, 3))

        old_nodes_coords = self.get_nodes_coords()
        new_nodes_coords = old_nodes_coords + Ue_nodes

        self.set_nodes_coords(new_nodes_coords)

        list_vec_stress_nodes = []
        if return_strain:
            list_vec_strain_nodes = []

        mat_invJJJ = self.__compute_invJJJ_at_node()
        mat_B = np.dot(mat_G, np.dot(mat_invJJJ, np.dot(mat_P, tet4_mat_De_gauss)))

        vec_strain = np.dot(mat_B, vec_Ue)
        if return_strain:
            for _ in range(tet4_n_gauss):
                list_vec_strain_nodes.append(vec_strain)

        vec_stress = np.dot(self.get_material().get_mat_C(), vec_strain)
        for _ in range(tet4_n_gauss):
            list_vec_stress_nodes.append(vec_stress)

        self.set_nodes_coords(old_nodes_coords)

        if return_strain:
            return list_vec_strain_nodes, list_vec_stress_nodes
        else:
            return list_vec_stress_nodes
        

            