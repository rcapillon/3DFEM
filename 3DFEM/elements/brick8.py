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

def brick8_shapefun_value(index, reference_coords):
    # N_i(x, y, z) = a + b*x + c*y + d*z + e*xy + f*xz + g*yz + h*x*y*z
    
    x = reference_coords[0]
    y = reference_coords[1]
    z = reference_coords[2]
    
    value = brick8_shapefun_coeffs[0, index]\
            + brick8_shapefun_coeffs[1, index] * x\
            + brick8_shapefun_coeffs[2, index] * y\
            + brick8_shapefun_coeffs[3, index] * z\
            + brick8_shapefun_coeffs[4, index] * x * y\
            + brick8_shapefun_coeffs[5, index] * x * z\
            + brick8_shapefun_coeffs[6, index] * y * z\
            + brick8_shapefun_coeffs[7, index] * x * y * z
    
    return value
    
def brick8_derivative_shapefun_value(index_shapefun, index_coord, reference_coords):
    # index_coord: 1 -> derivative with respect to x
    # index_coord: 2 -> derivative with respect to y
    # index_coord: 3 -> derivative with respect to z
    
    if index_coord == 1:
        y = reference_coords[1]
        z = reference_coords[2]
        
        X = np.array([[1],[y],[z],[y * z]])
        
        coeffs = brick8_shapefun_coeffs[[1, 4, 5, 7], index_shapefun]
        
    elif index_coord == 2:
        x = reference_coords[0]
        z = reference_coords[2]
        
        X = np.array([[1],[x],[z],[x * z]])
        
        coeffs = brick8_shapefun_coeffs[[2, 4, 6, 7], index_shapefun]
        
    elif index_coord == 3:
        x = reference_coords[0]
        y = reference_coords[1]
        
        X = np.array([[1],[x],[y],[x * y]])
        
        coeffs = brick8_shapefun_coeffs[[3, 5, 6, 7], index_shapefun]
            
    value = np.dot(coeffs, X)            
    
    return value
    
def brick8_compute_mat_Ee(reference_coords):
    mat_I = np.eye(3)
    mat_E0 = brick8_shapefun_value(0, reference_coords) * mat_I
    mat_E1 = brick8_shapefun_value(1, reference_coords) * mat_I
    mat_E2 = brick8_shapefun_value(2, reference_coords) * mat_I
    mat_E3 = brick8_shapefun_value(3, reference_coords) * mat_I
    mat_E4 = brick8_shapefun_value(4, reference_coords) * mat_I
    mat_E5 = brick8_shapefun_value(5, reference_coords) * mat_I
    mat_E6 = brick8_shapefun_value(6, reference_coords) * mat_I
    mat_E7 = brick8_shapefun_value(7, reference_coords) * mat_I
    
    brick8_mat_Ee = np.concatenate((mat_E0, mat_E1, mat_E2, mat_E3, mat_E4, mat_E5, mat_E6, mat_E7), axis=1)
    
    return brick8_mat_Ee

def brick8_compute_mat_De(reference_coords):
    mat_I = np.eye(3)
    
    mat_D0dx = brick8_derivative_shapefun_value(0, 1, reference_coords) * mat_I
    mat_D1dx = brick8_derivative_shapefun_value(1, 1, reference_coords) * mat_I
    mat_D2dx = brick8_derivative_shapefun_value(2, 1, reference_coords) * mat_I
    mat_D3dx = brick8_derivative_shapefun_value(3, 1, reference_coords) * mat_I
    mat_D4dx = brick8_derivative_shapefun_value(4, 1, reference_coords) * mat_I
    mat_D5dx = brick8_derivative_shapefun_value(5, 1, reference_coords) * mat_I
    mat_D6dx = brick8_derivative_shapefun_value(6, 1, reference_coords) * mat_I
    mat_D7dx = brick8_derivative_shapefun_value(7, 1, reference_coords) * mat_I
    
    mat_Ddx = np.concatenate((mat_D0dx, mat_D1dx, mat_D2dx, mat_D3dx, mat_D4dx, mat_D5dx, mat_D6dx, mat_D7dx), axis=1)
    
    mat_D0dy = brick8_derivative_shapefun_value(0, 2, reference_coords) * mat_I
    mat_D1dy = brick8_derivative_shapefun_value(1, 2, reference_coords) * mat_I
    mat_D2dy = brick8_derivative_shapefun_value(2, 2, reference_coords) * mat_I
    mat_D3dy = brick8_derivative_shapefun_value(3, 2, reference_coords) * mat_I
    mat_D4dy = brick8_derivative_shapefun_value(4, 2, reference_coords) * mat_I
    mat_D5dy = brick8_derivative_shapefun_value(5, 2, reference_coords) * mat_I
    mat_D6dy = brick8_derivative_shapefun_value(6, 2, reference_coords) * mat_I
    mat_D7dy = brick8_derivative_shapefun_value(7, 2, reference_coords) * mat_I
    
    mat_Ddy = np.concatenate((mat_D0dy, mat_D1dy, mat_D2dy, mat_D3dy, mat_D4dy, mat_D5dy, mat_D6dy, mat_D7dy), axis=1)
    
    mat_D0dz = brick8_derivative_shapefun_value(0, 3, reference_coords) * mat_I
    mat_D1dz = brick8_derivative_shapefun_value(1, 3, reference_coords) * mat_I
    mat_D2dz = brick8_derivative_shapefun_value(2, 3, reference_coords) * mat_I
    mat_D3dz = brick8_derivative_shapefun_value(3, 3, reference_coords) * mat_I
    mat_D4dz = brick8_derivative_shapefun_value(4, 3, reference_coords) * mat_I
    mat_D5dz = brick8_derivative_shapefun_value(5, 3, reference_coords) * mat_I
    mat_D6dz = brick8_derivative_shapefun_value(6, 3, reference_coords) * mat_I
    mat_D7dz = brick8_derivative_shapefun_value(7, 3, reference_coords) * mat_I
    
    mat_Ddz = np.concatenate((mat_D0dz, mat_D1dz, mat_D2dz, mat_D3dz, mat_D4dz, mat_D5dz, mat_D6dz, mat_D7dz), axis=1)
            
    brick8_mat_De = np.concatenate((mat_Ddx, mat_Ddy, mat_Ddz), axis=0) 
    
    return brick8_mat_De

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

# brick8 parameters and matrices

brick8_n_nodes = 4
brick8_n_dofs = brick8_n_nodes * 3
brick8_nodes_reference_coords =\
    np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]\
              [-1, -1, +1], [1, -1, +1], [1, 1, +1], [-1, 1, +1]])
brick8_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]

brick8_n_gauss = 8

a = 1.0 / np.sqrt(3)
brick8_gauss = [(np.array([+a, +a, +a]), 1),\
                (np.array([+a, +a, -a]), 1),\
                (np.array([+a, -a, +a]), 1),\
                (np.array([+a, -a, -a]), 1),\
                (np.array([-a, +a, +a]), 1),\
                (np.array([-a, +a, -a]), 1),\
                (np.array([-a, -a, +a]), 1),\
                (np.array([-a, -a, -a]), 1)]

x0 = brick8_nodes_reference_coords[0,0]
y0 = brick8_nodes_reference_coords[0,1]
z0 = brick8_nodes_reference_coords[0,2]
x1 = brick8_nodes_reference_coords[1,0]
y1 = brick8_nodes_reference_coords[1,1]
z1 = brick8_nodes_reference_coords[1,2]
x2 = brick8_nodes_reference_coords[2,0]
y2 = brick8_nodes_reference_coords[2,1]
z2 = brick8_nodes_reference_coords[2,2]
x3 = brick8_nodes_reference_coords[3,0]
y3 = brick8_nodes_reference_coords[3,1]
z3 = brick8_nodes_reference_coords[3,2]
x4 = brick8_nodes_reference_coords[4,0]
y4 = brick8_nodes_reference_coords[4,1]
z4 = brick8_nodes_reference_coords[4,2]
x5 = brick8_nodes_reference_coords[5,0]
y5 = brick8_nodes_reference_coords[5,1]
z5 = brick8_nodes_reference_coords[5,2]
x6 = brick8_nodes_reference_coords[6,0]
y6 = brick8_nodes_reference_coords[6,1]
z6 = brick8_nodes_reference_coords[6,2]
x7 = brick8_nodes_reference_coords[7,0]
y7 = brick8_nodes_reference_coords[7,1]
z7 = brick8_nodes_reference_coords[7,2]

brick8_mat_A = np.array([[1, x0, y0, z0, x0*y0, x0*z0, y0*z0, x0*y0*z0],\
                         [1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],\
                         [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2],\
                         [1, x3, y3, z3, x3*y3, x3*z3, y3*z3, x3*y3*z3],\
                         [1, x4, y4, z4, x4*y4, x4*z4, y4*z4, x4*y4*z4],\
                         [1, x5, y5, z5, x5*y5, x5*z5, y5*z5, x5*y5*z5],\
                         [1, x6, y6, z6, x6*y6, x6*z6, y6*z6, x6*y6*z6],\
                         [1, x7, y7, z7, x7*y7, x7*z7, y7*z7, x7*y7*z7]])
brick8_mat_I = np.eye(8)

brick8_shapefun_coeffs = np.linalg.solve(brick8_mat_A, brick8_mat_I)

brick8_mats_EeTEe_gauss = []
brick8_mat_De_gauss = []
for ii in range(brick8_n_gauss):
    gauss_point_ii = brick8_gauss[ii][0]
    
    mat_Ee_ii = brick8_compute_mat_Ee(gauss_point_ii)
    brick8_mats_EeTEe_gauss.append(np.dot(mat_Ee_ii.transpose(),mat_Ee_ii))
    
    mat_De_ii = brick8_compute_mat_De(gauss_point_ii)
    brick8_mat_De_gauss.append(mat_De_ii)

##############################################################################
# TO DO

class brick8(element.Element):
    def __init__(self, material, nodes_coords):
        super(brick8, self).__init__(material)
        
        self.__nodes_coords = nodes_coords
        self.__vec_nodes_coords = np.reshape(nodes_coords, brick8_n_dofs)
            
    def get_n_nodes(self):
        return brick8_n_nodes
    
    def get_n_dofs(self):
        return brick8_n_dofs
    
    def get_nodes_coords(self):
        return self.__nodes_coords
    
    def get_vec_nodes_coords(self):
        return self.__vec_nodes_coords
    
    def get_nodes_reference_coords(self):
        return brick8_nodes_reference_coords
    
    def get_faces(self):
        return brick8_faces
    
    def get_n_gauss(self):
        return brick8_n_gauss
    
    def get_gauss(self):
        return brick8_gauss
        
    def __compute_jacobian(self, index_gauss):
        mat_J1 = np.dot(brick8_mat_De_gauss[index_gauss][:3 , :], self.__vec_nodes_coords)
        mat_J2 = np.dot(brick8_mat_De_gauss[index_gauss][3:6, :], self.__vec_nodes_coords)
        mat_J3 = np.dot(brick8_mat_De_gauss[index_gauss][6: , :], self.__vec_nodes_coords)
        
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
        
    def get_mat_J(self):
        return self.__mat_J
    
    def get_det_J(self):
        return self.__det_J
    
    def get_mat_invJJJ(self):
        return self.__mat_invJJJ
    
    def compute_mat_Me(self):
        self.__mat_Me = np.zeros((brick8_n_dofs, brick8_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            self.__mat_Me += gauss_weight * self.get_material().get_rho() * self.__det_J * brick8_mats_EeTEe_gauss[counter] 
            counter += 1
            
    def compute_mat_Ke(self):
        self.__mat_Ke = np.zeros((brick8_n_dofs, brick8_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, brick8_mat_De_gauss[counter])))
            self.__mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(self.get_material().get_mat_C(), mat_B))
            counter += 1
    
    def compute_mat_Me_mat_Ke(self):
        self.__mat_Me = np.zeros((brick8_n_dofs, brick8_n_dofs))
        self.__mat_Ke = np.zeros((brick8_n_dofs, brick8_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, brick8_mat_De_gauss[counter])))
            self.__mat_Me += gauss_weight * self.get_material().get_rho() * self.__det_J * brick8_mats_EeTEe_gauss[counter]
            self.__mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(self.get_material().get_mat_C(), mat_B))
            counter += 1
    
    def get_mat_Me(self):
        return self.__mat_Me
    
    def get_mat_Ke(self):
        return self.__mat_Ke
    
    def compute_factorized_mat_Me(self):
        self.__factorized_mat_Me = np.zeros((brick8_n_dofs, brick8_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            self.__factorized_mat_Me += gauss_weight * self.__det_J * brick8_mats_EeTEe_gauss[counter] 
            counter += 1
            
    def compute_list_factorized_mat_Ke(self):
        list_factorized_mat_C = self.get_material().get_factorized_mat_C()
        
        self.__list_factorized_mat_Ke = []
        
        for mat_C in list_factorized_mat_C:
            factorized_mat_Ke = np.zeros((brick8_n_dofs, brick8_n_dofs))
            
            counter = 0
            for g in self.get_gauss():
                gauss_weight = g[1]
                self.__compute_jacobian(counter)
                mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, brick8_mat_De_gauss[counter])))
                factorized_mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(mat_C, mat_B))
                
            self.__list_factorized_mat_Ke.append(factorized_mat_Ke)
            
    def compute_factorized_mat_Me_mat_Ke(self):
        list_factorized_mat_C = self.get_material().get_factorized_mat_C()
        
        self.__list_factorized_mat_Ke = []
        
        factorized_mat_Me = np.zeros((brick8_n_dofs, brick8_n_dofs))
        factorized_mat_Ke = np.zeros((brick8_n_dofs, brick8_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            
            factorized_mat_Me += gauss_weight * self.__det_J * brick8_mats_EeTEe_gauss[counter]
            
            mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, brick8_mat_De_gauss[counter])))
            
            for mat_C in list_factorized_mat_C:
                factorized_mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(mat_C, mat_B))
            counter += 1
                
            self.__factorized_mat_Me = factorized_mat_Me
            self.__list_factorized_mat_Ke.append(factorized_mat_Ke)
    
    def get_factorized_mat_Me(self):
        return self.__factorized_mat_Me
    
    def get_list_factorized_mat_Ke(self):
        return self.__list_factorized_mat_Ke