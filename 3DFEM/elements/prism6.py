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

def prism6_shapefun_value(index, reference_coords):
    # N_i(x, y, z) = a + b*x + c*y + d*z + e*xy + f*xz + g*yz + h*x*y*z
    
    x = reference_coords[0]
    y = reference_coords[1]
    z = reference_coords[2]
    
    value = prism6_shapefun_coeffs[0, index]\
            + prism6_shapefun_coeffs[1, index] * x\
            + prism6_shapefun_coeffs[2, index] * y\
            + prism6_shapefun_coeffs[3, index] * z\
            + prism6_shapefun_coeffs[4, index] * x * z\
            + prism6_shapefun_coeffs[5, index] * y * z\
    
    return value
    
def prism6_derivative_shapefun_value(index_shapefun, index_coord, reference_coords):
    # index_coord: 1 -> derivative with respect to x
    # index_coord: 2 -> derivative with respect to y
    # index_coord: 3 -> derivative with respect to z
    
    if index_coord == 1:
        y = reference_coords[1]
        z = reference_coords[2]
        
        X = np.array([[1],[z]])
        
        coeffs = prism6_shapefun_coeffs[[1, 4], index_shapefun]
        
    elif index_coord == 2:
        x = reference_coords[0]
        z = reference_coords[2]
        
        X = np.array([[1],[z]])
        
        coeffs = prism6_shapefun_coeffs[[2, 5], index_shapefun]
        
    elif index_coord == 3:
        x = reference_coords[0]
        y = reference_coords[1]
        
        X = np.array([[1],[x],[y]])
        
        coeffs = prism6_shapefun_coeffs[[3, 4, 5], index_shapefun]
            
    value = np.dot(coeffs, X)            
    
    return value
    
def prism6_compute_mat_Ee(reference_coords):
    mat_I = np.eye(3)
    mat_E0 = prism6_shapefun_value(0, reference_coords) * mat_I
    mat_E1 = prism6_shapefun_value(1, reference_coords) * mat_I
    mat_E2 = prism6_shapefun_value(2, reference_coords) * mat_I
    mat_E3 = prism6_shapefun_value(3, reference_coords) * mat_I
    mat_E4 = prism6_shapefun_value(4, reference_coords) * mat_I
    mat_E5 = prism6_shapefun_value(5, reference_coords) * mat_I
    
    prism6_mat_Ee = np.concatenate((mat_E0, mat_E1, mat_E2, mat_E3, mat_E4, mat_E5), axis=1)
    
    return prism6_mat_Ee

def prism6_compute_mat_De(reference_coords):
    mat_I = np.eye(3)
    
    mat_D0dx = prism6_derivative_shapefun_value(0, 1, reference_coords) * mat_I
    mat_D1dx = prism6_derivative_shapefun_value(1, 1, reference_coords) * mat_I
    mat_D2dx = prism6_derivative_shapefun_value(2, 1, reference_coords) * mat_I
    mat_D3dx = prism6_derivative_shapefun_value(3, 1, reference_coords) * mat_I
    mat_D4dx = prism6_derivative_shapefun_value(4, 1, reference_coords) * mat_I
    mat_D5dx = prism6_derivative_shapefun_value(5, 1, reference_coords) * mat_I
    
    mat_Ddx = np.concatenate((mat_D0dx, mat_D1dx, mat_D2dx, mat_D3dx, mat_D4dx, mat_D5dx), axis=1)
    
    mat_D0dy = prism6_derivative_shapefun_value(0, 2, reference_coords) * mat_I
    mat_D1dy = prism6_derivative_shapefun_value(1, 2, reference_coords) * mat_I
    mat_D2dy = prism6_derivative_shapefun_value(2, 2, reference_coords) * mat_I
    mat_D3dy = prism6_derivative_shapefun_value(3, 2, reference_coords) * mat_I
    mat_D4dy = prism6_derivative_shapefun_value(4, 2, reference_coords) * mat_I
    mat_D5dy = prism6_derivative_shapefun_value(5, 2, reference_coords) * mat_I
    
    mat_Ddy = np.concatenate((mat_D0dy, mat_D1dy, mat_D2dy, mat_D3dy, mat_D4dy, mat_D5dy), axis=1)
    
    mat_D0dz = prism6_derivative_shapefun_value(0, 3, reference_coords) * mat_I
    mat_D1dz = prism6_derivative_shapefun_value(1, 3, reference_coords) * mat_I
    mat_D2dz = prism6_derivative_shapefun_value(2, 3, reference_coords) * mat_I
    mat_D3dz = prism6_derivative_shapefun_value(3, 3, reference_coords) * mat_I
    mat_D4dz = prism6_derivative_shapefun_value(4, 3, reference_coords) * mat_I
    mat_D5dz = prism6_derivative_shapefun_value(5, 3, reference_coords) * mat_I
    
    mat_Ddz = np.concatenate((mat_D0dz, mat_D1dz, mat_D2dz, mat_D3dz, mat_D4dz, mat_D5dz), axis=1)
            
    prism6_mat_De = np.concatenate((mat_Ddx, mat_Ddy, mat_Ddz), axis=0) 
    
    return prism6_mat_De

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

# prism6 parameters and matrices

prism6_n_nodes = 4
prism6_n_dofs = prism6_n_nodes * 3
prism6_nodes_reference_coords =\
    np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1]\
              [0, 0, +1], [1, 0, +1], [0, 1, +1]])
prism6_faces = [[0, 1, 2], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]]

prism6_n_gauss = 6

a = 0.5773502692
b = 1.0 / 6.0
prism6_gauss = [(np.array([0.5, 0.0, -a]), b),\
                (np.array([0.0, 0.5, -a]), b),\
                (np.array([0.5, 0.5, -a]), b),\
                (np.array([0.5, 0.0, +a]), b),\
                (np.array([0.0, 0.5, +a]), b),\
                (np.array([0.5, 0.5, +a]), b)]

x0 = prism6_nodes_reference_coords[0,0]
y0 = prism6_nodes_reference_coords[0,1]
z0 = prism6_nodes_reference_coords[0,2]
x1 = prism6_nodes_reference_coords[1,0]
y1 = prism6_nodes_reference_coords[1,1]
z1 = prism6_nodes_reference_coords[1,2]
x2 = prism6_nodes_reference_coords[2,0]
y2 = prism6_nodes_reference_coords[2,1]
z2 = prism6_nodes_reference_coords[2,2]
x3 = prism6_nodes_reference_coords[3,0]
y3 = prism6_nodes_reference_coords[3,1]
z3 = prism6_nodes_reference_coords[3,2]
x4 = prism6_nodes_reference_coords[4,0]
y4 = prism6_nodes_reference_coords[4,1]
z4 = prism6_nodes_reference_coords[4,2]
x5 = prism6_nodes_reference_coords[5,0]
y5 = prism6_nodes_reference_coords[5,1]
z5 = prism6_nodes_reference_coords[5,2]

prism6_mat_A = np.array([[1, x0, y0, z0, x0*z0, y0*z0],\
                         [1, x1, y1, z1, x1*z1, y1*z1],\
                         [1, x2, y2, z2, x2*z2, y2*z2],\
                         [1, x3, y3, z3, x3*z3, y3*z3],\
                         [1, x4, y4, z4, x4*z4, y4*z4],\
                         [1, x5, y5, z5, x5*z5, y5*z5]])

prism6_mat_I = np.eye(6)

prism6_shapefun_coeffs = np.linalg.solve(prism6_mat_A, prism6_mat_I)

prism6_mats_EeTEe_gauss = []
prism6_mat_De_gauss = []
for ii in range(prism6_n_gauss):
    gauss_point_ii = prism6_gauss[ii][0]
    
    mat_Ee_ii = prism6_compute_mat_Ee(gauss_point_ii)
    prism6_mats_EeTEe_gauss.append(np.dot(mat_Ee_ii.transpose(),mat_Ee_ii))
    
    mat_De_ii = prism6_compute_mat_De(gauss_point_ii)
    prism6_mat_De_gauss.append(mat_De_ii)

##############################################################################
# TO DO

class prism6(element.Element):
    def __init__(self, rho, Y, nu, nodes_coords):
        super(prism6, self).__init__(rho, Y, nu)
        
        self.__nodes_coords = nodes_coords
        self.__vec_nodes_coords = np.reshape(nodes_coords, prism6_n_dofs)
            
    def get_n_nodes(self):
        return prism6_n_nodes
    
    def get_n_dofs(self):
        return prism6_n_dofs
    
    def get_nodes_coords(self):
        return self.__nodes_coords
    
    def get_vec_nodes_coords(self):
        return self.__vec_nodes_coords
    
    def get_nodes_reference_coords(self):
        return prism6_nodes_reference_coords
    
    def get_faces(self):
        return prism6_faces
    
    def get_n_gauss(self):
        return prism6_n_gauss
    
    def get_gauss(self):
        return prism6_gauss
        
    def __compute_jacobian(self, index_gauss):
        mat_J1 = np.dot(prism6_mat_De_gauss[index_gauss][:3 , :], self.__vec_nodes_coords)
        mat_J2 = np.dot(prism6_mat_De_gauss[index_gauss][3:6, :], self.__vec_nodes_coords)
        mat_J3 = np.dot(prism6_mat_De_gauss[index_gauss][6: , :], self.__vec_nodes_coords)
        
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
        self.__mat_Me = np.zeros((prism6_n_dofs, prism6_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            self.__mat_Me += gauss_weight * self.get_rho() * self.__det_J * prism6_mats_EeTEe_gauss[counter] 
            counter += 1
            
    def compute_mat_Ke(self):
        self.__mat_Ke = np.zeros((prism6_n_dofs, prism6_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, prism6_mat_De_gauss[counter])))
            self.__mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(self.get_mat_C(), mat_B))
            counter += 1
    
    def compute_mat_Me_mat_Ke(self):
        self.__mat_Me = np.zeros((prism6_n_dofs, prism6_n_dofs))
        self.__mat_Ke = np.zeros((prism6_n_dofs, prism6_n_dofs))
        
        counter = 0
        for g in self.get_gauss():
            gauss_weight = g[1]
            self.__compute_jacobian(counter)
            mat_B = np.dot(mat_G, np.dot(self.__mat_invJJJ, np.dot(mat_P, prism6_mat_De_gauss[counter])))
            self.__mat_Me += gauss_weight * self.get_rho() * self.__det_J * prism6_mats_EeTEe_gauss[counter]
            self.__mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(self.get_mat_C(), mat_B))
            counter += 1
    
    def get_mat_Me(self):
        return self.__mat_Me
    
    def get_mat_Ke(self):
        return self.__mat_Ke