import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh, spsolve
import scipy.spatial

class IsotropicElasticMaterial:
    def __init__(self, rho, Y, nu):
        self.set_properties(rho, Y, nu)
        
    def set_properties(self, rho, Y, nu):
        # rho: mass density (kg/m^3)
        # Y: Young's modulus (Pa)
        # nu: Poisson coefficient
        
        self.__rho = rho
        self.__Y = Y
        self.__nu = nu
        
        self.__lame1 = (self.__Y * self.__nu) / ((1 + self.__nu) * (1 - 2*self.__nu))
        self.__lame2 = self.__Y / (2 * (1 + self.__nu))
        
        repmat_lame1 = np.tile(self.__lame1, (3, 3))
        self.__mat_C = 2 * self.__lame2 * np.eye(6)
        self.__mat_C[:3,:3] += repmat_lame1
        
    def get_rho(self):
        return self.__rho
    
    def get_Y(self):
        return self.__Y
    
    def get_nu(self):
        return self.__nu
        
    def get_mat_C(self):
        return self.__mat_C

class Element(IsotropicElasticMaterial):
    def __init__(self, rho, Y, nu):
        super(Element, self).__init__(rho, Y, nu)
        
        s = np.sqrt(2)/2
        self.__mat_G = np.zeros((6, 9))
        self.__mat_G[0,0] = 1
        self.__mat_G[1,4] = 1
        self.__mat_G[2,8] = 1
        self.__mat_G[3,2] = s
        self.__mat_G[3,6] = s
        self.__mat_G[4,1] = s
        self.__mat_G[4,3] = s
        self.__mat_G[5,5] = s
        self.__mat_G[5,7] = s
        
        self.__mat_P = np.zeros((9, 9))
        self.__mat_P[0,0] = 1
        self.__mat_P[1,3] = 1
        self.__mat_P[2,6] = 1
        self.__mat_P[3,1] = 1
        self.__mat_P[4,4] = 1
        self.__mat_P[5,7] = 1
        self.__mat_P[6,2] = 1
        self.__mat_P[7,5] = 1
        self.__mat_P[8,8] = 1
        
    def get_mat_G(self):
        return self.__mat_G
    
    def get_mat_P(self):
        return self.__mat_P
    
    def set_element_number(self, num):
        self.__num = num
        
    def get_element_number(self):
        return self.__num
    
    def set_nodes_dofs(self, nodes_nums):
        self.__nodes_nums = nodes_nums
        self.__dofs_nums = np.array([], dtype=np.int32)
        for node in self.__nodes_nums:
            self.__dofs_nums = np.append(self.__dofs_nums,[node * 3, node * 3 + 1, node * 3 + 2])
        
    def get_nodes_nums(self):
        return self.__nodes_nums
    
    def get_dofs_nums(self):
        return self.__dofs_nums

class Tet4(Element):
    def __init__(self, rho, Y, nu, nodes_coords):
        super(Tet4, self).__init__(rho, Y, nu)
        self.__n_nodes = 4
        self.__n_dofs = self.__n_nodes * 3
        self.__nodes_coords = nodes_coords
        self.__vec_nodes_coords = np.reshape(self.__nodes_coords, self.__n_nodes * 3)
        self.__nodes_reference_coords =\
            np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        self.__faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        self.__n_gauss = 4
        self.__gauss = [(np.array([0.5854101966, 0.1381966011, 0.1381966011]), 0.0416666667),\
                      (np.array([0.1381966011, 0.5854101966, 0.1381966011]), 0.0416666667),\
                      (np.array([0.1381966011, 0.1381966011, 0.5854101966]), 0.0416666667),\
                      (np.array([0.1381966011, 0.1381966011, 0.1381966011]), 0.0416666667)]
        self.__compute_shapefun_coeffs()
            
    def __compute_shapefun_coeffs(self):
        mat_A = np.array([[1, self.__nodes_reference_coords[0,0], self.__nodes_reference_coords[0,1], self.__nodes_reference_coords[0,2]],\
                          [1, self.__nodes_reference_coords[1,0], self.__nodes_reference_coords[1,1], self.__nodes_reference_coords[1,2]],\
                          [1, self.__nodes_reference_coords[2,0], self.__nodes_reference_coords[2,1], self.__nodes_reference_coords[2,2]],\
                          [1, self.__nodes_reference_coords[3,0], self.__nodes_reference_coords[3,1], self.__nodes_reference_coords[3,2]]])
        mat_I = np.eye(4)
        
        self.__shapefun_coeffs = np.linalg.solve(mat_A, mat_I)
            
    def get_n_nodes(self):
        return self.__n_nodes
    
    def get_n_dofs(self):
        return self.__n_dofs
    
    def get_nodes_coords(self):
        return self.__nodes_coords
    
    def get_vec_nodes_coords(self):
        return self.__vec_nodes_coords
    
    def get_nodes_reference_coords(self):
        return self.__nodes_reference_coords
    
    def get_faces(self):
        return self.__faces
    
    def get_n_gauss(self):
        return self.__n_gauss
    
    def get_gauss(self):
        return self.__gauss
    
    def __shapefun_value(self, index, reference_coords):
        # N_i(x, y, z) = a + b*x + c*y + d*z
        
        x = reference_coords[0]
        y = reference_coords[1]
        z = reference_coords[2]
        
        value = self.__shapefun_coeffs[0, index]\
                + self.__shapefun_coeffs[1, index] * x\
                + self.__shapefun_coeffs[2, index] * y\
                + self.__shapefun_coeffs[3, index] * z
        
        return value
    
    def __derivative_shapefun_value(self, index_shapefun, index_coord, reference_coords):
        # dNdx_i(x, y, z) = b
        # dNdy_i(x, y, z) = c
        # dNdz_i(x, y, z) = d
        
        # index_coord: 1 -> derivative with respect to x
        # index_coord: 2 -> derivative with respect to y
        # index_coord: 3 -> derivative with respect to z
        
        value = self.__shapefun_coeffs[index_coord, index_shapefun]
        
        return value
        
    def __compute_mat_Ee(self, reference_coords):
        mat_I = np.eye(3)
        mat_E0 = self.__shapefun_value(0, reference_coords) * mat_I
        mat_E1 = self.__shapefun_value(1, reference_coords) * mat_I
        mat_E2 = self.__shapefun_value(2, reference_coords) * mat_I
        mat_E3 = self.__shapefun_value(3, reference_coords) * mat_I
        
        self.__mat_Ee = np.concatenate((mat_E0, mat_E1, mat_E2, mat_E3), axis=1)
        
    def get_mat_Ee(self):
        return self.__mat_Ee
    
    def __compute_mat_De(self, reference_coords):
        mat_I = np.eye(3)
        
        mat_D0dx = self.__derivative_shapefun_value(0, 1, reference_coords) * mat_I
        mat_D1dx = self.__derivative_shapefun_value(1, 1, reference_coords) * mat_I
        mat_D2dx = self.__derivative_shapefun_value(2, 1, reference_coords) * mat_I
        mat_D3dx = self.__derivative_shapefun_value(3, 1, reference_coords) * mat_I
        
        mat_Ddx = np.concatenate((mat_D0dx, mat_D1dx, mat_D2dx, mat_D3dx), axis=1)
        
        mat_D0dy = self.__derivative_shapefun_value(0, 2, reference_coords) * mat_I
        mat_D1dy = self.__derivative_shapefun_value(1, 2, reference_coords) * mat_I
        mat_D2dy = self.__derivative_shapefun_value(2, 2, reference_coords) * mat_I
        mat_D3dy = self.__derivative_shapefun_value(3, 2, reference_coords) * mat_I
        
        mat_Ddy = np.concatenate((mat_D0dy, mat_D1dy, mat_D2dy, mat_D3dy), axis=1)
        
        mat_D0dz = self.__derivative_shapefun_value(0, 3, reference_coords) * mat_I
        mat_D1dz = self.__derivative_shapefun_value(1, 3, reference_coords) * mat_I
        mat_D2dz = self.__derivative_shapefun_value(2, 3, reference_coords) * mat_I
        mat_D3dz = self.__derivative_shapefun_value(3, 3, reference_coords) * mat_I
        
        mat_Ddz = np.concatenate((mat_D0dz, mat_D1dz, mat_D2dz, mat_D3dz), axis=1)
                
        self.__mat_De = np.concatenate((mat_Ddx, mat_Ddy, mat_Ddz), axis=0)    
    
    def get_mat_De(self):
        return self.__mat_De
        
    def __compute_jacobian(self, reference_coords):
        self.__compute_mat_De(reference_coords)
        mat_J1 = np.dot(self.__mat_De[:3 , :], self.__vec_nodes_coords)
        mat_J2 = np.dot(self.__mat_De[3:6, :], self.__vec_nodes_coords)
        mat_J3 = np.dot(self.__mat_De[6: , :], self.__vec_nodes_coords)
        
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
        self.__mat_Me = np.zeros((self.__n_dofs, self.__n_dofs))
                
        for g in self.get_gauss():
            gauss_coords = g[0]
            gauss_weight = g[1]
            
            self.__compute_jacobian(gauss_coords)
            self.__compute_mat_Ee(gauss_coords)
            
            self.__mat_Me += gauss_weight * self.get_rho() * self.__det_J * np.dot(self.__mat_Ee.transpose(), self.__mat_Ee) 
            
    def compute_mat_Ke(self):
        self.__mat_Ke = np.zeros((self.__n_dofs, self.__n_dofs))
        
        for g in self.get_gauss():
            gauss_coords = g[0]
            gauss_weight = g[1]
            
            self.__compute_jacobian(gauss_coords)
            
            mat_B = np.dot(self.get_mat_G(), np.dot(self.__mat_invJJJ, np.dot(self.get_mat_P(), self.__mat_De)))
            
            self.__mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(self.get_mat_C(), mat_B))
    
    def compute_mat_Me_mat_Ke(self):
        self.__mat_Me = np.zeros((self.__n_dofs, self.__n_dofs))
        self.__mat_Ke = np.zeros((self.__n_dofs, self.__n_dofs))
        
        for g in self.get_gauss():
            gauss_coords = g[0]
            gauss_weight = g[1]
            
            self.__compute_jacobian(gauss_coords)
            self.__compute_mat_Ee(gauss_coords)
            
            mat_B = np.dot(self.get_mat_G(), np.dot(self.__mat_invJJJ, np.dot(self.get_mat_P(), self.__mat_De)))
            
            self.__mat_Me += gauss_weight * self.get_rho() * self.__det_J * np.dot(self.__mat_Ee.transpose(), self.__mat_Ee)
            self.__mat_Ke += gauss_weight * self.__det_J * np.dot(mat_B.transpose(), np.dot(self.get_mat_C(), mat_B))
    
    def get_mat_Me(self):
        return self.__mat_Me
    
    def get_mat_Ke(self):
        return self.__mat_Ke
            
class Tet10(Element):
    def __init__(self):
        super(Tet10, self).__init__()
    
class Prism6(Element):
    def __init__(self):
        super(Prism6, self).__init__()
    
class Prism18(Element):
    def __init__(self):
        super(Prism18, self).__init__()
    
class Brick8(Element):
    def __init__(self):
        super(Brick8, self).__init__()
    
class Brick27(Element):
    def __init__(self):
        super(Brick27, self).__init__()
        
class Mesh:
    def __init__(self, points):
        self.__n_points = points.shape[0]
        self.__n_total_dofs = self.__n_points * 3
        self.__points = points
        self.__ls_dofs_dir = [] # zero-dirichlet condition
        self.__ls_dofs_free = [ii for ii in range(self.__n_total_dofs)]
    
    def set_elements_list(self, elem_list):
        self.__n_elements = len(elem_list)
        self.__elements_table = elem_list
        
    def delaunay3D_from_points(self, rho, Y, nu):
        tri = scipy.spatial.Delaunay(self.__points)
        self.__points = tri.points
        self.__n_elements = tri.simplices.shape[0]
                
        self.__elements_table = []
        
        elem_counter = 0
        for ii in range(self.__n_elements):
            nodes_ii = tri.simplices[ii, :]
            nodes_coords = self.__points[nodes_ii, :]
            
            P1 = nodes_coords[0,:]
            P2 = nodes_coords[1,:]
            P3 = nodes_coords[2,:]
            P4 = nodes_coords[3,:]
            
            mixed_product = np.dot(np.cross((P2-P1), (P3-P1)), (P4-P1))
            
            if mixed_product < 0:
                nodes_ii[[0, 1]] = nodes_ii[[1, 0]]
                nodes_coords = self.__points[nodes_ii,:]
                
            if mixed_product != 0:
                element_ii = Tet4(rho, Y, nu, nodes_coords)
                element_ii.set_element_number(elem_counter)
                element_ii.set_nodes_dofs(nodes_ii)
                self.__elements_table.append(element_ii)
                
    def add_UL_to_points(self, UL):    
        U = np.zeros((self.__n_total_dofs,))
        U[self.__ls_dofs_free] = UL
        
        self.__points[:,0] += U[::3]
        self.__points[:,1] += U[1::3]
        self.__points[:,2] += U[2::3]
        
    def add_U_to_points(self, U):    
        self.__points[:,0] += U[::3]
        self.__points[:,1] += U[1::3]
        self.__points[:,2] += U[2::3]
                
    def get_n_points(self):
        return self.__n_points
    
    def get_n_total_dofs(self):
        return self.__n_total_dofs
    
    def get_points(self):
        return self.__points
    
    def get_n_elements(self):
        return self.__n_elements
    
    def get_elements_table(self):
        return self.__elements_table
    
    def set_dirichlet(self, ls_dofs_dir):
        self.__ls_dofs_dir = ls_dofs_dir
        self.__ls_dofs_free = list(set(range(self.__n_total_dofs)) - set(ls_dofs_dir))
        
    def get_dirichlet_dofs(self):
        return self.__ls_dofs_dir
    
    def get_free_dofs(self):
        return self.__ls_dofs_free
    
class Force:
    def __init__(self, mesh):
        self.__mesh = mesh
        self.__ls_nodal_forces_t0 = []
        self.__vec_variation = np.array([0])
        
    def get_mesh(self):
        return self.__mesh
    
    def add_nodal_forces_t0(self, ls_nodes, nodal_force_vector):
        self.__ls_nodal_forces_t0.append((ls_nodes, nodal_force_vector))
        
    def get_nodal_forces_t0(self):
        return self.__ls_nodal_forces_t0
    
    def compute_F0(self):
        n_total_dofs = self.__mesh.get_n_total_dofs()
        
        self.__vec_F0 = np.zeros((n_total_dofs,))
        
        for group in self.get_nodal_forces_t0():
            nodes = group[0]
            force_vector = group[1]
            
            ls_dofs_x = [node * 3 for node in nodes]
            ls_dofs_y = [node * 3 + 1 for node in nodes]
            ls_dofs_z = [node * 3 + 2 for node in nodes]
            
            self.__vec_F0[ls_dofs_x] += np.repeat(force_vector[0], len(ls_dofs_x))
            self.__vec_F0[ls_dofs_y] += np.repeat(force_vector[1], len(ls_dofs_y))
            self.__vec_F0[ls_dofs_z] += np.repeat(force_vector[2], len(ls_dofs_z))
            
    def compute_constant_F(self, n_timesteps):        
        self.__mat_F = np.tile(self.__vec_F0, n_timesteps)
        
    def set_F_variation(self, vec_variation):
        self.__vec_variation = vec_variation
        
    def compute_varying_F(self):
        n_timesteps = len(self.__vec_variation)
        
        self.__mat_F = np.zeros((self.__vec_F0.shape[0], n_timesteps))
        
        for ii in range(n_timesteps):
            vec_F_ii = self.__vec_F0 * self.__vec_variation[ii]
            self.__mat_F[:, ii] = vec_F_ii
            
    def apply_dirichlet_F0(self):
        self.__vec_F0L = self.__vec_F0[self.__mesh.get_free_dofs()]
        self.__vec_F0D = self.__vec_F0[self.__mesh.get_dirichlet_dofs()]
        
    def get_F0L(self):
        return self.__vec_F0L
    
    def get_F0D(self):
        return self.__vec_F0D
    
    def apply_dirichlet_F(self):
        self.__mat_FL = self.__mat_F[self.__mesh.get_free_dofs(), :]
        self.__mat_FD = self.__mat_F[self.__mesh.get_dirichlet_dofs(), :]
        
    def get_FL(self):
        return self.__mat_FL
    
    def get_FD(self):
        return self.__mat_FD
    
class Structure:
    def __init__(self, mesh):
        self.__mesh = mesh
        self.__n_total_dofs = self.__mesh.get_n_total_dofs()
        self.__n_free_dofs = len(self.__mesh.get_free_dofs())
        self.__n_dir_dofs = len(self.__mesh.get_dirichlet_dofs())
        self.__alphaM = 0
        self.__alphaK = 0
        
    def get_mesh(self):
        return self.__mesh
    
    def compute_M(self, symmetrization=False):    
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        
        for element in self.get_mesh().get_elements_table():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Me()
            
            vec_dataM = np.append(vec_dataM, element.get_mat_Me().flatten(order='F'))
            
        vec_rows = np.array(vec_rows)
        vec_cols = np.array(vec_cols)
        
        self.__mat_M = scipy.sparse.csr_matrix((vec_dataM, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
        
    def compute_K(self, symmetrization=False):    
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataK = np.array([])
        
        for element in self.get_mesh().get_elements_table():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Ke()
            
            vec_dataK = np.append(vec_dataK, element.get_mat_Ke().flatten(order='F'))
            
        vec_rows = np.array(vec_rows)
        vec_cols = np.array(vec_cols)
        
        self.__mat_K = scipy.sparse.csr_matrix((vec_dataK, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
                
    def compute_M_K(self, symmetrization=False):
        vec_rows = np.array([])
        vec_cols = np.array([])
        vec_dataM = np.array([])
        vec_dataK = np.array([])
        
        for element in self.get_mesh().get_elements_table():
            n_dofs = element.get_n_dofs()
            
            ind_I = list(range(n_dofs)) * n_dofs
            ind_J = []
            for ii in range(n_dofs):
                ind_J.extend([ii] * n_dofs)
                                
            vec_rows = np.append(vec_rows, element.get_dofs_nums()[ind_I])
            vec_cols = np.append(vec_cols, element.get_dofs_nums()[ind_J])
            
            element.compute_mat_Me_mat_Ke()
            
            vec_dataM = np.append(vec_dataM, element.get_mat_Me().flatten(order='F'))
            vec_dataK = np.append(vec_dataK, element.get_mat_Ke().flatten(order='F'))
            
        self.__mat_M = scipy.sparse.csr_matrix((vec_dataM, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_M = 0.5 * (self.__mat_M + self.__mat_M.transpose())
        
        self.__mat_K = scipy.sparse.csr_matrix((vec_dataK, (vec_rows, vec_cols)), shape=(self.__n_total_dofs, self.__n_total_dofs))
        
        if symmetrization == True:
            self.__mat_K = 0.5 * (self.__mat_K + self.__mat_K.transpose())
        
    def set_rayleigh(self, alphaM, alphaK):
        self.__alphaM = alphaM
        self.__alphaK = alphaK
        
    def compute_D(self):
        self.__mat_D = self.__alphaM * self.__mat_M + self.__alphaK * self.__mat_K
        
    def get_M(self):
        return self.__mat_M
    
    def get_K(self):
        return self.__mat_K
    
    def get_D(self):
        return self.__mat_D
    
    def apply_dirichlet_M(self):
        self.__mat_MLL = self.__mat_M[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_free_dofs()]
        self.__mat_MLD = self.__mat_M[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_dirichlet_dofs()]
        
    def apply_dirichlet_K(self):
        self.__mat_KLL = self.__mat_K[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_free_dofs()]
        self.__mat_KLD = self.__mat_K[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_dirichlet_dofs()]
        
    def apply_dirichlet_D(self):
        self.__mat_DLL = self.__mat_D[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_free_dofs()]
        self.__mat_DLD = self.__mat_D[self.__mesh.get_free_dofs(), :][:, self.__mesh.get_dirichlet_dofs()]
                
    def get_MLL(self):
        return self.__mat_MLL
    
    def get_MLD(self):
        return self.__mat_MLD
    
    def get_KLL(self):
        return self.__mat_KLL
    
    def get_KLD(self):
        return self.__mat_KLD
    
    def get_DLL(self):
        return self.__mat_DLL
    
    def get_DLD(self):
        return self.__mat_DLD
    
    def set_U0L(self, vec_U0L=None):
        if vec_U0L == None:
            self.__vec_U0L = np.zeros((self.__n_free_dofs,))
        else:
            self.__vec_U0L = vec_U0L
        
    def get_U0L(self):
        return self.__vec_U0L
    
    def set_V0L(self, vec_V0L=None):
        if vec_V0L == None:
            self.__vec_V0L = np.zeros((self.__n_free_dofs,))
        else:
            self.__vec_V0L = vec_V0L
        
    def get_V0L(self):
        return self.__vec_V0L
    
    def set_A0L(self, vec_A0L=None):
        if vec_A0L == None:
            self.__vec_A0L = np.zeros((self.__n_free_dofs,))
        else:
            self.__vec_A0L = vec_A0L
        
    def get_A0L(self):
        return self.__vec_A0L
            
class Solver:
    def __init__(self, structure, force=None):
        self.__structure = structure
        self.__force = force
        self.__n_total_dofs = self.__structure.get_mesh().get_n_total_dofs()
        
    def get_structure(self):
        return self.__structure
    
    def get_force(self):
        return self.__force

    def modal_solver(self, n_modes):
        # Eigenvectors are mass-normalized
        
        self.__structure.compute_M_K()
        self.__structure.apply_dirichlet_M()
        self.__structure.apply_dirichlet_K()
        
        (eigvals, eigvects) = eigsh(self.__structure.get_KLL(), n_modes, self.__structure.get_MLL(), which='SM')
        sort_indices = np.argsort(eigvals)
        eigvals = eigvals[sort_indices]
        eigvects = eigvects[:, sort_indices]
        
        self.__eigenfreqs = np.sqrt(eigvals) / (2 * np.pi)
        self.__modesL = eigvects
        self.__modes = np.zeros((self.__n_total_dofs, n_modes))
        self.__modes[self.__structure.get_mesh().get_free_dofs(), :] = eigvects
        
    def get_eigenfreqs(self):
        return self.__eigenfreqs
    
    def get_modesL(self):
        return self.__modesL
    
    def get_modes(self):
        return self.__modes
            
    def linear_static_solver(self):
        self.__structure.compute_K()
        self.__structure.apply_dirichlet_K()
        
        self.__force.compute_F0()
        self.__force.apply_dirichlet_F0()
        
        vec_UL = spsolve(self.__structure.get_KLL(), self.__force.get_F0L())
                
        self.__vec_U = np.zeros((self.__n_total_dofs,))
        self.__vec_U[self.__structure.get_mesh().get_free_dofs()] = vec_UL
        
    def get_vec_U(self):
        return self.__vec_U
    
    def linear_newmark_solver(self, beta1, beta2, t0, tmax, n_timesteps, n_modes, verbose=True):
        dt = (tmax - t0) / (n_timesteps - 1)
        
        print("Computing reduced-order model...")
        
        self.modal_solver(n_modes)
        
        self.__structure.compute_D()
        self.__structure.apply_dirichlet_D()
        
        self.__force.compute_F0()
        self.__force.compute_varying_F()
        self.__force.apply_dirichlet_F()
        
        Mrom = np.dot(self.__modesL.transpose(), self.__structure.get_MLL().dot(self.__modesL))
        Krom = np.dot(self.__modesL.transpose(), self.__structure.get_KLL().dot(self.__modesL))
        Drom = np.dot(self.__modesL.transpose(), self.__structure.get_DLL().dot(self.__modesL))
        
        From = np.dot(self.__modesL.transpose(), self.__force.get_FL())
        
        print("Applying initial conditions...")
        
        qU0 = np.dot(self.__modesL.transpose(), self.__structure.get_U0L())
        qV0 = np.dot(self.__modesL.transpose(), self.__structure.get_V0L())
        qA0 = np.dot(self.__modesL.transpose(), self.__structure.get_A0L())
        
        qU = np.zeros((n_modes, n_timesteps + 1))
        qV = np.zeros((n_modes, n_timesteps + 1))
        qA = np.zeros((n_modes, n_timesteps + 1))
        
        qU[:, 0] = qU0
        qV[:, 0] = qV0
        qA[:, 0] = qA0
        
        # resolution
        
        print("Starting time-domain resolution...")
        
        for ii in range(1, n_timesteps + 1):
            
            if verbose == True:
                print("Timestep n° ", ii)
            
            spmatrix_ii = Mrom + beta2 * dt**2 * Krom / 2 + dt * Drom / 2
                        
            vector1_ii = From[:, ii - 1]
            vector2_ii = np.dot(Krom, qU[:, ii - 1] + dt * qV[:, ii - 1] + 0.5 * (1 - beta2) * dt**2 * qA[:, ii - 1])
            vector3_ii = np.dot(Drom, qV[:, ii - 1] + dt * qA[:, ii - 1] / 2)
                        
            qA_ii = np.linalg.solve(spmatrix_ii, vector1_ii - vector2_ii - vector3_ii)
            qV_ii = qV[:, ii - 1] + dt * (beta1 * qA_ii + (1 - beta1) * qA[:, ii - 1])
            qU_ii = qU[:, ii - 1] + dt * qV[:, ii - 1] + dt**2 * (beta2 * qA_ii + (1 - beta2) * qA[:, ii - 1]) / 2
        
            qU[:, ii] = qU_ii
            qV[:, ii] = qV_ii
            qA[:, ii] = qA_ii
            
        print("End of time-domain resolution.")
        
        self.__mat_U = np.zeros((self.__n_total_dofs, n_timesteps + 1))
        self.__mat_V = np.zeros((self.__n_total_dofs, n_timesteps + 1))
        self.__mat_A = np.zeros((self.__n_total_dofs, n_timesteps + 1))
        
        mat_UL = np.dot(self.__modesL, qU)
        mat_VL = np.dot(self.__modesL, qV)
        mat_AL = np.dot(self.__modesL, qA)
        
        self.__mat_U[self.__structure.get_mesh().get_free_dofs(), :] = mat_UL
        self.__mat_V[self.__structure.get_mesh().get_free_dofs(), :] = mat_VL
        self.__mat_A[self.__structure.get_mesh().get_free_dofs(), :] = mat_AL
        
    def get_mat_U(self):
        return self.__mat_U
    
    def get_mat_V(self):
        return self.__mat_V
    
    def get_mat_A(self):
        return self.__mat_A