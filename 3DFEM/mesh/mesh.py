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
import scipy.spatial

import importlib.util
spec1 = importlib.util.spec_from_file_location("tet4", "../elements/tet4.py")
tet4 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(tet4)

class Mesh:
    def __init__(self, points):
        self.__n_elements = None
        self.__n_points = points.shape[0]
        self.__n_total_dofs = self.__n_points * 3
        self.__points = points
        self.__ls_dofs_dir = [] # zero-dirichlet condition
        self.__ls_dofs_free = [ii for ii in range(self.__n_total_dofs)]
        self.__ls_dofs_observed = []
        self.__n_dofs_observed = 0

    def set_elements_list(self, elem_list):
        self.__n_elements = len(elem_list)
        self.__elements_list = elem_list

    def delaunay3D_from_points(self, material, remove_degenerate_elements=False, caracteristic_volume=1e-3,
                               min_length=1e-2, max_length=1e0,tolerance=1e1):
        tri = scipy.spatial.Delaunay(self.__points)
        self.__points = tri.points
        self.__n_elements = tri.simplices.shape[0]

        self.__elements_list = []

        elem_counter = 0
        for ii in range(self.__n_elements):

            nodes_ii = tri.simplices[ii, :]
            nodes_coords = self.__points[nodes_ii, :]

            P1 = nodes_coords[0, :]
            P2 = nodes_coords[1, :]
            P3 = nodes_coords[2, :]
            P4 = nodes_coords[3, :]

            signed_volume = np.dot(np.cross((P2-P1), (P3-P1)), (P4-P1)) / 3

            if signed_volume < -1e-8:
                nodes_ii[[0, 1]] = nodes_ii[[1, 0]]
                nodes_coords = self.__points[nodes_ii, :]
                volume = -signed_volume
            else:
                volume = signed_volume

            if remove_degenerate_elements:
                if (caracteristic_volume * 5) > volume > (caracteristic_volume / 5)\
                        and max_length >= np.linalg.norm(P2-P1) >= min_length\
                        and max_length >= np.linalg.norm(P3-P1) >= min_length\
                        and max_length >= np.linalg.norm(P4-P1) >= min_length\
                        and max_length >= np.linalg.norm(P3-P2) >= min_length\
                        and max_length >= np.linalg.norm(P4-P2) >= min_length\
                        and max_length >= np.linalg.norm(P4-P3) >= min_length\
                        and volume > 1e-8:
                    element_ii = tet4.Tet4(material, nodes_coords)
                    element_ii.set_element_number(elem_counter)
                    element_ii.set_nodes_dofs(nodes_ii)
                    self.__elements_list.append(element_ii)
                    elem_counter += 1
            else:
                if volume > 1e-8:
                    element_ii = tet4.Tet4(material, nodes_coords)
                    element_ii.set_element_number(elem_counter)
                    element_ii.set_nodes_dofs(nodes_ii)
                    self.__elements_list.append(element_ii)
                    elem_counter += 1

        self.__n_elements = elem_counter

    def set_new_points(self, points):
        self.__n_points = points.shape[0]
        self.__n_total_dofs = self.__n_points * 3
        self.__points = points

        for ii in range(self.__n_elements):
            nodes_ii = self.__elements_list[ii].get_nodes_nums()
            nodes_coords = self.__points[nodes_ii, :]
            self.__elements_list[ii].set_nodes_coords(nodes_coords)

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

    def get_elements_list(self):
        return self.__elements_list

    def set_dirichlet(self, ls_dofs_dir):
        self.__ls_dofs_dir = ls_dofs_dir
        self.__ls_dofs_free = list(set(range(self.__n_total_dofs)) - set(ls_dofs_dir))
        self.__n_dofs_dir = len(self.__ls_dofs_dir)
        self.__n_dofs_free = len(self.__ls_dofs_free)

    def get_n_dofs_dir(self):
        return self.__n_dofs_dir

    def get_n_dofs_free(self):
        return self.__n_dofs_free

    def get_dirichlet_dofs(self):
        return self.__ls_dofs_dir

    def get_free_dofs(self):
        return self.__ls_dofs_free

    def set_observed_dofs(self, ls_dofs_observed):
        self.__ls_dofs_observed = ls_dofs_observed
        self.__n_dofs_observed = len(ls_dofs_observed)

    def add_observed_dofs(self, dof_observed):
        self.__ls_dofs_observed.append(dof_observed)
        self.__n_dofs_observed += 1

    def get_observed_dofs(self):
        return self.__ls_dofs_observed

    def get_n_observed_dofs(self):
        return self.__n_dofs_observed

    def compute_random_materials(self):
        for ii in range(len(self.__id_list)):
            self.__list_of_elements_lists[ii][0].get_material().compute_random_material()

    def set_random_materials(self, index):
        for ii in range(len(self.__id_list)):
            self.__list_of_elements_lists[ii][0].get_material().set_random_material(index)

    def restore_materials(self):
        for ii in range(len(self.__id_list)):
            self.__list_of_elements_lists[ii][0].get_material().restore_material()

    def compute_sub_elements_lists(self):
        self.__id_list = []
        self.__list_of_elements_lists = []

        for element in self.__elements_list:
            element_id = element.get_material().get_id()
            if element_id not in self.__id_list:
                self.__id_list.append(element_id)
                self.__list_of_elements_lists.append([])
                self.__list_of_elements_lists[self.__id_list.index(element_id)].append(element)
            else:
                self.__list_of_elements_lists[self.__id_list.index(element_id)].append(element)

    def get_id_list(self):
        return self.__id_list

    def get_list_of_elements_lists(self):
        return self.__list_of_elements_lists

    def compute_stress_at_nodes(self, vec_U, return_strain=True):
        mat_nodal_stress = np.zeros((self.__n_points, 6))
        if return_strain:
            mat_nodal_strain = np.zeros((self.__n_points, 6))

        for element in self.__elements_list:
            nodes_nums = element.get_nodes_nums()
            dofs_nums = element.get_dofs_nums()
            vec_Ue = vec_U[dofs_nums]

            if return_strain:
                (list_vec_strain_nodes, list_vec_stress_nodes) = element.compute_element_stress_at_nodes(vec_Ue,
                                                                                                         return_strain)
            else:
                list_vec_stress_nodes = element.compute_element_stress_at_nodes(vec_Ue, return_strain)

            for nn, num in enumerate(nodes_nums):
                mat_nodal_stress[num, :] = list_vec_stress_nodes[nn]
                if return_strain:
                    mat_nodal_strain[num, :] = list_vec_strain_nodes[nn]

        if return_strain:
            return mat_nodal_strain, mat_nodal_stress
        else:
            return mat_nodal_stress
