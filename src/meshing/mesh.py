import numpy as np
from scipy.spatial import Delaunay
from elements import tet4


class Mesh:
    def __init__(self, from_file=None, name='Unnamed Mesh'):
        if type(name) == str:
            self.name = name
        else:
            raise TypeError

        if from_file is None:
            self.nodes = None
            self.n_nodes = None
            self.n_total_dofs = None
            self.elements = None
            self.n_elements = None
            self.materials = []
            self.n_materials = 0
            self.elements_by_material = None
            self.observed_dofs = []
            self.n_observed_dofs = 0
        elif type(from_file) == str:
            todo = True
        else:
            raise TypeError

    def set_nodes(self, array_of_points):
        self.n_nodes = array_of_points.shape[0]
        self.n_total_dofs = self.n_nodes * 3
        self.nodes = array_of_points

    def set_elements(self, list_of_elements):
        self.n_elements = len(list_of_elements)
        self.elements = list_of_elements

    def set_materials_list(self, list_of_materials):
        self.materials = list_of_materials
        self.n_materials = len(list_of_materials)

    def add_material(self, material):
        self.materials.append(material)
        self.n_materials += 1

    def add_observed_dofs(self, dofs):
        self.observed_dofs.extend(dofs)
        self.n_observed_dofs += len(dofs)

    def create_from_points(self, material_id):
        # compute Delaunay triangulation
        tri = Delaunay(self.nodes)
        self.nodes = tri.points
        self.n_elements = tri.simplices.shape[0]

        # loop through generated elements to create element objects and append the list of elements in the mesh
        self.elements = []

        element_counter = 0
        for ii in range(self.n_elements):
            nodes_ii = tri.simplices[ii, :]
            nodes_coords = self.nodes[nodes_ii, :]

            P1 = nodes_coords[0, :]
            P2 = nodes_coords[1, :]
            P3 = nodes_coords[2, :]
            P4 = nodes_coords[3, :]

            signed_volume = np.dot(np.cross((P2-P1), (P3-P1)), (P4-P1)) / 3

            # signed_volume is negative if the generated tetrahedron has a negative jacobian
            # which is fixed by switching 2 nodes in the element
            if signed_volume < 0:
                nodes_ii[[0, 1]] = nodes_ii[[1, 0]]
                nodes_coords = self.nodes[nodes_ii, :]

            if np.abs(signed_volume) >= 1e-8:
                element = tet4.Tet4(number=element_counter, material_id=material_id,
                                    nodes_nums=nodes_ii, nodes_coords=nodes_coords)
                self.elements.append(element)

                element_counter += 1

        self.n_elements = element_counter

    def sort_elements_by_material(self):
        # Sorts elements in a list of lists, one list for each material specified in the list 'self.materials'.
        #

        materials_id_list = []
        self.elements_by_material = []

        for element in self.elements:
            if element.material_id in materials_id_list:
                self.elements_by_material[materials_id_list.index(element.material_id)].append(element)
            else:
                materials_id_list.append(element.material_id)
                self.elements_by_material.append([])
                self.elements_by_material[materials_id_list.index(element.material_id)].append(element)
