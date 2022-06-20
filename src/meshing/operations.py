import numpy as np
from meshing.mesh import Mesh


def merge_meshes(input_mesh_1, input_mesh_2, tolerance=1e-6):
    # Merges two meshes into a single mesh object by fusing nodes whose distance is inferior to a specified tolerance
    #
    # input_mesh_1 : first mesh to be merged
    # --
    # input_mesh_2 : second mesh to be merged
    # --
    # tolerance : distance under which two nodes are considered the same and fused
    #

    def find_point_up_to_tolerance(coords_to_find, points, tol):
        n_points = points.shape[0]
        found = False

        idx = 0
        while not found and idx < n_points:
            points_coords = points[idx, :]

            if np.linalg.norm(points_coords - coords_to_find) <= tol:
                found = True
            else:
                idx += 1

        if not found:
            idx = -1

        return idx

    name_output_mesh = input_mesh_1.name + '+' + input_mesh_2.name
    output_mesh = Mesh(name=name_output_mesh)

    nodes_1 = input_mesh_1.nodes
    nodes_2 = input_mesh_2.nodes
    n_nodes_1 = nodes_1.shape[0]
    n_nodes_2 = nodes_2.shape[0]
    elements_1 = input_mesh_1.elements
    elements_2 = input_mesh_2.elements

    new_nodes = nodes_1
    table_of_nodes_replacements = np.zeros((n_nodes_2,))
    counter_added_nodes = 0
    for ii in range(nodes_2.shape[0]):
        node_coords_2 = np.zeros((1, 3))
        node_coords_2[0, :] = nodes_2[ii, :]
        index = find_point_up_to_tolerance(node_coords_2, nodes_1, tolerance)
        if index == -1:
            new_nodes = np.vstack((new_nodes, node_coords_2))
            table_of_nodes_replacements[ii] = n_nodes_1 + counter_added_nodes
            counter_added_nodes += 1
        else:
            table_of_nodes_replacements[ii] = index

    new_elements = elements_1
    for element in elements_2:
        new_element = []
        for node in element:
            new_element.append(table_of_nodes_replacements[node])
        new_elements.append(new_element)

    output_mesh.set_nodes(new_nodes)
    output_mesh.set_elements(new_elements)

    return output_mesh
