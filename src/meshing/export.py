import os


def export_to_txt(mesh):
    # Exports the mesh object to a text file which can be imported for a later use.
    #
    # mesh : Instance of the 'Mesh' class to be exported to .txt format.
    #

    # create file using mesh name
    file = open(mesh.name + ".txt", "w")

    # special lines
    line_empty = '\n'
    line_delimiter = '----\n'
    line_eof = 'EOF'

    # define header section
    line_header_name = mesh.name + '\n'
    line_header_1 = 'Mesh file usable by the 3DFEM python code:\n'
    line_header_2 = 'https://github.com/rcapillon/3DFEM\n'
    line_header_3 = line_empty
    line_header_4 = line_empty
    line_header_5 = line_empty

    # write header
    file.write(line_header_name)
    file.write(line_delimiter)
    file.write(line_header_1)
    file.write(line_header_2)
    file.write(line_header_3)
    file.write(line_header_4)
    file.write(line_header_5)
    file.write(line_delimiter)

    # Nodes section
    #   section title
    line_nodes_title = 'Nodes\n'
    file.write(line_nodes_title)
    #   number of nodes
    n_nodes = mesh.n_nodes
    line_nodes_total = str(n_nodes) + '\n'
    file.write(line_nodes_total)
    #   delimiter between coordinates
    nodes_delimiter = ';'
    #   node list with node number and x, y, z coordinates
    #   across 2 lines for each node
    for nn in range(n_nodes):
        node_x_coord = mesh.nodes[nn, 0]
        node_y_coord = mesh.nodes[nn, 1]
        node_z_coord = mesh.nodes[nn, 2]

        line_node_number = str(nn) + '\n'
        line_node_coords = str(node_x_coord) + nodes_delimiter \
                         + str(node_y_coord) + nodes_delimiter \
                         + str(node_z_coord) + nodes_delimiter + '\n'

        file.write(line_node_number)
        file.write(line_node_coords)
    file.write(line_delimiter)

    # Elements section
    #    section title
    line_elements_title = 'Elements\n'
    file.write(line_elements_title)
    #    number of elements
    n_elements = mesh.n_elements
    line_elements_total = str(n_elements) + '\n'
    file.write(line_elements_total)
    #    delimiter between node numbers
    elements_delimiter = ';'
    #    element list with element number, type, id of assigned material and numbers of assigned nodes
    #    across 4 lines for each element

    for element in mesh.elements:
        line_element_number = str(element.number) + '\n'
        line_element_type = str(element.type) + '\n'
        line_element_material_id = str(element.material_id) + '\n'

        file.write(line_element_number)
        file.write(line_element_type)
        file.write(line_element_material_id)

        n_nodes = element.n_nodes
        line_element_node_numbers = ''
        for nn, node_number in enumerate(element.node_numbers):
            if nn < n_nodes - 1:
                line_element_node_numbers += str(node_number) + elements_delimiter
            else:
                line_element_node_numbers += str(node_number)
        line_element_node_numbers += '\n'
        file.write(line_element_node_numbers)
    file.write(line_delimiter)

    # Materials section
    #    section title
    line_materials_title = 'Materials\n'
    file.write(line_materials_title)
    #    number of materials
    n_materials = mesh.n_materials
    line_materials_total = str(n_materials) + '\n'
    file.write(line_materials_total)
    #    material list with material id number, type across 2 lines
    #    and property name and value across 2 lines for each property
    for material in mesh.materials:
        line_material_id = str(material.id) + '\n'
        line_material_type = str(material.type) + '\n'

        file.write(line_material_id)
        file.write(line_material_type)

        for property_name, property_value in material.dict_of_properties:
            line_material_property_name = str(property_name) + '\n'
            line_material_property_value = str(property_value) + '\n'

            file.write(line_material_property_name)
            file.write(line_material_property_value)
    file.write(line_delimiter)

    # End of file
    # Any line after this one will not be read by the import method of the Mesh class
    file.write(line_eof)


def export_mesh_to_vtk(file_name, mesh, n_points=None, n_faces=None, n_cols=None):
    folder = "vtk_files/"
    os.makedirs(folder, exist_ok=True)

    file = open(folder + file_name + ".vtk", "w")

    if n_points is None:
        n_points = mesh.n_nodes

    if n_faces is None:
        n_faces = 0
    if n_cols is None:
        n_cols = 0

    if n_faces is None or n_cols is None:
        for element in mesh.get_elements_list():
            if n_faces is None:
                n_faces += len(element.faces)
            if n_cols is None:
                for face in element.faces:
                    n_cols += 1 + len(face)

    str_beginning = "# vtk DataFile Version 1.0\n" + file_name + "\nASCII\n\nDATASET POLYDATA\nPOINTS " \
                    + str(n_points) + " float\n"
    file.write(str_beginning)

    for ii in range(n_points):
        point_ii = mesh.nodes[ii, :]
        point_x = point_ii[0]
        point_y = point_ii[1]
        point_z = point_ii[2]

        str_points = "%.6f" % point_x + " " + "%.6f" % point_y + " " + "%.6f" % point_z + "\n"

        file.write(str_points)

    polygons = "POLYGONS " + str(n_faces) + " " + str(n_cols) + "\n"
    file.write(polygons)

    for element in mesh.elements:
        for face in element.faces:
            str_face = str(len(face))
            for node_num in face:
                str_face += " " + str(element.nodes_nums[node_num])
            file.write(str_face + "\n")

    file.close()
