import numpy as np
from elements import brick8, prism6


def brick(n_nodes_x=21, n_nodes_y=4, n_nodes_z=3,
          L_x=2e0, L_y=3e-1, L_z=2e-1,
          X_0=(0.0, 0.0, 0.0),
          material=None):
    # Returns elements and/or nodes of a rectangular box discretized with brick8 elements.
    #
    # n_nodes_x : number of nodes along the x-axis
    # --
    # n_nodes_y : number of nodes along the y-axis
    # --
    # n_nodes_z : number of nodes along the z-axis
    # --
    # L_x       : dimension of the box along the x-axis
    # --
    # L_y       : dimension of the box along the y-axis
    # --
    # L_z       : dimension of the box along the z-axis
    # --
    # X_0       : tuple containing the coordinates of the point with the lowest x, y and z coordinates
    # --
    # material  : instance of the Material class to be assigned to every element of the mesh.
    #

    if n_nodes_x < 2:
        n_nodes_x = 2
    if n_nodes_y < 2:
        n_nodes_y = 2
    if n_nodes_z < 2:
        n_nodes_z = 2

    n_nodes_total = n_nodes_x * n_nodes_y * n_nodes_z
    n_elements_x = n_nodes_x - 1
    n_elements_y = n_nodes_y - 1
    n_elements_z = n_nodes_z - 1

    element_L_x = L_x / n_elements_x
    element_L_y = L_y / n_elements_y
    element_L_z = L_z / n_elements_z

    # element_volume = element_L_x * element_L_y * element_L_z

    nodes = np.zeros((n_nodes_total, 3))
    for ny in range(n_nodes_y):
        for nz in range(n_nodes_z):
            for nx in range(n_nodes_x):
                node_idx = ny + nz * n_nodes_y + nx * n_nodes_y * n_nodes_z
                nodes[node_idx, 0] = nx * element_L_x + X_0[0]
                nodes[node_idx, 1] = ny * element_L_y + X_0[1]
                nodes[node_idx, 2] = nz * element_L_z + X_0[2]

    if material is not None:
        materials = [material]

        elements = []
        for ey in range(n_elements_y):
            for ez in range(n_elements_z):
                for ex in range(n_elements_x):
                    element_number = ey + ez * n_elements_y + ex * n_elements_y * n_elements_z

                    node1_idx = ey + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node2_idx = ey+1 + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node3_idx = ey+n_nodes_y+1 + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node4_idx = ey+n_nodes_y + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node5_idx = ey+n_nodes_y*n_nodes_z + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node6_idx = ey+n_nodes_y*n_nodes_z+1 + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node7_idx = ey+n_nodes_y+1+n_nodes_y*n_nodes_z + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z
                    node8_idx = ey+n_nodes_y+n_nodes_y*n_nodes_z + ez*n_nodes_y + ex*n_nodes_y*n_nodes_z

                    nodes_nums = [node1_idx, node2_idx, node3_idx, node4_idx,
                                  node5_idx, node6_idx, node7_idx, node8_idx]
                    nodes_coords = nodes[nodes_nums, :]

                    element = brick8.Brick8(number=element_number, material_id=material.id,
                                            nodes_nums=nodes_nums, nodes_coords=nodes_coords)

                    elements.append(element)

        return nodes, elements, materials

    else:
        return nodes


def cylinder(n_nodes_r=4, n_nodes_theta=10, n_nodes_z=21,
             L_r=1.5e-1, L_z=2e0,
             X_0=(0.0, 0.0, 0.0),
             material=None):
    # Returns elements and/or nodes of a discreztized cylinder whose main axis is the z-axis.
    # Elements touching the center line are prism6 elements. Others are brick8 elements.
    #
    # n_nodes_r     : number of nodes along the radius of the cross-section
    # --
    # n_nodes_theta : number of nodes around the cross-section at a given radius
    # --
    # n_nodes_z     : number of nodes along the z-axis (main axis)
    # --
    # L_r           : radius of the cross-section
    # --
    # L_z           : length of the cylinder along its main axis
    # --
    # X_0           : tuple containing the coordinates of the point on the center line with the lowest z-coordinate
    # --
    # material      : instance of the Material class to be assigned to every element of the mesh.
    #

    if n_nodes_r < 2:
        n_nodes_r = 2
    if n_nodes_theta < 2:
        n_nodes_theta = 2
    if n_nodes_z < 2:
        n_nodes_z = 2

    n_nodes_total = n_nodes_z * (1 + (n_nodes_r-1) * n_nodes_theta)
    n_elements_r = n_nodes_r - 1
    n_elements_theta = n_nodes_theta
    n_elements_z = n_nodes_z - 1

    element_L_r = L_r / n_elements_r
    element_L_z = L_z / n_elements_z
    element_angle_theta = 2 * np.pi / n_elements_theta

    line_z = np.array([nz*element_L_z for nz in range(n_nodes_z)])

    central_line_points = np.zeros((n_nodes_z, 3))
    central_line_points[:, 2] = line_z
    nodes_numbers_central_line = np.array([nz for nz in range(n_nodes_z)])

    nodes = np.zeros((n_nodes_total, 3))
    nodes[:n_nodes_z, :] = central_line_points

    counter_lines = 1
    prism_outer_lines = []

    for nt in range(n_nodes_theta):
        outer_line_points = np.zeros((n_nodes_z, 3))
        outer_line_points[:, 0] = element_L_r * np.cos(nt * element_angle_theta)
        outer_line_points[:, 1] = element_L_r * np.sin(nt * element_angle_theta)
        outer_line_points[:, 2] = line_z

        nodes_idx_start = counter_lines * n_nodes_z
        nodes_idx_end = nodes_idx_start + n_nodes_z

        nodes[nodes_idx_start:nodes_idx_end, :] = outer_line_points
        prism_outer_lines.append(outer_line_points)

        counter_lines += 1

    brick_inner_lines = prism_outer_lines
    for nr in range(n_elements_r - 1):
        brick_outer_lines = []

        for nt in range(n_nodes_theta):
            inner_line_points = brick_inner_lines[nt]

            outer_line_points = np.zeros((n_nodes_z, 3))
            outer_line_points[:, 0] = inner_line_points[:, 0] + element_L_r * np.cos(nt * element_angle_theta)
            outer_line_points[:, 1] = inner_line_points[:, 1] + element_L_r * np.sin(nt * element_angle_theta)
            outer_line_points[:, 2] = inner_line_points[:, 2]

            nodes_idx_start = counter_lines * n_nodes_z
            nodes_idx_end = nodes_idx_start + n_nodes_z

            nodes[nodes_idx_start:nodes_idx_end, :] = outer_line_points
            brick_outer_lines.append(outer_line_points)

            counter_lines += 1

        brick_inner_lines = brick_outer_lines

    nodes[:, 0] += X_0[0]
    nodes[:, 1] += X_0[1]
    nodes[:, 2] += X_0[2]

    if material is not None:
        materials = [material]

        elements = []

        counter_elements = 0
        # prism_volume = element_L_r * element_L_r * np.sin(element_angle_theta) * element_L_z / 2

        counter_lines_2 = 1
        nodes_numbers_first_outer_line = nodes_numbers_central_line + n_elements_theta * n_nodes_z

        for et in range(n_elements_theta):
            nodes_numbers_second_outer_line = nodes_numbers_central_line + counter_lines_2 * n_nodes_z

            for ez in range(n_elements_z):
                node1_num = nodes_numbers_first_outer_line[ez]
                node2_num = nodes_numbers_second_outer_line[ez]
                node3_num = nodes_numbers_central_line[ez]
                node4_num = nodes_numbers_first_outer_line[ez + 1]
                node5_num = nodes_numbers_second_outer_line[ez + 1]
                node6_num = nodes_numbers_central_line[ez + 1]

                nodes_nums = [node1_num, node2_num, node3_num, node4_num, node5_num, node6_num]
                nodes_coords = nodes[nodes_nums, :]

                prism_element = prism6.Prism6(number=counter_elements, material_id=material.id,
                                              nodes_nums=nodes_nums, nodes_coords=nodes_coords)
                elements.append(prism_element)

                counter_elements += 1

            nodes_numbers_first_outer_line = nodes_numbers_second_outer_line
            counter_lines_2 += 1

        for er in range(n_elements_r - 1):
            # inner_r = (er + 1) * element_L_r
            # outer_r = (er + 2) * element_L_r
            # brick_volume_r = element_L_z * (outer_r**2 - inner_r**2) * np.cos(element_angle_theta) / 2

            nodes_numbers_first_inner_line = nodes_numbers_central_line + (er + 1) * n_elements_theta * n_nodes_z
            nodes_numbers_first_outer_line = nodes_numbers_central_line + (er + 2) * n_elements_theta * n_nodes_z

            counter_lines_3 = 1

            for et in range(n_elements_theta):
                nodes_numbers_second_inner_line = nodes_numbers_central_line + counter_lines_3 * n_nodes_z \
                                                  + er * n_elements_theta * n_nodes_z
                nodes_numbers_second_outer_line = nodes_numbers_central_line + counter_lines_3 * n_nodes_z \
                                                  + (er + 1) * n_elements_theta * n_nodes_z

                for ez in range(n_elements_z):
                    node1_num = nodes_numbers_first_outer_line[ez]
                    node2_num = nodes_numbers_second_outer_line[ez]
                    node3_num = nodes_numbers_second_inner_line[ez]
                    node4_num = nodes_numbers_first_inner_line[ez]
                    node5_num = nodes_numbers_first_outer_line[ez + 1]
                    node6_num = nodes_numbers_second_outer_line[ez + 1]
                    node7_num = nodes_numbers_second_inner_line[ez + 1]
                    node8_num = nodes_numbers_first_inner_line[ez + 1]

                    nodes_nums = [node1_num, node2_num, node3_num, node4_num,
                                  node5_num, node6_num, node7_num, node8_num]
                    nodes_coords = nodes[nodes_nums, :]

                    brick_element = brick8.Brick8(number=counter_elements, material_id=material.id,
                                                  nodes_nums=nodes_nums, nodes_coords=nodes_coords)
                    elements.append(brick_element)

                    counter_elements += 1

                nodes_numbers_first_inner_line = nodes_numbers_second_inner_line
                nodes_numbers_first_outer_line = nodes_numbers_second_outer_line

                counter_lines_3 += 1

        return nodes, elements, materials

    else:
        return nodes
