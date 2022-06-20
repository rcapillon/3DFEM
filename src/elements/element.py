import numpy as np


class Element:
    def __init__(self, number, material_id, nodes_nums, nodes_coords):
        self.number = number
        self.material_id = material_id

        self.nodes_nums = nodes_nums
        list_dofs_nums = []
        for node in self.nodes_nums:
            list_dofs_nums.append(node * 3)
            list_dofs_nums.append(node * 3 + 1)
            list_dofs_nums.append(node * 3 + 2)
        self.dofs_nums = np.array(list_dofs_nums)

        self.n_nodes = len(nodes_nums)
        self.n_dofs = 3 * self.n_nodes

        self.nodes_coords = nodes_coords
        self.vec_nodes_coords = np.reshape(self.nodes_coords, self.n_dofs)

    def calculate_dofs_nums(self):
        dofs_nums = []
        for node in self.nodes_nums:
            dofs_nums.append(node * 3)
            dofs_nums.append(node * 3 + 1)
            dofs_nums.append(node * 3 + 2)
        dofs_nums = np.array(dofs_nums)

        return dofs_nums
