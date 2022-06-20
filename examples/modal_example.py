import time
import numpy as np

import node_selection.node_selection as ns
from materials.materials import LinearElasticIsotropic
from meshing.mesh import Mesh
from meshing.primitive_shapes import cylinder
from boundary_conditions.boundary_conditions import DirichletBC
from structure.structure import Structure
from solvers.modal import ModalSolver
from plotting.plotting import vtk_mode_animation


##############################################################################
# Modal analysis of a beam
##############################################################################

computation_time_start = time.time()

####
# material
####

print("Defining material...")

material1_rho = 7850
material1_Y = 210e9
material1_nu = 0.29
material1_id = 1

material1 = LinearElasticIsotropic(material1_id, material1_rho, material1_Y, material1_nu)

####
# geometry and meshing
####

print("Defining geometry, meshing and Dirichlet conditions...")

# geometry

L_r = 9e-2
L_z = 2e0

X_0 = (0.0, 0.0, 0.0)

# meshing

n_nodes_r = 4
n_nodes_theta = 12
n_nodes_z = 41

nodes, elements, materials = cylinder(n_nodes_r=n_nodes_r, n_nodes_theta=n_nodes_theta, n_nodes_z=n_nodes_z,
                                      L_r=L_r, L_z=L_z,
                                      X_0=X_0,
                                      material=material1)

beam_mesh = Mesh(name="Beam mesh")
beam_mesh.set_nodes(nodes)
beam_mesh.set_elements(elements)
beam_mesh.set_materials_list(materials)

# Dirichlet conditions

ls_nodes_dir_1 = ns.find_nodes_in_xyplane(beam_mesh.nodes, 0)
ls_nodes_dir_2 = ns.find_nodes_in_xyplane(beam_mesh.nodes, L_z)

ls_nodes_dir_x = ls_nodes_dir_1 + ls_nodes_dir_2
ls_nodes_dir_y = ls_nodes_dir_1 + ls_nodes_dir_2
ls_nodes_dir_z = ls_nodes_dir_1 + ls_nodes_dir_2

ls_dofs_dir_x = [node * 3 for node in ls_nodes_dir_x]
ls_dofs_dir_y = [node * 3 + 1 for node in ls_nodes_dir_y]
ls_dofs_dir_z = [node * 3 + 2 for node in ls_nodes_dir_z]

ls_dofs_dir = np.unique(ls_dofs_dir_x + ls_dofs_dir_y + ls_dofs_dir_z)

dirichlet_BC = DirichletBC()
dirichlet_BC.add_list_of_dirichlet_dofs(ls_dofs_dir)

####
# structures
####

print("Defining structure...")

beam_structure = Structure(beam_mesh, dirichlet_BC)

####
# solvers
####

print("Defining solver...")

n_modes = 10
beam_solver = ModalSolver(beam_structure)


print("Running solver...")

solver_subtime_start = time.time()

beam_solver.run(n_modes=n_modes)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

index_mode = 9
n_frames = 200

file_name = "animation_modal_example"
scale = 1e0
vtk_mode_animation(file_name, beam_solver, index_mode, scale, n_frames)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")
