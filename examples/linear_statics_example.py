import time
import numpy as np

import node_selection.node_selection as ns
from materials.materials import LinearElasticIsotropic
from meshing.mesh import Mesh
from meshing.primitive_shapes import cylinder
from boundary_conditions.boundary_conditions import DirichletBC, NeumannBC
from structure.structure import Structure
from solvers.linear_statics import LinearStaticsSolver
from plotting.plotting import vtk_U_on_mesh, vtk_undeformed_mesh

##############################################################################
# Static analysis of a beam
##############################################################################

computation_time_start = time.time()

####
# material
####

print("Defining materials...")

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

list_node1_dir = ns.find_nodes_in_xyplane(beam_mesh.nodes, 0.0)
list_node2_dir = ns.find_nodes_in_xyplane(beam_mesh.nodes, L_z)

ls_nodes_dir_x = list_node1_dir + list_node2_dir
ls_nodes_dir_y = list_node1_dir + list_node2_dir
ls_nodes_dir_z = list_node1_dir + list_node2_dir

ls_dofs_dir_x = [node * 3 for node in ls_nodes_dir_x]
ls_dofs_dir_y = [node * 3 + 1 for node in ls_nodes_dir_y]
ls_dofs_dir_z = [node * 3 + 2 for node in ls_nodes_dir_z]

ls_dofs_dir = np.unique(ls_dofs_dir_x + ls_dofs_dir_y + ls_dofs_dir_z)

dirichlet_BC = DirichletBC()
dirichlet_BC.add_list_of_dirichlet_dofs(ls_dofs_dir)

####
# structures
####

print("Defining structures...")

beam_structure = Structure(beam_mesh, dirichlet_BC)

####
# forces
####

print("Defining forces...")

neumann_BC = NeumannBC()


node1_force_coords = np.array([0.0, 0.0, L_z/2])
list_node1_force = ns.find_nodes_with_coordinates(beam_mesh.nodes, node1_force_coords)

nodal_force_vector = np.array([1e8, 0.0, 0.0])

neumann_BC.add_nodal_forces_t0(list_node1_force, nodal_force_vector)

####
# solvers
####

file_name_2 = "./image_undeformed_mesh_linear_statics_example"
vtk_undeformed_mesh(file_name_2, beam_mesh)

print("Defining solvers...")

beam_solver = LinearStaticsSolver(beam_structure, neumann_BC)

print("Running solvers...")

solver_subtime_start = time.time()

beam_solver.run()

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

vec_U = beam_solver.vec_U

file_name_1 = "./image_deformed_mesh_linear_statics_example"
scale = 1e0
vtk_U_on_mesh(file_name_1, beam_solver, vec_U, scale)

file_name_2 = "./image_undeformed_mesh_linear_statics_example"
vtk_undeformed_mesh(file_name_2, beam_mesh)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")