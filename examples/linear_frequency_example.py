import time
import numpy as np

import node_selection.node_selection as ns
from materials.materials import LinearElasticIsotropic
from meshing.mesh import Mesh
from meshing.primitive_shapes import brick
from boundary_conditions.boundary_conditions import DirichletBC, NeumannBC
from structure.structure import Structure
from solvers.linear_diagonal_frequency import LinearDiagonalFrequencySolver
from solvers.linear_frequency import LinearFrequencySolver
from plotting.plotting import plot_mat_U_observed, vtk_mesh_U_animation

##############################################################################
# Frequency-domain analysis of a plate
##############################################################################

computation_time_start = time.time()

####
# frequency steps
####

print("Defining frequency band of analysis and sampling rate...")

f0 = 12000
fmax = 16000
n_freqsteps = 100

vec_freq = np.linspace(f0, fmax, n_freqsteps)

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

L_x = 1e0
L_y = 1e0
L_z = 1e-1

# meshing

Nn_x = 41
Nn_y = 41
Nn_z = 5

nodes, elements, materials = brick(n_nodes_x=Nn_x, n_nodes_y=Nn_y, n_nodes_z=Nn_z,
                                   L_x=L_x, L_y=L_y, L_z=L_z,
                                   X_0=(0.0, 0.0, 0.0),
                                   material=material1)

plate_mesh = Mesh(name="Plate mesh")
plate_mesh.set_nodes(nodes)
plate_mesh.set_elements(elements)
plate_mesh.set_materials_list(materials)

# Observed DOFs

observed_node1_coords = np.array([L_x/2, L_y/2, L_z])
observed_node1_number = ns.find_nodes_with_coordinates(plate_mesh.nodes, observed_node1_coords)[0]
observed_dof1_number = observed_node1_number * 3
observed_dof2_number = observed_node1_number * 3 + 1
observed_dof3_number = observed_node1_number * 3 + 2

observed_node2_coords = np.array([L_x/4, L_y/4, L_z])
observed_node2_number = ns.find_nodes_with_coordinates(plate_mesh.nodes, observed_node2_coords)[0]
observed_dof4_number = observed_node2_number * 3
observed_dof5_number = observed_node2_number * 3 + 1
observed_dof6_number = observed_node2_number * 3 + 2

ls_dofs_observed = [observed_dof1_number, observed_dof2_number, observed_dof3_number,
                    observed_dof4_number, observed_dof5_number, observed_dof6_number]

plate_mesh.add_observed_dofs(ls_dofs_observed)

# Dirichlet conditions

ls_nodes_dir_1 = ns.find_nodes_in_xyplane(plate_mesh.nodes, 0)
ls_nodes_dir_2 = ns.find_nodes_in_yzplane(plate_mesh.nodes, 0)
ls_nodes_dir_3 = ns.find_nodes_in_yzplane(plate_mesh.nodes, L_x)
ls_nodes_dir_4 = ns.find_nodes_in_xzplane(plate_mesh.nodes, 0)
ls_nodes_dir_5 = ns.find_nodes_in_xzplane(plate_mesh.nodes, L_y)

ls_nodes_dir_x = ls_nodes_dir_1 + ls_nodes_dir_2 + ls_nodes_dir_3 + ls_nodes_dir_4 + ls_nodes_dir_5
ls_nodes_dir_y = ls_nodes_dir_1 + ls_nodes_dir_2 + ls_nodes_dir_3 + ls_nodes_dir_4 + ls_nodes_dir_5
ls_nodes_dir_z = ls_nodes_dir_1 + ls_nodes_dir_2 + ls_nodes_dir_3 + ls_nodes_dir_4 + ls_nodes_dir_5

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

plate_structure = Structure(plate_mesh, dirichlet_BC)

alpha_M = 0
alpha_K = 4e-7
plate_structure.set_rayleigh_parameters(alpha_M, alpha_K)

####
# forces
####

print("Defining force...")

neumann_BC = NeumannBC()

force_coords = np.array([L_x/2, L_y/2, L_z])
ls_nodes_force = ns.find_nodes_with_coordinates(plate_mesh.nodes, force_coords)

nodal_force_vector = np.array([0, 0, -1e0])
neumann_BC.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

vec_variation = np.ones((n_freqsteps,))
neumann_BC.set_variation(vec_variation)

####
# reduced-order model
####

print("Defining reduced-order model...")

n_modes = 100

####
# solvers
####

print("Defining solver...")

solver_type = "diagonal"

if solver_type == "diagonal":
    plate_solver = LinearDiagonalFrequencySolver(plate_structure, neumann_BC=neumann_BC)
else:
    plate_solver = LinearFrequencySolver(plate_structure, neumann_BC=neumann_BC)

print("Running solver...")

solver_subtime_start = time.time()

plate_solver.run(vec_freq, n_modes,
                 verbose=False)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

file_name_frf = "plot_linear_frequency_example"
plot_mat_U_observed(file_name_frf, plate_solver,
                    x_name="Frequency (Hz)", y_name="Displacement amplitude (m)", plot_type="semilogy")

file_name_animation = "animation_linear_frequency_example"
scale = 8e8
vtk_mesh_U_animation(file_name_animation, plate_solver, scale)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")