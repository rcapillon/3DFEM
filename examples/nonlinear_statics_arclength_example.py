import time
import numpy as np

import node_selection.node_selection as ns
from materials.materials import LinearElasticIsotropic
from meshing.mesh import Mesh
from meshing.primitive_shapes import cylinder
from boundary_conditions.boundary_conditions import DirichletBC, NeumannBC
from structure.structure import Structure
from solvers.nonlinear_statics_arclength import NonlinearStaticsArcLengthSolver
from plotting.plotting import vtk_nonlin_U_animation, \
                              plot_mat_U_observed_load_displacement, \
                              plot_norm_U_load_displacement, \
                              plot_norm_U_load_displacement_animation

##############################################################################
# Bending of a curved beam
##############################################################################

computation_time_start = time.time()

####
# material
####

print("Defining materials...")

material1_rho = 1000
material1_Y = 1e6
material1_nu = 0.45
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
n_nodes_z = 51

nodes, elements, materials = cylinder(n_nodes_r=n_nodes_r, n_nodes_theta=n_nodes_theta, n_nodes_z=n_nodes_z,
                                      L_r=L_r, L_z=L_z,
                                      X_0=X_0,
                                      material=material1)

beam_mesh = Mesh(name="Beam mesh")
beam_mesh.set_nodes(nodes)
beam_mesh.set_elements(elements)
beam_mesh.set_materials_list(materials)

# Observed DOFs

observed_node1_coords = np.array([0.0, 0.0, L_z/2])
observed_node1_number = ns.find_nodes_with_coordinates_within_tolerance(beam_mesh.nodes, observed_node1_coords, 1e-6)[0]
observed_dof1_number = observed_node1_number * 3
observed_dof2_number = observed_node1_number * 3 + 1
observed_dof3_number = observed_node1_number * 3 + 2

ls_dofs_observed = [observed_dof1_number, observed_dof2_number, observed_dof3_number]

beam_mesh.add_observed_dofs(ls_dofs_observed)

# Dirichlet conditions

ls_nodes_dir_1 = ns.find_nodes_in_xyplane_within_tolerance(beam_mesh.nodes, 0, tol=1e-6)
ls_nodes_dir_2 = ns.find_nodes_in_xyplane_within_tolerance(beam_mesh.nodes, L_z, tol=1e-6)

ls_dofs_dir_x = [node*3 for node in ls_nodes_dir_1] + [node*3 for node in ls_nodes_dir_2]
ls_dofs_dir_y = [node*3+1 for node in ls_nodes_dir_1] + [node*3+1 for node in ls_nodes_dir_2]
ls_dofs_dir_z = [node*3+2 for node in ls_nodes_dir_1] + [node*3+2 for node in ls_nodes_dir_2]

ls_dofs_dir = ls_dofs_dir_x + ls_dofs_dir_y + ls_dofs_dir_z

dirichlet_BC = DirichletBC()
dirichlet_BC.add_list_of_dirichlet_dofs(ls_dofs_dir)

####
# forces
####

print("Defining forces...")

force_1_value = -1e0

neumann_BC = NeumannBC()

node_force_1_coords = np.array([0.0, 0.0, L_z/2])
ls_nodes_force_1 = ns.find_nodes_with_coordinates_within_tolerance(beam_mesh.nodes, node_force_1_coords, tol=1e-6)
nodal_force_vector_1 = np.array([force_1_value, 0.0, 0.0])
neumann_BC.add_nodal_forces_t0(ls_nodes_force_1, nodal_force_vector_1)

####
# Mesh transformation: curving of the beam along a circle
####

x0 = -15*L_r
z0 = L_z/2

vec_X0 = np.array([x0, z0])
R = np.linalg.norm(vec_X0)

nodes = beam_mesh.nodes

for ii in range(beam_mesh.n_nodes):
    vec_X = nodes[ii, :]

    R_ii = R + vec_X[0]

    vec_r = vec_X[[0, 2]] - vec_X0
    r = np.linalg.norm(vec_r)

    vec_u = vec_r / r

    k = R_ii - r

    nodes[ii, [0, 2]] += k * vec_u

beam_mesh.set_nodes(nodes)

####
# structures
####

print("Defining structures...")

beam_structure = Structure(beam_mesh, dirichlet_BC)

####
# solvers
####

print("Defining solvers...")

beam_solver = NonlinearStaticsArcLengthSolver(beam_structure, neumann_BC=neumann_BC)

n_arclengths = 70
Delta_L = 1e4
n_iter_max = 20
psi = 1e2

tol = 1e-9

corrections = "spherical"
attenuation = 0.5
n_restart = 8
n_grow = 10
n_switch = 4
corrector_root_selection = "default"
n_selection = 1

print("Running solvers...")

solver_subtime_start = time.time()

beam_solver.run(Delta_L, psi=psi, n_arclengths=n_arclengths, n_iter_max=n_iter_max, tol=tol,
                corrections=corrections, corrector_root_selection=corrector_root_selection,
                attenuation=attenuation, n_selection=n_selection,
                n_restart=n_restart, n_grow=n_grow, n_switch=n_switch,
                verbose=True)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

file_name_plot = "plot_nonlinear_statics_arclength_example_"
plot_mat_U_observed_load_displacement(file_name_plot, beam_solver,
                                      x_name="Displacement (m)", y_name="Load factor", plot_type="linear")

file_name_norm = "plot_norm_U_nonlinear_statics_arclength_example_"
plot_norm_U_load_displacement(file_name_norm, beam_solver,
                              x_name="Displacement (m)", y_name="Load factor", plot_type="linear")

file_name_norm_animation = "animation_norm_U_nonlinear_statics_arclength_example_"
stepsize = 1
plot_norm_U_load_displacement_animation(file_name_norm_animation, beam_solver, stepsize, fps=10)

file_name_animation = "animation_nonlinear_statics_arclength_example_"
scale = 1e0
vtk_nonlin_U_animation(file_name_animation, beam_solver, scale)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")
