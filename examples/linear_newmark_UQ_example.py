import time
import numpy as np

import node_selection.node_selection as ns
from materials.materials import LinearElasticIsotropic
from meshing.mesh import Mesh
from meshing.primitive_shapes import brick
from boundary_conditions.boundary_conditions import DirichletBC, NeumannBC
from initial_conditions.initial_conditions import InitialConditions
from structure.structure import Structure
from solvers.linear_newmark_UQ import LinearNewmarkUQSolver
from plotting.plotting import plot_array_U_rand_observed, plot_array_U_rand_observed_ksdensity

##############################################################################
# Parametric stochastic frequency-domain analysis of a beam
##############################################################################

computation_time_start = time.time()

####
# rng seeding
####

rng_seed = 12345
np.random.seed(rng_seed)

####
# stochastic parameters
####

print("Defining stochastic analysis parameters...")

uncertainty_type = "generalized"
n_samples = 2000

# parametric dispersion parameters

# delta_Y = 1 / np.sqrt(2)
delta_Y = 0.05

# nonparametric dispersion parameters

# delta_M = np.sqrt((n_modes + 1) / (n_modes + 5))
# delta_K = np.sqrt((n_modes + 1) / (n_modes + 5))
delta_M = 0.05
delta_K = 0.05

####
# timesteps
####

print("Defining frequency band of analysis and sampling rate...")

n_timesteps = 500
t0 = 0.0
dt = 4e-4
tmax = n_timesteps * dt

####
# material
####

print("Defining materials...")

material1_rho = 7850
material1_Y = 210e9
material1_nu = 0.29
material1_id = 1

material1 = LinearElasticIsotropic(material1_id, material1_rho, material1_Y, material1_nu)
material1.set_dispersion_coefficient_Y(delta_Y)

####
# geometry and meshing
####

print("Defining geometry, meshing and Dirichlet conditions...")

# geometry

L_x = 1e0
L_y = 1e-1
L_z = 1e-1

# meshing

Nn_x = 41
Nn_y = 5
Nn_z = 5

nodes, elements, materials = brick(n_nodes_x=Nn_x, n_nodes_y=Nn_y, n_nodes_z=Nn_z,
                                   L_x=L_x, L_y=L_y, L_z=L_z,
                                   X_0=(0.0, 0.0, 0.0),
                                   material=material1)

beam_mesh = Mesh(name="Beam mesh")
beam_mesh.set_nodes(nodes)
beam_mesh.set_elements(elements)
beam_mesh.set_materials_list(materials)

# Observed DOFs

observed_node1_coords = np.array([L_x / 2, L_y, L_z / 2])
observed_node1_number = ns.find_nodes_with_coordinates(beam_mesh.nodes, observed_node1_coords)[0]
observed_dof1_number = observed_node1_number * 3
observed_dof2_number = observed_node1_number * 3 + 1
observed_dof3_number = observed_node1_number * 3 + 2

observed_node2_coords = np.array([L_x, 3 * L_y / 4, L_z])
observed_node2_number = ns.find_nodes_with_coordinates(beam_mesh.nodes, observed_node2_coords)[0]
observed_dof4_number = observed_node2_number * 3
observed_dof5_number = observed_node2_number * 3 + 1
observed_dof6_number = observed_node2_number * 3 + 2

ls_dofs_observed = [observed_dof1_number, observed_dof2_number, observed_dof3_number, observed_dof4_number,
                    observed_dof5_number, observed_dof6_number]

beam_mesh.add_observed_dofs(ls_dofs_observed)

# Dirichlet conditions

ls_nodes_dir_1 = ns.find_nodes_in_yzplane(beam_mesh.nodes, 0)

ls_nodes_dir_x = ls_nodes_dir_1
ls_nodes_dir_y = ls_nodes_dir_1
ls_nodes_dir_z = ls_nodes_dir_1

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

alpha_M = 0
alpha_K = 2e-4
beam_structure.set_rayleigh_parameters(alpha_M, alpha_K)

U0 = np.zeros((beam_mesh.n_total_dofs,))
V0 = np.zeros((beam_mesh.n_total_dofs,))
A0 = np.zeros((beam_mesh.n_total_dofs,))

initial_conditions = InitialConditions(U0=U0, V0=V0, A0=A0)

beam_structure.dispersion_coefficient_M = delta_M
beam_structure.dispersion_coefficient_K = delta_K

####
# forces
####

print("Defining forces...")

neumann_BC = NeumannBC()

force_coords = np.array([L_x, L_y/2, L_z/2])
ls_nodes_force = ns.find_nodes_with_coordinates(beam_mesh.nodes, force_coords)

nodal_force_vector = np.array([0, 0, 2e6])
neumann_BC.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

t1 = tmax/8
t2 = tmax/4
vec_t = np.linspace(t0, tmax, n_timesteps + 1)

vec_variation = np.zeros((n_timesteps + 1,))
vec_variation[vec_t <= t1] = (1 / t1) * vec_t[vec_t <= t1]
vec_variation[vec_t > t1] = 1.0
vec_variation[vec_t > t2] = 0.0

neumann_BC.set_variation(vec_variation)

####
# reduced-order model
####

print("Defining reduced-order model...")

n_modes = 10

####
# solvers
####

print("Defining solvers...")

beam_solver = LinearNewmarkUQSolver(beam_structure, neumann_BC=neumann_BC, initial_conditions=initial_conditions)

beta1 = 1.0 / 2
beta2 = 1.0 / 2

print("Running solvers...")

solver_subtime_start = time.time()

beam_solver.run(beta1=beta1, beta2=beta2, t0=t0, dt=dt, n_timesteps=n_timesteps, n_modes=n_modes,
                n_samples=n_samples, uncertainty_type=uncertainty_type, add_deterministic=True,
                verbose=False)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

file_name_U = "plot_U_rand_linear_newmark_UQ_generalized_example"
confidence_level = 0.95

plot_array_U_rand_observed(file_name_U, beam_solver, confidence_level,
                           x_name="Time (s)", y_name="Displacement (m)", plot_type="linear",
                           add_deterministic=True)

observed_time_1 = 0.1125
num_step = round(n_timesteps * (observed_time_1 - t0) / (tmax - t0))
file_name_gkde = "plot_gkde_linear_newmark_UQ_generalized_example"

plot_array_U_rand_observed_ksdensity(file_name_gkde, beam_solver, num_step, x_name="Displacement at 0.1125 s")

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")