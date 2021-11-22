##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM code available at:                   #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import time
import numpy as np

import importlib.util
spec0 = importlib.util.spec_from_file_location("functions", "../functions/functions.py")
fun = importlib.util.module_from_spec(spec0)
spec0.loader.exec_module(fun)

spec1 = importlib.util.spec_from_file_location("materials", "../materials/materials.py")
materials = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(materials)

spec2 = importlib.util.spec_from_file_location("mesh", "../mesh/mesh.py")
mesh = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mesh)

spec3 = importlib.util.spec_from_file_location("force", "../force/force.py")
force = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(force)

spec4 = importlib.util.spec_from_file_location("structure", "../structure/structure.py")
structure = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(structure)

spec5 = importlib.util.spec_from_file_location("solver", "../solver/solver.py")
solver = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(solver)

##############################################################################
# Parametric stochastic frequency-domain analysis of a plate
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

n_samples = 2000

# parametric dispersion parameters

# delta_rho = 1 / np.sqrt(2)
# delta_Y = 1 / np.sqrt(2)
delta_rho = 0.05
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
t_0 = 0.0
t_max = 2e-1

vec_t = np.linspace(t_0, t_max, n_timesteps)

####
# material
####

print("Defining materials...")

rho = 7850
Y = 210e9
nu = 0.29
id_number = 1

material1 = materials.LinearIsotropicElasticMaterial(rho, Y, nu, id_number)
material1.set_dispersion_coefficient_rho(delta_rho)
material1.set_dispersion_coefficient_Y(delta_Y)
material1.set_n_samples(n_samples)

####
# geometry and mesh
####

print("Defining geometry, mesh and Dirichlet conditions...")

# geometry

L_x = 1e0
L_y = 1e-1
L_z = 1e-1

# mesh

Nn_x = 41
Nn_y = 5
Nn_z = 5

line_x = np.linspace(0, L_x, Nn_x)
line_y = np.linspace(0, L_y, Nn_y)
line_z = np.linspace(0, L_z, Nn_z)

points = np.zeros((Nn_x * Nn_y * Nn_z, 3))
points[:,0] = np.tile(line_x, Nn_y * Nn_z)

for kk in range(Nn_z):
    ind_start_kk = kk * Nn_x * Nn_y
    ind_end_kk = ind_start_kk + Nn_x * Nn_y
    points[ind_start_kk:ind_end_kk, 2] = np.repeat(line_z[kk], Nn_x * Nn_y)
    for jj in range(Nn_y):
        ind_start_jj = jj * Nn_x + ind_start_kk
        ind_end_jj = ind_start_jj + Nn_x
        points[ind_start_jj:ind_end_jj, 1] = np.repeat(line_y[jj], Nn_x)

beam_mesh = mesh.Mesh(points)
beam_mesh.delaunay3D_from_points(material1)

# Observed DOFs

observed_node1_coords = np.array([L_x/2, L_y, L_z/2])
observed_node1_number = fun.find_nodes_with_coordinates(beam_mesh.get_points(), observed_node1_coords)[0]
observed_dof1_number = observed_node1_number * 3
observed_dof2_number = observed_node1_number * 3 + 1
observed_dof3_number = observed_node1_number * 3 + 2

observed_node2_coords = np.array([L_x, 3*L_y/4, L_z])
observed_node2_number = fun.find_nodes_with_coordinates(beam_mesh.get_points(), observed_node2_coords)[0]
observed_dof4_number = observed_node2_number * 3
observed_dof5_number = observed_node2_number * 3 + 1
observed_dof6_number = observed_node2_number * 3 + 2

ls_dofs_observed = [observed_dof1_number, observed_dof2_number, observed_dof3_number, observed_dof4_number, observed_dof5_number, observed_dof6_number]

beam_mesh.set_observed_dofs(ls_dofs_observed)

# Dirichlet conditions

ls_nodes_dir_1 = fun.find_nodes_in_yzplane(beam_mesh.get_points(), 0)

ls_nodes_dir_x = ls_nodes_dir_1
ls_nodes_dir_y = ls_nodes_dir_1
ls_nodes_dir_z = ls_nodes_dir_1

ls_dofs_dir_x = [node * 3 for node in ls_nodes_dir_x]
ls_dofs_dir_y = [node * 3 + 1 for node in ls_nodes_dir_y]
ls_dofs_dir_z = [node * 3 + 2 for node in ls_nodes_dir_z]

ls_dofs_dir = np.unique(ls_dofs_dir_x + ls_dofs_dir_y + ls_dofs_dir_z)
beam_mesh.set_dirichlet(ls_dofs_dir)

####
# structure
####

print("Defining structure...")

beam_structure = structure.Structure(beam_mesh)

alphaM = 0
alphaK = 2e-4
beam_structure.set_rayleigh(alphaM, alphaK)

beam_structure.set_U0L()
beam_structure.set_V0L()
beam_structure.set_A0L()

beam_structure.set_dispersion_coefficient_M(delta_M)
beam_structure.set_dispersion_coefficient_K(delta_K)

beam_structure.set_n_samples(n_samples)

####
# force
####

print("Defining forces...")

beam_force = force.Force(beam_mesh)

force_coords = np.array([L_x, L_y/2, L_z/2])
ls_nodes_force = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords)

nodal_force_vector = np.array([0, 0, 2e6])
beam_force.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

t_1 = t_max/8
t_2 = t_max/4
vec_t = np.linspace(t_0, t_max, n_timesteps)

vec_variation = np.zeros((n_timesteps,))
vec_variation[vec_t <= t_1] = (1 / t_1) * vec_t[vec_t <= t_1]
vec_variation[vec_t > t_1] = 1.0
vec_variation[vec_t > t_2] = 0.0

beam_force.set_F_variation(vec_variation)

####
# reduced-order model
####

print("Defining reduced-order model...")

n_modes = 10

####
# solver
####

print("Defining solver...")

beam_solver = solver.Solver(beam_structure, beam_force)

beta1 = 1.0/2
beta2 = 1.0/2

print("Running solver...")

solver_subtime_start = time.time()

beam_solver.linear_newmark_solver_UQ(beta1, beta2, vec_t, n_modes,\
                                        uncertainty_type="parametric", add_deterministic=True, verbose=False)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

file_name_U = "./UQ_parametric_time_beam_U_DOF"
confidence_level = 0.95

fun.plot_random_observed_U(file_name_U, beam_solver, confidence_level,\
                           x_name="Time (s)", y_name="Displacement (m)", plot_type="linear",\
                           add_deterministic=True)
    
observed_time_1 = 0.1125
num_step = round(n_timesteps * (observed_time_1 - t_0) / (t_max - t_0))
file_name_gkde = "./UQ_parametric_time_beam_gkde_DOF"

fun.plot_ksdensity_random_observed_U(file_name_gkde, beam_solver, num_step, x_name="Displacement at 0.1 s")

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")