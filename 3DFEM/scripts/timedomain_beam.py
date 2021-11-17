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

spec1 = importlib.util.spec_from_file_location("mesh", "../mesh/mesh.py")
mesh = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(mesh)

spec2 = importlib.util.spec_from_file_location("force", "../force/force.py")
force = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(force)

spec3 = importlib.util.spec_from_file_location("structure", "../structure/structure.py")
structure = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(structure)

spec4 = importlib.util.spec_from_file_location("solver", "../solver/solver.py")
solver = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(solver)

##############################################################################
# Time-domain analysis of a beam
##############################################################################

computation_time_start = time.time()

####
# timesteps
####

print("Defining time interval of analysis and sampling rate...")

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

####
# geometry and mesh
####

print("Defining geometry, mesh and Dirichlet conditions...")

# geometry

L_x = 1e0
L_y = 1e-1
L_z = 1e-1

# mesh

Nn_x = 61
Nn_y = 7
Nn_z = 7

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

plate_mesh = mesh.Mesh(points)
plate_mesh.delaunay3D_from_points(rho, Y, nu)

# Observed DOFs

observed_node1_coords = np.array([L_x, L_y/2, L_z])
observed_node1_number = fun.find_nodes_with_coordinates(plate_mesh.get_points(), observed_node1_coords)[0]
observed_dof1_number = observed_node1_number * 3 + 1

observed_node2_coords = np.array([L_x, L_y/2, L_z/2])
observed_node2_number = fun.find_nodes_with_coordinates(plate_mesh.get_points(), observed_node2_coords)[0]
observed_dof2_number = observed_node2_number * 3 + 2

ls_dofs_observed = [observed_dof1_number, observed_dof2_number]

plate_mesh.set_observed_dofs(ls_dofs_observed)

# Dirichlet conditions

ls_nodes_dir_1 = fun.find_nodes_in_yzplane(plate_mesh.get_points(), 0)

ls_nodes_dir_x = ls_nodes_dir_1
ls_nodes_dir_y = ls_nodes_dir_1
ls_nodes_dir_z = ls_nodes_dir_1

ls_dofs_dir_x = [node * 3 for node in ls_nodes_dir_x]
ls_dofs_dir_y = [node * 3 + 1 for node in ls_nodes_dir_y]
ls_dofs_dir_z = [node * 3 + 2 for node in ls_nodes_dir_z]

ls_dofs_dir = np.unique(ls_dofs_dir_x + ls_dofs_dir_y + ls_dofs_dir_z)
plate_mesh.set_dirichlet(ls_dofs_dir)

####
# structure
####

print("Defining structure...")

plate_structure = structure.Structure(plate_mesh)

alphaM = 0
alphaK = 2e-4
plate_structure.set_rayleigh(alphaM, alphaK)

plate_structure.set_U0L()
plate_structure.set_V0L()
plate_structure.set_A0L()

####
# force
####

print("Defining forces...")

plate_force = force.Force(plate_mesh)

force_coords = np.array([L_x, L_y/2, L_z/2])
ls_nodes_force = fun.find_nodes_with_coordinates(plate_mesh.get_points(), force_coords)

nodal_force_vector = np.array([0, 0, 2e6])
plate_force.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

t_1 = t_max/8
t_2 = t_max/4
vec_t = np.linspace(t_0, t_max, n_timesteps)

vec_variation = np.zeros((n_timesteps,))
vec_variation[vec_t <= t_1] = (1 / t_1) * vec_t[vec_t <= t_1]
vec_variation[vec_t > t_1] = 1.0
vec_variation[vec_t > t_2] = 0.0

plate_force.set_F_variation(vec_variation)

plate_force.compute_F0()

####
# solver
####

print("Defining solver...")

plate_solver = solver.Solver(plate_structure, plate_force)

beta1 = 1.0/2
beta2 = 1.0/2

n_modes = 10

print("Running solver...")

solver_subtime_start = time.time()

plate_solver.linear_newmark_solver(beta1, beta2, vec_t, n_modes, verbose=False)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

file_name_plot = "./timedomain_beam_plot"
fun.plot_observed_U(file_name_plot, plate_solver, x_name="Time (s)", y_name="Displacement (m)", plot_type="linear")

file_name = "./animation_timedomain_beam"
scale = 1e0
fun.export_U_newmark_animation(file_name, plate_solver, scale)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")