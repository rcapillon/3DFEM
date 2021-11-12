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

import functions as fun

import importlib.util
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
# Frequency-domain analysis of a plate
##############################################################################

computation_time_start = time.time()

####
# freqsteps
####

print("Defining frequency band of analysis and sampling rate...")

f0 = 5000
fmax = 10000
n_freqsteps = 5000

vec_freq = np.linspace(f0, fmax, n_freqsteps)

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
L_y = 1e0
L_z = 2e-1

# mesh

Nn_x = 41
Nn_y = 41
Nn_z = 10

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

observed_node_coords = np.array([20*L_x/(Nn_x - 1), 20*L_y/(Nn_y - 1), L_z])
observed_node_number = fun.find_nodes_with_coordinates(plate_mesh.get_points(), observed_node_coords)[0]
observed_dof_number = observed_node_number * 3 + 2

ls_dofs_observed = [observed_dof_number]

plate_mesh.set_observed_dofs(ls_dofs_observed)

# Dirichlet conditions

ls_nodes_dir_1 = fun.find_nodes_in_xyplane(plate_mesh.get_points(), 0)
ls_nodes_dir_2 = fun.find_nodes_in_yzplane(plate_mesh.get_points(), 0)
ls_nodes_dir_3 = fun.find_nodes_in_yzplane(plate_mesh.get_points(), L_x)
ls_nodes_dir_4 = fun.find_nodes_in_xzplane(plate_mesh.get_points(), 0)
ls_nodes_dir_5 = fun.find_nodes_in_xzplane(plate_mesh.get_points(), L_y)

ls_nodes_dir_x = ls_nodes_dir_2 + ls_nodes_dir_3
ls_nodes_dir_y = ls_nodes_dir_4 + ls_nodes_dir_5
ls_nodes_dir_z = ls_nodes_dir_1

ls_dofs_dir_x = [node * 3 for node in ls_nodes_dir_x]
ls_dofs_dir_y = [node * 3 + 1 for node in ls_nodes_dir_y]
ls_dofs_dir_z = [node * 3 + 2 for node in ls_nodes_dir_z]

ls_dofs_dir = np.unique(ls_dofs_dir_z + ls_dofs_dir_x + ls_dofs_dir_y)
plate_mesh.set_dirichlet(ls_dofs_dir)

####
# structure
####

print("Defining structure...")

plate_structure = structure.Structure(plate_mesh)

alphaM = 0
alphaK = 1e-7
plate_structure.set_rayleigh(alphaM, alphaK)

####
# force
####

print("Defining forces...")

plate_force = force.Force(plate_mesh)

force_coords = np.array([L_x/2, L_y/2, L_z])
ls_nodes_force = fun.find_nodes_with_coordinates(plate_mesh.get_points(), force_coords)

nodal_force_vector = np.array([0, 0, -1e0])
plate_force.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

plate_force.compute_F0()

####
# solver
####

print("Defining solver...")

plate_solver = solver.Solver(plate_structure, plate_force)

n_modes = 300

print("Running solver...")

solver_subtime_start = time.time()

plate_solver.linear_frequency_solver(vec_freq, n_modes, verbose=False)

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

file_name_frf = "./freqdomain_plate_frf"
fun.plot_observed_U(file_name_frf, plate_solver, x_name="frequency (Hz)", y_name="Displacement amplitude (m)", plot_type="semilogy")

observed_freq_index = 4280
vec_U_step = plate_solver.get_vec_absU_step(observed_freq_index)

file_name_deformedmesh = "./freqdomain_plate_deformedmesh"
scale = 1e9
fun.export_U_on_mesh(file_name_deformedmesh, plate_mesh, vec_U_step, scale)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")