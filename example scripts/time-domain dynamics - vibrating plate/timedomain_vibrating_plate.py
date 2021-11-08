##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM library available at:                #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import time
import numpy as np

import classes
import functions as fun

##############################################################################
# Time-domain analysis of a vibrating plate
##############################################################################

computation_time_start = time.time()

####
# timesteps
####

print("Defining timestepping parameters...")

n_timesteps = 800
t_0 = 0.0
t_max = 0.04

####
# material
####

print("Defining materials...")

rho = 930
Y = 240e6
nu = 0.27

####
# geometry and mesh
####

print("Defining geometry, mesh and Dirichlet conditions...")

# geometry

L_x = 1e0
L_y = 1e0
L_z = 1e-1

# mesh

Nn_x = 41
Nn_y = 41
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

mesh = classes.Mesh(points)
mesh.delaunay3D_from_points(rho, Y, nu)

# Dirichlet conditions

ls_nodes_dir_1 = fun.find_nodes_in_xyplane(mesh.get_points(), 0)
ls_nodes_dir_2 = fun.find_nodes_in_yzplane(mesh.get_points(), 0)
ls_nodes_dir_3 = fun.find_nodes_in_yzplane(mesh.get_points(), L_x)
ls_nodes_dir_4 = fun.find_nodes_in_xzplane(mesh.get_points(), 0)
ls_nodes_dir_5 = fun.find_nodes_in_xzplane(mesh.get_points(), L_y)

ls_nodes_dir_z = ls_nodes_dir_1
ls_nodes_dir_x = ls_nodes_dir_2 + ls_nodes_dir_3
ls_nodes_dir_y = ls_nodes_dir_4 + ls_nodes_dir_5

ls_dofs_dir_z = [node * 3 + 2 for node in ls_nodes_dir_z]
ls_dofs_dir_x = [node * 3 for node in ls_nodes_dir_x]
ls_dofs_dir_y = [node * 3 + 1 for node in ls_nodes_dir_y]

ls_dofs_dir = np.unique(ls_dofs_dir_z + ls_dofs_dir_x + ls_dofs_dir_y)
mesh.set_dirichlet(ls_dofs_dir)

####
# structure
####

print("Defining structure...")

structure = classes.Structure(mesh)

alphaM = 1e-6
alphaK = 1e-6
structure.set_rayleigh(alphaM, alphaK)

structure.set_U0L()
structure.set_V0L()
structure.set_A0L()

####
# force
####

print("Defining forces...")

force = classes.Force(mesh)

force_coords = np.array([L_x/2, L_y/2, L_z])
ls_nodes_force = fun.find_nodes_with_coordinates(mesh.get_points(), force_coords)

nodal_force_vector = np.array([0, 0, -1e4])
force.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

vec_variation = np.zeros((n_timesteps,))

t_prime = 2e-2
tau = 1e-2

vec_t = np.linspace(t_0, t_max, n_timesteps)
vec_variation[:] = np.exp(-np.power(vec_t - np.repeat(t_prime, n_timesteps), 2) / tau)

force.set_F_variation(vec_variation)

force.compute_F0()
force.compute_varying_F()

####
# solver
####

print("Defining solver...")

solver = classes.Solver(structure, force)

beta1 = 1.0/2
beta2 = 1.0/2

n_modes = 300

solver.linear_newmark_solver(beta1, beta2, t_0, t_max, n_timesteps, n_modes, verbose=False)

####
# post-processing
####

print("Post-processing...")

mat_U = solver.get_mat_U()

file_name = "./animations/animation_newmark_plaque/animation_newmark_plaque"
scale = 2e3
fun.export_U_newmark_animation(file_name, mesh, mat_U, scale)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print(np.round_(computation_time_end - computation_time_start, 3), 'sec elapsed')