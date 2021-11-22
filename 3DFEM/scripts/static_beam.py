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
# Static analysis of a beam
##############################################################################

computation_time_start = time.time()

####
# material
####

print("Defining materials...")

rho = 7850
Y = 210e9
nu = 0.29
id_number = 1

material1 = materials.LinearIsotropicElasticMaterial(rho, Y, nu, id_number)

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

# Dirichlet conditions

ls_nodes_dir_1 = fun.find_nodes_in_yzplane(beam_mesh.get_points(), 0)
ls_nodes_dir_2 = fun.find_nodes_in_yzplane(beam_mesh.get_points(), L_x)

ls_nodes_dir_x = ls_nodes_dir_1 + ls_nodes_dir_2
ls_nodes_dir_y = ls_nodes_dir_1 + ls_nodes_dir_2
ls_nodes_dir_z = ls_nodes_dir_1 + ls_nodes_dir_2

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

####
# force
####

print("Defining forces...")

beam_force = force.Force(beam_mesh)

force_coords = np.array([L_x/2, L_y/2, L_z])
ls_nodes_force = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords)

nodal_force_vector = np.array([0, 0, -5e7])
beam_force.add_nodal_forces_t0(ls_nodes_force, nodal_force_vector)

####
# solver
####

print("Defining solver...")

beam_solver = solver.Solver(beam_structure, beam_force)

print("Running solver...")

solver_subtime_start = time.time()

beam_solver.linear_static_solver()

solver_subtime_end = time.time()

print("Solver sub-time: ", np.round_(solver_subtime_end - solver_subtime_start, 3), "seconds.")

####
# post-processing
####

print("Post-processing...")

vec_U = beam_solver.get_vec_U()

file_name = "./image_static_beam"
scale = 1e0
fun.export_U_on_mesh(file_name, beam_solver, vec_U, scale)

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")