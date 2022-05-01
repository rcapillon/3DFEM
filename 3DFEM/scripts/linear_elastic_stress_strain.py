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

L_x = 5e-1
L_y = 2e-2
L_z = 2e-2

# mesh

Nn_x = 601
Nn_y = 3
Nn_z = 3

line_x = np.linspace(0, L_x, Nn_x)
line_y = np.linspace(0, L_y, Nn_y)
line_z = np.linspace(0, L_z, Nn_z)

points = np.zeros((Nn_x * Nn_y * Nn_z, 3))
points[:, 0] = np.tile(line_x, Nn_y * Nn_z)

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

observed_node1_coords = np.array([L_x, L_y/2, L_z/2])
observed_node1_number = fun.find_nodes_with_coordinates(beam_mesh.get_points(), observed_node1_coords)[0]
observed_dof1_number = observed_node1_number * 3

ls_dofs_observed = [observed_dof1_number]

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

####
# force
####

print("Defining forces...")

beam_force = force.Force(beam_mesh)

base_force_vector = np.array([-1e0/24, 0, 0])

# corner nodes

force_coords_1 = np.array([L_x, 0, 0])
node_force_1 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_1)
beam_force.add_nodal_forces_t0(node_force_1, base_force_vector)

force_coords_2 = np.array([L_x, L_y, 0])
node_force_2 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_2)
beam_force.add_nodal_forces_t0(node_force_2, base_force_vector)

force_coords_3 = np.array([L_x, L_y, L_z])
node_force_3 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_3)
beam_force.add_nodal_forces_t0(node_force_3, 2*base_force_vector)

force_coords_4 = np.array([L_x, 0, L_z])
node_force_4 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_4)
beam_force.add_nodal_forces_t0(node_force_4, 2*base_force_vector)

# side nodes

force_coords_5 = np.array([L_x, L_y/2, 0])
node_force_5 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_5)
beam_force.add_nodal_forces_t0(node_force_5, 4*base_force_vector)

force_coords_6 = np.array([L_x, L_y, L_z/2])
node_force_6 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_6)
beam_force.add_nodal_forces_t0(node_force_6, 3*base_force_vector)

force_coords_7 = np.array([L_x, L_y/2, L_z])
node_force_7 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_7)
beam_force.add_nodal_forces_t0(node_force_7, 2*base_force_vector)

force_coords_8 = np.array([L_x, 0, L_z/2])
node_force_8 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_8)
beam_force.add_nodal_forces_t0(node_force_8, 3*base_force_vector)

# center node

force_coords_9 = np.array([L_x, L_y/2, L_z/2])
node_force_9 = fun.find_nodes_with_coordinates(beam_mesh.get_points(), force_coords_9)
beam_force.add_nodal_forces_t0(node_force_9, 6*base_force_vector)


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

file_name = "./stress_strain_linear_static_beam"
components = [0]
fun.plot_observed_stress_strain(file_name, beam_solver, components,
                                x_name="Strain (x-x component)", y_name="Stress (x-x component)", plot_type="linear")

####
# end
####

print("Computation done.")

computation_time_end = time.time()

print("Total computation time: ", np.round_(computation_time_end - computation_time_start, 3), "seconds.")