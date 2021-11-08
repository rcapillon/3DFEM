##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM library available at:                #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import numpy as np
import copy

def find_nodes_in_yzplane(points, x):
    ls_nodes = np.where(points[:,0] == x)[0].tolist()
    
    return ls_nodes
    
def find_nodes_in_xzplane(points, y):
    ls_nodes = np.where(points[:,1] == y)[0].tolist()
    
    return ls_nodes
    
def find_nodes_in_xyplane(points, z):
    ls_nodes = np.where(points[:,2] == z)[0].tolist()
    
    return ls_nodes

def find_nodes_with_coordinates(points, coords):    
    node = np.where(((points[:,0] == coords[0]).astype(int) + (points[:,1] == coords[1]).astype(int) + (points[:,2] == coords[2]).astype(int)) == 3)[0].tolist()
    
    return node

def find_nodes_in_yzplane_within_tolerance(points, x, tol):
    array_nodes = np.where(np.abs(points[:,0] - x) <= tol)[0].tolist()
    
    return array_nodes
    
def find_nodes_in_xzplane_within_tolerance(points, y, tol):
    array_nodes = np.where(np.abs(points[:,1] - y) <= tol)[0].tolist()
    
    return array_nodes
    
def find_nodes_in_xyplane_within_tolerance(points, z, tol):
    array_nodes = np.where(np.abs(points[:,2] - z) <= tol)[0].tolist()
    
    return array_nodes

def find_nodes_with_coordinates_within_tolerance(points, coords, tol):
    node = np.where(((np.abs(points[:,0] - coords[0]) <= tol).astype(int) + (np.abs(points[:,1] - coords[1]) <= tol).astype(int) + (np.abs(points[:,2] - coords[2]) <= tol).astype(int)) == 3)[0].tolist()
    
    return node

def export_mesh_to_vtk(file_name, mesh):
    file = open(file_name + ".vtk","w")
    
    n_points = mesh.get_n_points()
    
    n_faces = 0
    n_cols = 0
    for element in mesh.get_elements_table():
        element_faces = element.get_faces()
        n_faces += len(element_faces)
        for face in element_faces:
            n_cols += 1 + len(face)
    
    str_beginning = "# vtk DataFile Version 1.0\n" + file_name + "\nASCII\n\nDATASET POLYDATA\nPOINTS " + str(n_points) + " float\n"
    file.write(str_beginning)
    
    for ii in range(n_points):
        point_ii = mesh.get_points()[ii,:]
        point_x = point_ii[0]
        point_y = point_ii[1]
        point_z = point_ii[2]
        
        str_points = "%.6f" % point_x + " " + "%.6f" % point_y + " " + "%.6f" % point_z + "\n"
        
        file.write(str_points)
    
    polygons = "POLYGONS " + str(n_faces) + " " + str(n_cols) + "\n"
    file.write(polygons)
    
    for element in mesh.get_elements_table():
        element_faces = element.get_faces()
        for face in element_faces:
            str_face = str(len(face))
            for node_num in face:
                str_face += " " + str(element.get_nodes_nums()[node_num])
            file.write(str_face + "\n")
    
    file.close()
    
def export_mode_animation(file_name, mesh, mode, scale, n_frames):
    
    for ii in range(n_frames):
        deformed_mesh = copy.deepcopy(mesh)
        deformed_mesh.add_U_to_points(scale * np.sin(2 * np.pi * ii / n_frames) * mode)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh)
    
def export_U_newmark_animation(file_name, mesh, mat_U, scale):
    n_frames = mat_U.shape[1]
    
    for ii in range(n_frames):
        deformed_mesh = copy.deepcopy(mesh)
        U = mat_U[:, ii]
        deformed_mesh.add_U_to_points(scale * U)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh)
