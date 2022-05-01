##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM code available at:                   #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import numpy as np
import os
import imageio
import copy
import matplotlib.pyplot as plt
from scipy import stats

import importlib.util
spec1 = importlib.util.spec_from_file_location("tet4", "../elements/tet4.py")
tet4 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(tet4)


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
    nodes = np.where(((points[:,0] == coords[0]).astype(int) + (points[:,1] == coords[1]).astype(int) + (points[:,2] == coords[2]).astype(int)) == 3)[0].tolist()
    
    return nodes


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
    nodes = np.where(((np.abs(points[:,0] - coords[0]) <= tol).astype(int) + (np.abs(points[:,1] - coords[1]) <= tol).astype(int) + (np.abs(points[:,2] - coords[2]) <= tol).astype(int)) == 3)[0].tolist()
    
    return nodes


def export_mesh_to_vtk(file_name, mesh, n_points=None, n_faces=None, n_cols=None):
    file = open(file_name + ".vtk","w")
    
    if n_points is None:
        n_points = mesh.get_n_points()
    
    if n_faces is None:
        n_faces = 0
    if n_cols is None:
        n_cols = 0
    
    if n_faces is None or n_cols is None:
        for element in mesh.get_elements_list():
            element_faces = element.get_faces()
            if n_faces is None:
                n_faces += len(element_faces)
            if n_cols is None:
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
    
    for element in mesh.get_elements_list():
        element_faces = element.get_faces()
        for face in element_faces:
            str_face = str(len(face))
            for node_num in face:
                str_face += " " + str(element.get_nodes_nums()[node_num])
            file.write(str_face + "\n")
    
    file.close()


def export_mode_animation(file_name, solver, index_mode, scale, n_frames):
    mesh = solver.get_structure().get_mesh()
    
    mode = solver.get_modes()[:, index_mode]
    
    n_points = mesh.get_n_points()
    
    n_faces = 0
    n_cols = 0

    for element in mesh.get_elements_list():
        element_faces = element.get_faces()
        n_faces += len(element_faces)
        for face in element_faces:
            n_cols += 1 + len(face)
    
    for ii in range(n_frames):
        deformed_mesh = copy.deepcopy(mesh)
        deformed_mesh.add_U_to_points(scale * np.sin(2 * np.pi * ii / n_frames) * mode)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh, n_points, n_faces, n_cols)


def export_undeformed_mesh(file_name, mesh):    
    n_points = mesh.get_n_points()
    
    n_faces = 0
    n_cols = 0

    for element in mesh.get_elements_list():
        element_faces = element.get_faces()
        n_faces += len(element_faces)
        for face in element_faces:
            n_cols += 1 + len(face)
    
    export_mesh_to_vtk(file_name, mesh, n_points, n_faces, n_cols)


def export_U_on_mesh(file_name, solver, vec_U, scale):
    mesh = solver.get_structure().get_mesh()
    
    n_points = mesh.get_n_points()
    
    n_faces = 0
    n_cols = 0

    for element in mesh.get_elements_list():
        element_faces = element.get_faces()
        n_faces += len(element_faces)
        for face in element_faces:
            n_cols += 1 + len(face)
    
    deformed_mesh = copy.deepcopy(mesh)
    deformed_mesh.add_U_to_points(scale * vec_U)
    export_mesh_to_vtk(file_name, deformed_mesh, n_points, n_faces, n_cols)


def export_U_newmark_animation(file_name, solver, scale):
    n_frames = len(solver.get_x_axis())
    
    n_points = solver.get_structure().get_mesh().get_n_points()
    
    n_faces = 0
    n_cols = 0

    for element in solver.get_structure().get_mesh().get_elements_list():
        element_faces = element.get_faces()
        n_faces += len(element_faces)
        for face in element_faces:
            n_cols += 1 + len(face)
    
    for ii in range(n_frames):
        deformed_mesh = copy.deepcopy(solver.get_structure().get_mesh())
        U = solver.get_vec_U_step(ii)
        deformed_mesh.add_U_to_points(scale * U)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh, n_points, n_faces, n_cols)


def plot_observed_U(file_name, solver, x_name="", y_name="", plot_type="linear"):
    vec_x = solver.get_x_axis()
    ls_dofs_observed = solver.get_structure().get_mesh().get_observed_dofs()
    mat_U = solver.get_mat_U_observed()
    
    for ii in range(len(ls_dofs_observed)):
        dof_number = ls_dofs_observed[ii]
        image_name = file_name + str(dof_number)
        
        vec_U = mat_U[ii, :]
        
        fig, ax = plt.subplots()
        if plot_type == "linear":
            ax.plot(vec_x, vec_U)
        elif plot_type == "semilogy":
            ax.semilogy(vec_x, vec_U)
        elif plot_type == "semilogx":
            ax.semilogx(vec_x, vec_U)
        elif plot_type == "loglog":
            ax.loglog(vec_x, vec_U)
        
        ax.set(xlabel = x_name, ylabel = y_name, title="DOF " + str(dof_number))
        ax.grid()
    
        fig.savefig(image_name + ".png")
        plt.close('all')


def plot_reversed_observed_U(file_name, solver, x_name="", y_name="", plot_type="linear"):
    vec_x = solver.get_x_axis()
    ls_dofs_observed = solver.get_structure().get_mesh().get_observed_dofs()
    mat_U = solver.get_mat_U_observed()
    
    for ii in range(len(ls_dofs_observed)):
        dof_number = ls_dofs_observed[ii]
        image_name = file_name + str(dof_number)
        
        vec_U = mat_U[ii, :]
        
        fig, ax = plt.subplots()
        if plot_type == "linear":
            ax.plot(vec_U, vec_x)
        elif plot_type == "semilogy":
            ax.semilogy(vec_U, vec_x)
        elif plot_type == "semilogx":
            ax.semilogx(vec_U, vec_x)
        elif plot_type == "loglog":
            ax.loglog(vec_U, vec_x)
        
        ax.set(xlabel = x_name, ylabel = y_name, title="DOF " + str(dof_number))
        ax.grid()
    
        fig.savefig(image_name + ".png")
        plt.close('all')


def plot_reversed_norm_U(file_name, solver, x_name="", y_name="", plot_type="linear"):
    vec_x = solver.get_x_axis()
    norm_U = np.sqrt(np.sum(np.power(solver.get_mat_U_nonlin(), 2), axis=0))
                    
    fig, ax = plt.subplots()
    if plot_type == "linear":
        ax.plot(norm_U, vec_x)
    elif plot_type == "semilogy":
        ax.semilogy(norm_U, vec_x)
    elif plot_type == "semilogx":
        ax.semilogx(norm_U, vec_x)
    elif plot_type == "loglog":
        ax.loglog(norm_U, vec_x)
    
    ax.set(xlabel = x_name, ylabel = y_name, title="norm of U")
    ax.grid()

    fig.savefig(file_name + ".png")
    plt.close('all')


def plot_random_observed_U(file_name, solver, confidence_level, x_name="", y_name="", plot_type="linear", add_deterministic=False):
    vec_x = solver.get_x_axis()
    ls_dofs_observed = solver.get_structure().get_mesh().get_observed_dofs()
    array_U = solver.get_array_U_rand_observed()
    
    if add_deterministic == True:
        mat_U_deterministic = solver.get_mat_U_observed()
    
    mat_mean_U = np.mean(array_U, axis=2)
    
    n_samples = array_U.shape[2]
    n_leftout_up = round(n_samples * (1 - confidence_level) / 2)
    n_leftout_down = n_leftout_up
    
    sorted_array_U = np.sort(array_U, axis=2)
        
    mat_lowerbound_U = sorted_array_U[:, :, n_leftout_down]
    mat_upperbound_U = sorted_array_U[:, :, -(1 + n_leftout_up)]
    
    for ii in range(len(ls_dofs_observed)):
        dof_number = ls_dofs_observed[ii]
        image_name = file_name + str(dof_number)
        
        vec_mean_U = mat_mean_U[ii, :]
        vec_lowerbound_U = mat_lowerbound_U[ii, :]
        vec_upperbound_U = mat_upperbound_U[ii, :]
        
        if add_deterministic == True:
            vec_U_deterministic = mat_U_deterministic[ii, :]
        
        fig, ax = plt.subplots()
        if plot_type == "linear":
            ax.plot(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.plot(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.plot(vec_x, vec_mean_U, '-b')
            if add_deterministic == True:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        elif plot_type == "semilogy":
            ax.semilogy(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.semilogy(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.semilogy(vec_x, vec_mean_U, '-b')
            if add_deterministic == True:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        elif plot_type == "semilogx":
            ax.semilogx(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.semilogx(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.semilogx(vec_x, vec_mean_U, '-b')
            if add_deterministic == True:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        elif plot_type == "loglog":
            ax.loglog(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.loglog(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.loglog(vec_x, vec_mean_U, '-b')
            if add_deterministic == True:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        
        ax.set(xlabel = x_name, ylabel = y_name, title="DOF " + str(dof_number))
        ax.grid()
    
        fig.savefig(image_name + ".png")
        plt.close('all')


def plot_ksdensity_random_observed_U(file_name, solver, num_step, x_name="U"):
    ls_dofs_observed = solver.get_structure().get_mesh().get_observed_dofs()
    mat_U = np.squeeze(solver.get_array_U_rand_observed()[:, num_step, :])
    
    for ii in range(len(ls_dofs_observed)):
        dof_number = ls_dofs_observed[ii]
        image_name = file_name + str(dof_number)
        
        vec_U_ii = mat_U[ii, :]
        kde_ii = stats.gaussian_kde(vec_U_ii)
        vec_u_ii = np.linspace(vec_U_ii.min(), vec_U_ii.max(), 500)
        vec_p_ii = kde_ii(vec_u_ii)
        
        fig, ax = plt.subplots()
        ax.plot(vec_u_ii, vec_p_ii)
        ax.set(xlabel = x_name, ylabel = "Probability density function", title="DOF " + str(dof_number))
        ax.grid()
    
        fig.savefig(image_name + ".png")
        plt.close('all')


def export_nonlin_U_animation(file_name, solver, scale):
    n_frames = len(solver.get_x_axis())
    
    mat_U = solver.get_mat_U_nonlin()
    
    n_points = solver.get_structure().get_mesh().get_n_points()
    
    n_faces = 0
    n_cols = 0

    for element in solver.get_structure().get_mesh().get_elements_list():
        element_faces = element.get_faces()
        n_faces += len(element_faces)
        for face in element_faces:
            n_cols += 1 + len(face)
    
    for ii in range(n_frames):
        deformed_mesh = copy.deepcopy(solver.get_structure().get_mesh())
        U = mat_U[:, ii]
        deformed_mesh.add_U_to_points(scale * U)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh, n_points, n_faces, n_cols)


def plot_nonlin_U_animation(file_name, solver, stepsize=1):
    vec_x = solver.get_x_axis()
    full_norm_U = np.sqrt(np.sum(np.power(solver.get_mat_U_nonlin(), 2), axis=0))
    
    ls_laststep = [ii for ii in range(0, len(vec_x), stepsize)]    
    # n_frames = len(ls_laststep)
    # n_fps = int(np.round(n_frames / total_time))
    # if n_fps < 1:
    #     n_fps = 1
    
    file_names = []
    
    for step in ls_laststep:
        norm_U = np.sqrt(np.sum(np.power(solver.get_mat_U_nonlin()[:, :(step + 1)], 2), axis=0))
                    
        fig, ax = plt.subplots()
        ax.plot(full_norm_U, vec_x, '-b')
        ax.plot(norm_U, vec_x[:(step + 1)], '-r')
        
        ax.set(xlabel = "Norm of U", ylabel = "Load factor", title="")
        ax.grid()
                
        file_name_step = file_name + "_" + str(step) + ".png"
        # for ii in range(n_fps):
        file_names.append(file_name_step)
    
        fig.savefig(file_name_step)
        plt.close('all')
                
    with imageio.get_writer(file_name + ".gif", mode='I') as writer:
        for file_name in file_names:
            image = imageio.imread(file_name)
            writer.append_data(image)
            
    for file_name in set(file_names):
        os.remove(file_name)


def plot_observed_stress_strain(file_name, solver, components,
                                x_name="", y_name="", plot_type="linear"):
    ls_dofs_observed = solver.get_structure().get_mesh().get_observed_dofs()
    vec_U = solver.get_vec_U()
    mat_nodal_strain, mat_nodal_stress = solver.get_structure().get_mesh().compute_stress_at_nodes(vec_U,
                                                                                                   return_strain=True)
    mat_nodal_strain[:, 3:6] /= np.sqrt(2)
    mat_nodal_stress[:, 3:6] /= np.sqrt(2)

    done_nodes = []

    for dof_num in ls_dofs_observed:
        node_num = int((dof_num - dof_num % 3) / 3)

        if node_num not in done_nodes:
            done_nodes.append(node_num)

            for component in components:
                image_name = file_name + "_" + str(node_num) + "_" + str(component)

                vec_strain = np.array([0, mat_nodal_strain[node_num, component]])
                vec_stress = np.array([0, mat_nodal_stress[node_num, component]])

                fig, ax = plt.subplots()
                if plot_type == "linear":
                    ax.plot(vec_strain, vec_stress)
                elif plot_type == "semilogy":
                    ax.semilogy(vec_strain, vec_stress)
                elif plot_type == "semilogx":
                    ax.semilogx(vec_strain, vec_stress)
                elif plot_type == "loglog":
                    ax.loglog(vec_strain, vec_stress)

                ax.set(xlabel=x_name, ylabel=y_name, title="Component " + str(component))
                ax.grid()

                fig.savefig(image_name + ".png")
                plt.close('all')



