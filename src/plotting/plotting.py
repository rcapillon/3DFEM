import numpy as np
import copy
import os
import imageio
import matplotlib.pyplot as plt
from scipy import stats

from meshing.export import export_mesh_to_vtk


def add_U_to_points(input_mesh, U):
    output_mesh = copy.deepcopy(input_mesh)

    output_mesh.nodes[:, 0] += U[::3]
    output_mesh.nodes[:, 1] += U[1::3]
    output_mesh.nodes[:, 2] += U[2::3]

    return output_mesh


def vtk_mode_animation(file_name, solver, index_mode, scale, n_frames):
    mesh = solver.structure.mesh

    mode = solver.mat_modes[:, index_mode]

    n_points = mesh.n_nodes

    n_faces = 0
    n_cols = 0

    for element in mesh.elements:
        n_faces += len(element.faces)
        for face in element.faces:
            n_cols += 1 + len(face)

    for ii in range(n_frames):
        oscillation = np.sin(2 * np.pi * ii / n_frames)
        deformed_mesh = add_U_to_points(mesh, scale * oscillation * mode)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh, n_points, n_faces, n_cols)


def vtk_undeformed_mesh(file_name, mesh):
    n_points = mesh.n_nodes

    n_faces = 0
    n_cols = 0

    for element in mesh.elements:
        n_faces += len(element.faces)
        for face in element.faces:
            n_cols += 1 + len(face)

    export_mesh_to_vtk(file_name, mesh, n_points, n_faces, n_cols)


def vtk_U_on_mesh(file_name, solver, vec_U, scale):
    mesh = solver.structure.mesh

    n_points = mesh.n_nodes

    n_faces = 0
    n_cols = 0

    for element in mesh.elements:
        n_faces += len(element.faces)
        for face in element.faces:
            n_cols += 1 + len(face)

    deformed_mesh = add_U_to_points(mesh, scale * vec_U)
    export_mesh_to_vtk(file_name, deformed_mesh, n_points, n_faces, n_cols)


def vtk_mesh_U_animation(file_name, solver, scale):
    n_frames = len(solver.x_axis)

    n_points = solver.structure.mesh.n_nodes

    n_faces = 0
    n_cols = 0

    for element in solver.structure.mesh.elements:
        n_faces += len(element.faces)
        for face in element.faces:
            n_cols += 1 + len(face)

    for ii in range(n_frames):
        if np.iscomplex(solver.mat_qU[0, ii]):
            U = np.abs(np.dot(solver.structure.mat_modes, solver.mat_qU[:, ii]))
        else:
            U = np.dot(solver.structure.mat_modes, solver.mat_qU[:, ii])
        deformed_mesh = add_U_to_points(solver.structure.mesh, scale * U)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh, n_points, n_faces, n_cols)


def vtk_nonlin_U_animation(file_name, solver, scale):
    n_frames = len(solver.y_axis)

    mat_U = solver.mat_U

    n_points = solver.structure.mesh.n_nodes

    n_faces = 0
    n_cols = 0

    for element in solver.structure.mesh.elements:
        n_faces += len(element.faces)
        for face in element.faces:
            n_cols += 1 + len(face)

    for ii in range(n_frames):
        U = mat_U[:, ii]
        deformed_mesh = add_U_to_points(solver.structure.mesh, scale * U)
        animation_frame_name = file_name + str(ii)
        export_mesh_to_vtk(animation_frame_name, deformed_mesh, n_points, n_faces, n_cols)


def plot_mat_U_observed(file_name, solver, x_name="", y_name="", plot_type="linear"):
    folder = "plots/"
    os.makedirs(folder, exist_ok=True)

    vec_x = solver.x_axis
    observed_dofs = solver.structure.mesh.observed_dofs
    mat_U = solver.mat_U_observed

    for ii, dof_number in enumerate(observed_dofs):
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

        ax.set(xlabel=x_name, ylabel=y_name, title="DOF " + str(dof_number))
        ax.grid()

        fig.savefig(folder + image_name + ".png")
        plt.close('all')


def plot_mat_U_observed_load_displacement(file_name, solver, x_name="", y_name="", plot_type="linear"):
    folder = "plots/"
    os.makedirs(folder, exist_ok=True)

    vec_y = solver.y_axis
    observed_dofs = solver.structure.mesh.observed_dofs
    mat_U = solver.mat_U_observed

    for ii, dof_number in enumerate(observed_dofs):
        image_name = file_name + str(dof_number)

        vec_U = mat_U[ii, :]

        fig, ax = plt.subplots()
        if plot_type == "linear":
            ax.plot(vec_U, vec_y)
        elif plot_type == "semilogy":
            ax.semilogy(vec_U, vec_y)
        elif plot_type == "semilogx":
            ax.semilogx(vec_U, vec_y)
        elif plot_type == "loglog":
            ax.loglog(vec_U, vec_y)

        ax.set(xlabel=x_name, ylabel=y_name, title="DOF " + str(dof_number))
        ax.grid()

        fig.savefig(folder + image_name + ".png")
        plt.close('all')


def plot_norm_U_load_displacement(file_name, solver, x_name="", y_name="", plot_type="linear"):
    folder = "plots/"
    os.makedirs(folder, exist_ok=True)

    vec_y = solver.y_axis
    norm_U = np.sqrt(np.sum(np.power(solver.mat_U, 2), axis=0))

    fig, ax = plt.subplots()
    if plot_type == "linear":
        ax.plot(norm_U, vec_y)
    elif plot_type == "semilogy":
        ax.semilogy(norm_U, vec_y)
    elif plot_type == "semilogx":
        ax.semilogx(norm_U, vec_y)
    elif plot_type == "loglog":
        ax.loglog(norm_U, vec_y)

    ax.set(xlabel=x_name, ylabel=y_name, title="norm of U")
    ax.grid()

    fig.savefig(folder + file_name + ".png")
    plt.close('all')


def plot_norm_U_load_displacement_animation(file_name, solver, stepsize=1, fps=10):
    folder = "plots/"
    os.makedirs(folder, exist_ok=True)

    vec_y = solver.y_axis
    full_norm_U = np.sqrt(np.sum(np.power(solver.mat_U, 2), axis=0))

    ls_laststep = [ii for ii in range(0, len(vec_y), stepsize)]

    file_names = []

    for step in ls_laststep:
        norm_U = np.sqrt(np.sum(np.power(solver.mat_U[:, :(step + 1)], 2), axis=0))

        fig, ax = plt.subplots()
        ax.plot(full_norm_U, vec_y, '-b')
        ax.plot(norm_U, vec_y[:(step + 1)], '-r')

        ax.set(xlabel="Norm of U", ylabel="Load factor", title="")
        ax.grid()

        file_name_step = folder + file_name + "_" + str(step) + ".png"
        # for ii in range(n_fps):
        file_names.append(file_name_step)

        fig.savefig(file_name_step)
        plt.close('all')

    with imageio.get_writer(folder + file_name + ".gif", mode='I', fps=fps) as writer:
        for name in file_names:
            image = imageio.imread(name)
            writer.append_data(image)

    for name in set(file_names):
        os.remove(name)


def plot_array_U_rand_observed(file_name, solver, confidence_level, x_name="", y_name="", plot_type="linear",
                               add_deterministic=False):
    folder = "plots/"
    os.makedirs(folder, exist_ok=True)

    vec_x = solver.x_axis
    observed_dofs = solver.structure.mesh.observed_dofs
    array_U = solver.array_U_rand_observed

    if add_deterministic:
        mat_U_deterministic = solver.mat_U_observed

    mat_mean_U = np.mean(array_U, axis=2)

    n_samples = array_U.shape[2]
    n_leftout_up = round(n_samples * (1 - confidence_level) / 2)
    n_leftout_down = n_leftout_up

    sorted_array_U = np.sort(array_U, axis=2)

    mat_lowerbound_U = sorted_array_U[:, :, n_leftout_down]
    mat_upperbound_U = sorted_array_U[:, :, -(1 + n_leftout_up)]

    for ii, dof_number in enumerate(observed_dofs):
        image_name = file_name + str(dof_number)

        vec_mean_U = mat_mean_U[ii, :]
        vec_lowerbound_U = mat_lowerbound_U[ii, :]
        vec_upperbound_U = mat_upperbound_U[ii, :]

        if add_deterministic:
            vec_U_deterministic = mat_U_deterministic[ii, :]

        fig, ax = plt.subplots()
        if plot_type == "linear":
            ax.plot(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.plot(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.plot(vec_x, vec_mean_U, '-b')
            if add_deterministic:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        elif plot_type == "semilogy":
            ax.semilogy(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.semilogy(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.semilogy(vec_x, vec_mean_U, '-b')
            if add_deterministic:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        elif plot_type == "semilogx":
            ax.semilogx(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.semilogx(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.semilogx(vec_x, vec_mean_U, '-b')
            if add_deterministic:
                ax.plot(vec_x, vec_U_deterministic, '-r')
        elif plot_type == "loglog":
            ax.loglog(vec_x, vec_lowerbound_U, '-k', linewidth=0.7)
            ax.loglog(vec_x, vec_upperbound_U, '-k', linewidth=0.7)
            ax.fill_between(vec_x, vec_lowerbound_U, vec_upperbound_U, color='c', alpha=0.3)
            ax.loglog(vec_x, vec_mean_U, '-b')
            if add_deterministic:
                ax.plot(vec_x, vec_U_deterministic, '-r')

        ax.set(xlabel=x_name, ylabel=y_name, title="DOF " + str(dof_number))
        ax.grid()

        fig.savefig(folder + image_name + ".png")
        plt.close('all')


def plot_array_U_rand_observed_ksdensity(file_name, solver, num_step, x_name="U"):
    folder = "plots/"
    os.makedirs(folder, exist_ok=True)

    observed_dofs = solver.structure.mesh.observed_dofs
    mat_U = np.squeeze(solver.array_U_rand_observed[:, num_step, :])

    for ii, dof_number in enumerate(observed_dofs):
        image_name = file_name + str(dof_number)

        vec_U_ii = mat_U[ii, :]
        kde_ii = stats.gaussian_kde(vec_U_ii)
        vec_u_ii = np.linspace(vec_U_ii.min(), vec_U_ii.max(), 500)
        vec_p_ii = kde_ii(vec_u_ii)

        fig, ax = plt.subplots()
        ax.plot(vec_u_ii, vec_p_ii)
        ax.set(xlabel=x_name, ylabel="Probability density function", title="DOF " + str(dof_number))
        ax.grid()

        fig.savefig(folder + image_name + ".png")
        plt.close('all')
