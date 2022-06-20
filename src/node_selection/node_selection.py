import numpy as np


def find_nodes_in_yzplane(points, x):
    ls_nodes = np.where(points[:, 0] == x)[0].tolist()

    return ls_nodes


def find_nodes_in_xzplane(points, y):
    ls_nodes = np.where(points[:, 1] == y)[0].tolist()

    return ls_nodes


def find_nodes_in_xyplane(points, z):
    ls_nodes = np.where(points[:, 2] == z)[0].tolist()

    return ls_nodes


def find_nodes_with_coordinates(points, coords):
    nodes = np.where(((points[:, 0] == coords[0]).astype(int) + (points[:, 1] == coords[1]).astype(int) + (
                points[:, 2] == coords[2]).astype(int)) == 3)[0].tolist()

    return nodes


def find_nodes_in_yzplane_within_tolerance(points, x, tol):
    array_nodes = np.where(np.abs(points[:, 0] - x) <= tol)[0].tolist()

    return array_nodes


def find_nodes_in_xzplane_within_tolerance(points, y, tol):
    array_nodes = np.where(np.abs(points[:, 1] - y) <= tol)[0].tolist()

    return array_nodes


def find_nodes_in_xyplane_within_tolerance(points, z, tol):
    array_nodes = np.where(np.abs(points[:, 2] - z) <= tol)[0].tolist()

    return array_nodes


def find_nodes_with_coordinates_within_tolerance(points, coords, tol):
    nodes = np.where(((np.abs(points[:, 0] - coords[0]) <= tol).astype(int) + (
                np.abs(points[:, 1] - coords[1]) <= tol).astype(int) + (np.abs(points[:, 2] - coords[2]) <= tol).astype(
        int)) == 3)[0].tolist()

    return nodes
