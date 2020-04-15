import torch
import numpy as np
import open3d as o3d
from utils.sphere_triangles import generate as generate_mesh


def generate_points_from_uniform_distribution(size, low=-1, high=1, normalize=False):
    while True:
        points = torch.zeros([size[0] * 3, *size[1:]]).uniform_(low, high)
        points = points[torch.norm(points, dim=1) < 1]
        if points.shape[0] >= size[0]:
            return points[:size[0]]


def generate_points_from_mesh(method, depth, number_of_points):
    mesh_points, triangulation = generate_mesh(method, depth)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_points)
    mesh.triangles = o3d.utility.Vector3iVector(triangulation.triangles)
    points = torch.tensor(mesh.sample_points_uniformly(number_of_points=number_of_points).points).float()

    return points


def generate_points(config, epoch, size, normalize_points=None):
    if normalize_points is None:
        normalize_points = config['target_network_input']['normalization']['enable']
    generate = generate_points_from_uniform_distribution

    if normalize_points and config['target_network_input']['normalization']['type'] == 'progressive':
        normalization_max_epoch = config['target_network_input']['normalization']['epoch']
        mesh = config['target_network_input']['normalization']['mesh']['enable']
        mesh_start_epoch = max(config['target_network_input']['normalization']['mesh']['epoch'],
                               normalization_max_epoch + 1)
        if mesh and epoch >= mesh_start_epoch:
            points = generate_points_from_mesh(
                method=config['target_network_input']['normalization']['mesh']['method'],
                depth=config['target_network_input']['normalization']['mesh']['depth'],
                number_of_points=size[0])
        else:
            normalization_coef = np.linspace(0, 1, normalization_max_epoch)[epoch - 1] \
                if epoch <= normalization_max_epoch else 1
            points = generate(size=size, normalize=False)
            points[np.linalg.norm(points, axis=1) < normalization_coef] = \
                normalization_coef * (
                        points[
                            np.linalg.norm(points, axis=1) < normalization_coef].T /
                        torch.from_numpy(
                            np.linalg.norm(points[np.linalg.norm(points, axis=1) < normalization_coef], axis=1)).float()
                ).T
    else:
        points = generate(size=size, normalize=normalize_points)

    return points
