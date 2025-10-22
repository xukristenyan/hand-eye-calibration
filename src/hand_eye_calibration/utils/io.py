import os
import open3d as o3d
import numpy as np


def load_pcd(file_path: str, visualize=False):
    '''
    Load a point cloud from a .pcd file.
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Point cloud file not found at path: {file_path}")

    pcd = o3d.io.read_point_cloud(file_path)

    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="PointCloud Viewer")

    return pcd


def load_npy(file_path: str, allow_pickle=False):
    '''
    Load from a .npy file.
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at path: {file_path}")

    if allow_pickle:
        return np.load(file_path, allow_pickle=True)
    else:
        return np.load(file_path)