import numpy as np


def get_transform_from_params(params):
    '''
    Creates a 4x4 transformation matrix from 6-DoF parameters.
    Rotations are applied in ZYX order (extrinsic) or XYZ (intrinsic).
    '''
    tx, ty, tz = params['tx'], params['ty'], params['tz']
    rx_rad = np.radians(params['rx'])
    ry_rad = np.radians(params['ry'])
    rz_rad = np.radians(params['rz'])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ])

    Ry = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])

    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T


def get_params_from_transform(transform):
    '''
    Extracts 6-DoF parameters (ZYX Euler angles) from a 4x4 transformation matrix.
    '''
    tx, ty, tz = transform[:3, 3]
    R = transform[:3, :3]

    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        rx_rad = np.arctan2(R[2, 1], R[2, 2])
        ry_rad = np.arctan2(-R[2, 0], sy)
        rz_rad = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx_rad = np.arctan2(-R[1, 2], R[1, 1])
        ry_rad = np.arctan2(-R[2, 0], sy)
        rz_rad = 0

    params = {
        'tx': tx, 'ty': ty, 'tz': tz,
        'rx': np.degrees(rx_rad),
        'ry': np.degrees(ry_rad),
        'rz': np.degrees(rz_rad)
    }
    
    return params