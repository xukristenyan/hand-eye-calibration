import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


def generate_random_poses(start_pose, num_samples=15):

    reference_pos, reference_euler = start_pose[:3], start_pose[3:]

    np.random.seed(42)
    position_offsets = np.random.uniform(low=-0.05, high=0.05, size=(num_samples, 3))
    euler_offsets = np.random.uniform(low=-0.1, high=0.1, size=(num_samples, 3))

    poses = []
    for i in range(num_samples):
        target_pos = reference_pos + position_offsets[i]

        # raw_euler = reference_euler + euler_offsets[i]
        # target_quat = R.from_euler('xyz', raw_euler).as_quat()
        # if target_quat[3] < 0:
        #     np.negative(target_quat, out=target_quat)
        # target_euler = R.from_quat(target_quat).as_euler('xyz')
        target_euler = reference_euler + euler_offsets[i]

        poses.append(np.concatenate((target_pos, target_euler)))

    return poses


def detect_marker_in_image(image, detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners, ids, _ = detector.detectMarkers(thresholded)

    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    if ids is not None:
        return image, corners, ids
    else:
        return image, None, None


def get_marker_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    return detector


def estimate_marker_pose(corners, marker_length, camera_matrix, dist_coeffs):       # new w/o using cv2 func
    # define 3D marker corners in marker frame
    obj_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    # solvePnP expects shape (N,1,3) and (N,1,2)
    obj_points = obj_points.reshape((4, 1, 3))
    img_points = corners.reshape((4, 1, 2))

    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    return rvec, tvec.squeeze()


# def estimate_marker_pose(corners, marker_length, camera_matrix, dist_coeffs):         # old cv2 func
#     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

#     return rvecs[0], tvecs[0]


def euler_to_rot(roll, pitch, yaw) -> np.ndarray:
    '''
    Assumes x-y-z extrinsic Tait-Bryan angles.
    Returns 3x3 rotation matrix (from base to gripper).
    '''
    Rx = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0, 0, 1]])

    return Rz @ Ry @ Rx




def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC
    return key == 27