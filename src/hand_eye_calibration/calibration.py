import os
import cv2
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from .utils import generate_random_poses, detect_marker_in_image, get_marker_detector, estimate_marker_pose, euler_to_rot
from itertools import combinations
from tabulate import tabulate


class HandEyeCalibrator:

    def __init__(self, camera, robot, data_dir='./data', save_images=True):
        self.camera = camera
        self.robot = robot

        # set up data saving
        self.save_dir = Path(data_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

        self.image_dir = None
        if save_images:
            self.image_dir = self.save_dir / "images"
            os.makedirs(self.image_dir, exist_ok=True)


# ==================================================================
# TODO: update with your robot's and camera's API
    def move_robot_to_pose(self, pose):
        self.robot.reach_pose(pose, "go_to_pose")


    def get_current_robot_pose(self):
        pose = self.robot.get_current_state()
        return pose.data


    def get_camera_image(self):
        color_image, _ = self.camera.get_images()
        return color_image


    def get_camera_intrinsics(self):
        intrinsics = self.camera.get_intrinsics()
        return intrinsics["matrix"], intrinsics["coeffs"]

    def get_T_marker_gripper(self):
        # define the fixed transform from marker to gripper
        T_marker_gripper = np.eye(4)
        T_marker_gripper[:3, :3] = euler_to_rot(np.deg2rad(90.0), np.deg2rad(0.0), np.deg2rad(0.0))
        # T_marker_gripper = T_marker_gripper[None, :, :]
        return T_marker_gripper

# ==================================================================

    def calibrate(self, marker_length=0.07, num_samples=15, filter=True):
        T_marker_gripper = self.get_T_marker_gripper()

        robot_poses, marker_poses = self.collect_marker_and_robot_poses(marker_length, num_samples)
        T_base_cam = self.get_rigid_transform(marker_poses, robot_poses, T_marker_gripper)
        mean_error, std_error = self.validate_calibration(T_base_cam, marker_poses, robot_poses, T_marker_gripper)

        headers = ["Mean Error", "Std Error", "Samples Used"]
        table_data = [[f"{mean_error * 1000:.2f}", f"{std_error * 1000:.2f}", 'all']] 

        if filter:
            print("\nRefining calibration by filtering out bad poses...")
            
            best_T_base_cam, best_mean_error, best_std_error, best_samples_indices = self.find_the_best_calibration(marker_poses, robot_poses, T_marker_gripper)
            if best_mean_error < mean_error:
                T_base_cam = best_T_base_cam

            table_data.append([f"{best_mean_error * 1000:.2f}", f"{best_std_error * 1000:.2f}", ', '.join([str(idx) for idx in best_samples_indices])])

        print(tabulate(table_data, headers, tablefmt="fancy_grid"))

        np.save(self.save_dir / f"T_{self.camera.serial}.npy", T_base_cam)

        return T_base_cam


    def find_the_best_calibration(self, marker_poses, robot_poses, T_marker_gripper, num_keep):
        num_samples = len(robot_poses)
        all_indices = list(range(num_samples))
        num_keep = num_samples - 2

        best_mean_error, best_std_error = 1000, 1000
        best_T_base_cam = None
        best_samples_indices = None
    
        for i in list(range(num_keep)):
            print(f"Processing subsets of size {i}...")

            for combo in combinations(all_indices, i):
                subset_indices = list(combo)

                subset_robot_poses = [robot_poses[idx] for idx in subset_indices]
                subset_marker_poses = [marker_poses[idx] for idx in subset_indices]

                T_base_cam_subset = self.get_rigid_transform(subset_marker_poses, subset_robot_poses, T_marker_gripper)
                mean_error_subset, std_error_subset = self.validate_calibration(T_base_cam_subset, subset_marker_poses, subset_robot_poses, T_marker_gripper)

                if mean_error_subset < best_mean_error:
                    best_mean_error = mean_error_subset
                    best_std_error = std_error_subset
                    best_T_base_cam = T_base_cam_subset
                    best_samples_indices = subset_indices
        
        return best_T_base_cam, best_mean_error, best_std_error, best_samples_indices

            
    def get_T_marker_camera(self, rvec, tvec):
        R_ct, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3,:3] = R_ct
        T[:3, 3] = tvec.flatten()

        return T


    def get_T_ee_robot(self, pose) -> np.ndarray:
        '''
        pose: 6-DoF pose = [x, y, z, tx, ty, tz]
        returns 4x4 matrix
        '''
        x, y, z, roll, pitch, yaw = pose
        T = np.eye(4)
        T[:3,:3] = euler_to_rot(roll, pitch, yaw)
        T[:3, 3] = [x, y, z]

        return T

    def collect_marker_and_robot_poses(self, marker_length, num_samples=15):
        start_pose = self.get_current_robot_pose()
        target_poses = generate_random_poses(start_pose, num_samples)

        detector = get_marker_detector()
        intrin_matrix, dist_coeffs = self.get_camera_intrinsics()

        robot_poses = []
        marker_poses = []

        for i, target_pose in enumerate(target_poses):
            print(f"[{i+1}/{num_samples}] Moving to target pose: {target_pose}")

            self.move_robot_to_pose(target_pose)
            time.sleep(3)

            # get marker pose
            color_image = self.get_camera_image()
            image, corners, ids = detect_marker_in_image(color_image, detector)

            if ids is None or len(ids) == 0:
                print(f"[WARNING] No markers detected in image at pose {i+1}. Skipping this sample.")
                continue

            rvec, tvec = estimate_marker_pose(corners, marker_length, intrin_matrix, dist_coeffs)

            T_marker_cam = self.get_T_marker_camera(rvec, tvec)

            # get robot pose
            cur_pose = self.get_current_robot_pose()
            robot_pose_rad = np.deg2rad(cur_pose[3:])
            robot_pose = np.concatenate((cur_pose[:3], robot_pose_rad))
            T_ee_robot = self.get_T_ee_robot(robot_pose)

    # my code : marker_pose = np.vstack([rs_r, rs_t])

            # record pose
            robot_poses.append(T_ee_robot)
            marker_poses.append(T_marker_cam)

            # save image
            if self.image_dir is not None:
                cv2.imwrite(f"{self.image_dir}/image_{i}.png", image)

        return robot_poses, marker_poses


    def get_rigid_transform(self, marker_poses, robot_poses, T_marker_gripper):

        assert len(robot_poses) == len(marker_poses), "Number of robot poses must match number of marker poses."
        assert len(robot_poses) >= 3, "Calibration requires at least 3 different poses."

        T_cam_marker = np.stack(marker_poses, axis=0)
        T_base_gripper = np.stack(robot_poses, axis=0)

        T_cam_gripper = T_cam_marker @ T_marker_gripper
        T_gripper_cam = np.linalg.inv(T_cam_gripper)

        # keep = np.ones(len(T_gripper_cam), dtype=bool)
        # keep[np.array(drop)-1] = 0 # ignore output{i}.png # TODO: make this code cleaner

        R_cam_base, t_cam_base = cv2.calibrateHandEye(
            T_gripper_cam[:, :3, :3], 
            T_gripper_cam[:, :3, 3],
            T_base_gripper[:, :3, :3], 
            T_base_gripper[:, :3, 3],
            # method=cv2.CALIB_HAND_EYE_TSAI,
            method=cv2.CALIB_HAND_EYE_PARK,
        )
        
        # post process
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R_cam_base
        T_cam_base[:3, 3] = t_cam_base.flatten()

        T_base_cam = np.linalg.inv(T_cam_base)

        return T_base_cam


    def validate_calibration(self, T_base_cam, marker_poses, robot_poses, T_marker_gripper):

        errors = []
        for T_base_gripper, T_cam_marker in zip(robot_poses, marker_poses):
            # ground truth
            T_base_marker_gt = T_base_gripper @ T_marker_gripper
            pos_gt = T_base_marker_gt[:3, 3]

            # estimated
            T_base_marker_est = T_base_cam @ T_cam_marker
            pos_est = T_base_marker_est[:3, 3]
            
            error = np.linalg.norm(pos_gt - pos_est)
            errors.append(error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        return mean_error, std_error