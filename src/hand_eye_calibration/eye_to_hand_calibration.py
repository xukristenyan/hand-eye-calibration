import os
import cv2
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from .utils.calibration import generate_random_poses, detect_marker_in_image, get_marker_detector, estimate_marker_pose, euler_to_rot
from itertools import combinations
from tabulate import tabulate


class HandEyeCalibrator:

    def __init__(self, camera, robot, offsets=None, data_dir='./data', save_images=True):
        self.camera = camera
        self.robot = robot

        self.save_dir = Path(data_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

        self.image_dir = None
        if save_images:
            self.image_dir = self.save_dir / "images"
            os.makedirs(self.image_dir, exist_ok=True)

        self.T_marker_gripper = self.get_T_marker_gripper(offsets)


    def get_T_marker_gripper(self, offsets):
        T_marker_gripper = np.eye(4)
        
        if offsets is not None:
            x, y, z, roll, pitch, yaw = offsets
            rotation = euler_to_rot(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))
            translation = np.array([x, y, z])

            T_marker_gripper[:3, :3] = rotation
            T_marker_gripper[:3, 3] = translation
        
        return T_marker_gripper


    def calibrate(self, marker_length=0.07, num_samples=15, move_range=0.25, filter=True):
        self.T_marker_gripper = self.get_T_marker_gripper()

        robot_poses, marker_poses = self.collect_marker_and_robot_poses(marker_length, num_samples, move_range)
        T_base_cam = self.get_rigid_transform(marker_poses, robot_poses, self.T_marker_gripper)
        mean_error, std_error = self.validate_calibration(T_base_cam, marker_poses, robot_poses, self.T_marker_gripper)

        headers = ["Mean Error", "Std Error", "Samples Used"]
        table_data = [[f"{mean_error * 100:.2f}", f"{std_error * 100:.2f}", 'all']] 

        print(f" Total valid poses collected: {len(robot_poses)}")
        if filter:
            print("\nRefining calibration by filtering out bad poses...")

            best_T_base_cam, best_mean_error, best_std_error, best_samples_indices = self.find_the_best_calibration(marker_poses, robot_poses, self.T_marker_gripper)
            if best_mean_error < mean_error:
                T_base_cam = best_T_base_cam

            table_data.append([f"{best_mean_error * 100:.2f}", f"{best_std_error * 100:.2f}", ', '.join([str(idx) for idx in best_samples_indices])])

        print(tabulate(table_data, headers, tablefmt="fancy_grid"))
        print("Unit: cm")

        np.save(self.save_dir / f"T_{str(self.camera.serial)}.npy", T_base_cam)

        return T_base_cam


    def find_the_best_calibration(self, marker_poses, robot_poses, T_marker_gripper, num_keep=None):
        num_samples = len(robot_poses)
        all_indices = list(range(num_samples))

        MIN_SAMPLES_REQUIRED = 4 

        if num_samples < MIN_SAMPLES_REQUIRED:
            print(f"[ERROR] Not enough samples ({num_samples}) to run calibration."
                  f"Need at least {MIN_SAMPLES_REQUIRED}."
                  f"Please adjust the initial pose and restart.")
            return None, float('inf'), float('inf'), None

        max_subset_size = num_keep if num_keep is not None else num_samples
        max_subset_size = max(MIN_SAMPLES_REQUIRED, min(max_subset_size, num_samples))

        best_mean_error, best_std_error = float('inf'), float('inf')
        best_T_base_cam = None
        best_samples_indices = None
        
        print(f"[filter] Checking subset sizes from {MIN_SAMPLES_REQUIRED} to {max_subset_size}...")
        
        for i in range(MIN_SAMPLES_REQUIRED, max_subset_size + 1):

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


    def collect_marker_and_robot_poses(self, marker_length, num_samples=15, move_range=0.25):
        start_pose = self.robot.get_current_robot_pose()
        target_poses = generate_random_poses(start_pose, num_samples, move_range)

        detector = get_marker_detector()
        intrin_matrix, dist_coeffs = self.camera.get_camera_intrinsics()

        robot_poses = []
        marker_poses = []

        for i, target_pose in enumerate(target_poses):
            print(f"[{i+1}/{num_samples}] Moving to target pose: {target_pose}")

            finished = self.robot.move_robot_to_pose(target_pose)
            
            if finished:
                time.sleep(3)

            # get marker pose
            color_image = self.camera.get_camera_image()
            image, corners, ids = detect_marker_in_image(color_image, detector)

            if ids is None or len(ids) == 0:
                print(f"[WARNING] No markers detected in image at pose {i+1}. Skipping this sample.")
                continue

            rvec, tvec = estimate_marker_pose(corners[0], marker_length, intrin_matrix, dist_coeffs)

            T_marker_cam = self.get_T_marker_camera(rvec, tvec)

            # get robot pose
            cur_pose = self.robot.get_current_robot_pose()
            robot_pose_rad = np.deg2rad(cur_pose[3:])
            robot_pose = np.concatenate((cur_pose[:3], robot_pose_rad))
            T_ee_robot = self.get_T_ee_robot(robot_pose)

            # record pose
            robot_poses.append(T_ee_robot)
            marker_poses.append(T_marker_cam)

            # save image
            if self.image_dir is not None:
                cv2.imwrite(f"{self.image_dir}/image_{i}.png", image)

        return robot_poses, marker_poses


    def get_rigid_transform(self, marker_poses, robot_poses, T_marker_gripper):
        
        assert len(robot_poses) == len(marker_poses), "Number of poses must match."
        assert len(robot_poses) >= 3, "Calibration requires at least 3 poses."

        T_gripper_marker = np.linalg.inv(T_marker_gripper)

        T_cam_marker_list = np.stack(marker_poses, axis=0)
        T_base_gripper_list = np.stack(robot_poses, axis=0)

        # Calculate the 3D points
        P_world = (T_base_gripper_list @ T_gripper_marker)[:, :3, 3]

        Q_cam = T_cam_marker_list[:, :3, 3]

        P = P_world.T
        Q = Q_cam.T

        # --- SVD solution for X * Q = P (Kabsch algorithm) ---
        # 1. Find the centroids
        centroid_P = np.mean(P, axis=1, keepdims=True)
        centroid_Q = np.mean(Q, axis=1, keepdims=True)
        
        # 2. Center the points
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # 3. Compute covariance matrix H
        H = Q_centered @ P_centered.T
        
        # 4. SVD
        U, S, Vt = np.linalg.svd(H)
        
        # 5. Calculate rotation matrix R (T_base_cam rotation)
        R = Vt.T @ U.T

        # 6. Handle reflection case (if determinant is -1)
        if np.linalg.det(R) < 0:
            print("Warning: Reflection detected. Fixing determinant.")
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # 7. Calculate translation t (T_base_cam translation)
        t = centroid_P - R @ centroid_Q

        # 8. Compose the final transformation
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R
        T_base_cam[:3, 3] = t.flatten()

        return T_base_cam


    def validate_calibration(self, T_base_cam, marker_poses, robot_poses, T_marker_gripper):

        errors = []
        for T_base_gripper, T_cam_marker in zip(robot_poses, marker_poses):
            # ground truth
            T_base_marker_gt = T_base_gripper @ np.linalg.inv(T_marker_gripper)
            pos_gt = T_base_marker_gt[:3, 3]

            # estimated
            T_base_marker_est = T_base_cam @ T_cam_marker
            pos_est = T_base_marker_est[:3, 3]
            
            error = np.linalg.norm(pos_gt - pos_est)
            errors.append(error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        return mean_error, std_error