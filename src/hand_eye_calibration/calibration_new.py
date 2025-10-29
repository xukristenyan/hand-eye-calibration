import os
import random
from pathlib import Path
from datetime import datetime

from hand_eye_calibration.calibration import HandEyeCalibrator
from hand_eye_calibration.config import serial, camera_config
from hand_eye_calibration.robot import Kinova
from realsense_toolbox import RealSenseCamera

MIN_X = 0
MIN_Y = 0
MIN_Z = 0
MAX_X = 10
MAX_Y = 10
MAX_Z = 10
THETA_X = 0
THETA_Y = 0
THETA_Z = 0

class HandEyeCalibratorNew():
    def __init__(self, camera, robot, marker_dist, data_dir='./data', save_images=True, num_poses = 10):
        self.camera = camera
        self.robot = robot
        self.marker_dist = marker_dist
        self.num_poses = num_poses

        # set up data saving
        self.save_dir = Path(data_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

        self.image_dir = None
        if save_images:
            self.image_dir = self.save_dir / "images"
            os.makedirs(self.image_dir, exist_ok=True)

        # for each point that is visited, record coordinates from camera and robot perspective
        self.camera_coordinates = []
        self.robot_coordinates = []
        
    def move_robot_to_pose(self, pose):
        finished = self.robot.reach_pose(pose, "go_to_pose")
        return finished

    def get_current_robot_pose(self):
        pose = self.robot.get_current_state()
        return pose.data

    def get_camera_image(self):
        color_image, _ = self.camera.get_images()
        return color_image

    def get_camera_intrinsics(self):
        intrinsics = self.camera.intrinsics
        return intrinsics["matrix"], intrinsics["coeffs"]

    def calibrate(self):
        poses = self.generate_poses()
        for pose in poses:
            self.move_robot_to_pose(pose) # TODO: figure out the proper input
            robot_pose = self.get_current_robot_pose()

    def generate_poses(self):
        poses = []
        for _ in range(self.num_poses):
            poses.append([
                random.uniform(MIN_X, MAX_X),
                random.uniform(MIN_Y, MAX_Y),
                random.uniform(MIN_Z, MAX_Z),
                THETA_X,
                THETA_Y,
                THETA_Z
            ])

        return poses

    def save_transform(self):
        pass

    def get_robot_coordinate(self):
        robot_pose = self.get_current_robot_pose()
        # TODO: use self.marker_dist to compute the center of the marker
    
    def get_camera_coordinate(self):
        pass

if __name__ == "__main__":
    specs = camera_config.get("specifications")

    camera = RealSenseCamera(serial, specs)
    camera.launch()

    robot = Kinova(10, 1)
    robot.launch()
    a = HandEyeCalibratorNew(camera, robot, 7)
    print(a.get_current_robot_pose())
