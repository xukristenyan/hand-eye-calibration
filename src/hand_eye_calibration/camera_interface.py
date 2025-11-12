from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):
    '''
    An abstract interface for all cameras to use the hand-eye calibration tool.
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def launch(self):
        '''Connects to the camera hardware.'''
        pass

    @abstractmethod
    def get_camera_image(self) -> np.ndarray:
        '''Fetches the latest RGB image.'''
        pass

    @abstractmethod
    def get_camera_intrinsics(self):
        '''Returns the camera's intrinsic parameters.'''
        pass

    @abstractmethod
    def shutdown(self):
        '''Disconnects from the camera.'''
        pass



# ==================================================================
# TODO: update with your robot's and camera's API
    def move_robot_to_pose(self, pose):
        finished = self.robot.reach_pose(pose, "go_to_pose")
        return finished


    def get_current_robot_pose(self):
        pose = self.robot.get_current_state()
        return pose.data


    # def get_camera_image(self):         # realsense
    #     color_image, _ = self.camera.get_images()
    #     return color_image


    # def get_camera_intrinsics(self):         # realsense
    #     intrinsics = self.camera.intrinsics
    #     return intrinsics["matrix"], intrinsics["coeffs"]


    def get_camera_image(self):         # zed
        color_image, _ = self.camera.get_rgbd()
        return color_image


    def get_camera_intrinsics(self):         # zed
        K, dist = self.camera.get_intrinsics()
        return K, dist


    def get_T_marker_gripper(self):
        T_marker_gripper = np.eye(4)
        
        # 1. Set the physical rotation
        # (Whatever you determine is correct for your setup)
        rotation = euler_to_rot(np.deg2rad(90.0), np.deg2rad(0.0), np.deg2rad(0.0))
        
        # 2. Set the physical translation (x, y, z)
        # (Measured from marker center to gripper TCP)
        translation = np.array([
            0.0,    # x-offset
            0.16,   # y-offset (5cm "up")
            0.02   # z-offset (2cm "behind")
        ])

        T_marker_gripper[:3, :3] = rotation
        T_marker_gripper[:3, 3] = translation
        
        return T_marker_gripper


    def reset_robot_to_home(self):
        self.robot.go_home()