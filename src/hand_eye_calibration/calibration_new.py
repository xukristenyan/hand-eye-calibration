from hand_eye_calibration.calibration import HandEyeCalibrator
from realsense_toolbox.camera import Camera
from hand_eye_calibration.config import serial, camera_config
from hand_eye_calibration.robot import Kinova


class HandEyeCalibratorNew(HandEyeCalibrator):
    def __init__(self):
        camera = Camera(serial, camera_config)
        robot = Kinova(10, 1)
        robot.launch()
        if robot.connected:
            print("Connection successful")

        super().__init__(camera, robot)


if __name__ == "__main__":
    a = HandEyeCalibratorNew()
    print(a.get_current_robot_pose())