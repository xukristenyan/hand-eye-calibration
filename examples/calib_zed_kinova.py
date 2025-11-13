from hand_eye_calibration import HandEyeCalibrator
from hand_eye_calibration.camera_zed import Zed
from hand_eye_calibration.robot_kinova import KinovaRobot
import traceback


def main():
    # ===== YOUR CHANGES =====
    serial = 24944966
    specs = {
        "fps": 30,
        "auto_exposure": False
        }
    camera = Zed(serial, specs)

    robot = KinovaRobot(10, 1)

    # TODO: pass in the offsets between the marker and the robot if there's any
    # first three are translation offsets in meter (Measured from marker center to gripper TCP): 
    #   x-offset, 
    #   y-offset (16cm "up")
    #   z-offset (2cm "behind")
    # last three are rotation offsets in degree:
    #   physical rotation (whatever you determine is correct for your setup)
    # default would be all zero
    offsets = [0.0, 0.16, 0.02, 90.0, 0.0, 0.0]

    # TODO: the length of your marker attached on the gripper in meter
    marker_length = 0.07

    # TODO: num of poses to move the robot to
    num_samples = 15
    # ========================

    try:
        camera.launch()
        robot.launch()

        calibrator = HandEyeCalibrator(camera, robot,  offsets=offsets, data_dir='./data', save_images=True)
        T_base_cam = calibrator.calibrate(marker_length=marker_length, num_samples=num_samples, move_range=0.25, filter=True)    # set filter to False if num_samples is large (>30)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        traceback.print_exc()
    
    finally:
        camera.shutdown()
        robot.shutdown()        



if __name__ == "__main__":
    main()
