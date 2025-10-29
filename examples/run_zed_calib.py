from hand_eye_calibration import HandEyeCalibrator
from hand_eye_calibration.robot import Kinova
import traceback
from zed_toolbox import ZedCamera


def main():
    # ===== YOUR CHANGES =====
    serial = 24944966

    # see readme for full configurations.
    camera_config = {"fps": 30}
    # ========================
    camera = None
    try:
        camera = ZedCamera(serial, camera_config)
        camera.launch()

        robot = Kinova(10, 1)
        robot.launch()
        if robot.connected:
            print("Connection successful")


        calibrator = HandEyeCalibrator(camera, robot, data_dir='./data', save_images=True)

        T_base_cam = calibrator.calibrate(marker_length=0.07, num_samples=15, filter=True)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        traceback.print_exc()
    
    finally:
        if camera:
            camera.shutdown()


if __name__ == "__main__":
    main()
