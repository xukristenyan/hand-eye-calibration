from hand_eye_calibration import HandEyeCalibrator
from realsense_toolbox import RealSenseCamera
from realsense_toolbox import PointCloudGenerator
from hand_eye_calibration.robot import Kinova
import traceback

def main():
    # ===== YOUR CHANGES =====
    serial = "244622072715"
    # 244622072715
    # 346522075401

    # see readme for full configurations.
    specs = {
            "fps": 30,
            "color_auto_exposure": False,
            "depth_auto_exposure": False,
        }
    # ========================
    camera = None
    try:
        camera = RealSenseCamera(serial, specs)
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
