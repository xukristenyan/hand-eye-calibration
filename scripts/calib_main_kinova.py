from hand_eye_calibration import HandEyeCalibrator
from hand_eye_calibration.camera_zed import Zed
from hand_eye_calibration.robot import Kinova
import traceback


def main():
    # ===== YOUR CHANGES =====
    serial = 24944966
    specs = {
        "fps": 30,
        "auto_exposure": False,
        "exposure": 20,
        "gain": 50,
        }
    
    camera = Zed(serial, specs)

    robot = Kinova(10, 1)

    offsets = [0.0, 0.16, 0.02, 90.0, 0.0, 0.0]
    marker_length = 0.07
    num_samples = 25
    move_range = 0.3
    # ========================

    try:
        camera.launch()
        robot.launch()

        calibrator = HandEyeCalibrator(camera, robot, offsets=offsets, data_dir='./data')
        T_base_cam = calibrator.calibrate(marker_length=marker_length, num_samples=num_samples, move_range=move_range, filter=True)    # set filter to False if num_samples is large (>30)

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
