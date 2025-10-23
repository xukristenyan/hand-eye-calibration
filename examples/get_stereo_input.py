'''
In this example, it launches a realsense camera with live view of the streaming and recording.
'''
import time
from realsense_toolbox.camera import Camera
from src.hand_eye_calibration.config import serial, camera_config
import cv2
from realsense_toolbox import RealSenseCamera

def main():

    # # ===== YOUR CHANGES =====
    serial = "244622072715"  # side
    # serial = "346522075401"  # main


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

        while True:
            color_image, depth_image, color_frame, depth_frame = camera.get_current_state()

            if color_image is not None and depth_image is not None:
                break

        left, right = camera.get_ir_images()
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        cv2.imwrite("ir_left_side.png", left)
        cv2.imwrite("ir_right_side.png", right)

        baseline = camera.get_baseline()
        intrinsic = camera.get_intrinsics()
        intr = intrinsic["matrix"]
        print(baseline)
        print(intr)

        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")

    finally:
        if camera:
            camera.shutdown()

        print("Shutdown complete!")



if __name__ == "__main__":
    main()