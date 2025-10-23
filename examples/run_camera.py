'''
In this example, it launches a realsense camera with live view of the streaming and recording.
'''
from realsense_toolbox.camera import Camera
from src.hand_eye_calibration.config import serial, camera_config

def main():

    # ===== YOUR CHANGES =====
    # serial = "346522075401"
    serial = "244622072715"
    # serial = "234222302792"

    camera_config = {
        "enable_viewer": True,
        "enable_recorder": False,

        "specifications": {
            "fps": 30,
        "color_auto_exposure": False,
        "depth_auto_exposure": False,
        },
        "viewer": {
            "show_depth": True
        }
    }
    # ========================


    camera = None
    try:
        camera = Camera(serial, camera_config)
        camera.launch()

        while True:

            camera.update()

            if not camera.is_alive:
                break

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