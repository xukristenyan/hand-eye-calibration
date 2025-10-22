'''
In this example, it launches a realsense camera with live view of the streaming and recording.
'''
import time
from realsense_toolbox.camera import Camera

def main():

    # ===== YOUR CHANGES =====
    serial = "346522075401"

    # see readme for full configurations.
    camera_config = {
        "enable_viewer": True,
        "enable_recorder": False,

        "specifications": {
            "fps": 30,
        "color_auto_exposure": False,
        "depth_auto_exposure": False,
        },

        "viewer": {                     # no need to keep this dict if "enable_viewer" is False
            "show_color": True,
            "show_depth": True,
            "fps": 10
        },

        "recorder": {                   # no need to keep this dict if "enable_recorder" is False
            "save_dir": "./recordings",
            "save_name": "test_session",
            "fps": 10,
            "save_with_overlays": True,
            "auto_start": False         # if False, press 's' to start recording at any time point
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