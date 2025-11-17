'''
In this example, it launches a realsense camera with live view of the streaming and recording.
'''
import time
from zed_toolbox import Camera

def main():

    # ===== YOUR CHANGES =====
    serial = 24944966

    camera_config = {
        "enable_viewer": True,
        "enable_recorder": False,

        "specifications": {
            "fps": 30,
            "auto_exposure": False,
            "exposure": 20,
            "gain": 50,
        },
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