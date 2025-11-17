from hand_eye_calibration.pixel_selection import PixelSelector
from hand_eye_calibration.robot import Kinova
from hand_eye_calibration.camera_zed import Zed
from hand_eye_calibration.utils.io import load_npy

import traceback
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
import torch

class WaypointGenerator:
    def __init__(self, T):
        self.bTc = T

    def get_waypoint(self, p_cam):
        p_homo = np.hstack([p_cam, np.ones(1)])
        p_base = self.bTc @ p_homo
        
        # option1: specified height
        target = [p_base[0], p_base[1], 0.10]
        waypoint = Waypoint(data = target)

        # option2: transformed height
        # waypoint = Waypoint(data = p_base[:3])
        
        return waypoint

@dataclass
class Waypoint():
    """
    A dataclass representing waypoints in the environment which the kinova arm can navigate to.
    """

    data: NDArray[np.float32]  # [px, py, pz] in robot coordinates
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        # self.data[2] = max(0.25, self.data[2])

    @property
    def numpy(self) -> NDArray:
        return self.data

    @property
    def torch(self) -> torch.Tensor:
        raise ValueError("Why are you calling me?")

    def __repr__(self) -> str:
        return f"Waypoint({self.data[0]:.2f}, {self.data[1]:.2f}, {self.data[2]:.2f})"


def convert_a_pixel_to_waypoint(camera, pixel, wp_generator):
    pixel_3d = camera.deproject_pixel_to_point(pixel)
    print(f"point in camera: {pixel_3d}")
    waypoint = wp_generator.get_waypoint(pixel_3d)
    print(f"point in robot: {waypoint}")

    return waypoint


def main():
    # ===== YOUR CHANGES =====
    file_name = "/home/necl/Projects/hand-eye-calibration/data/20251117_124701/T_24944966.npy"

    serial = 24944966
    specs = {
        "fps": 30,
        "auto_exposure": False,
        "exposure": 20,
        "gain": 50,
        }
    camera = Zed(serial, specs)
    
    robot = Kinova(10, 1)
    # ========================

    transform = load_npy(file_name)

    try:
        camera.launch()
        robot.launch()
        robot.go_home()

        pixel_selector = PixelSelector()

        wp_generator = WaypointGenerator(transform)

        while True:
            color_image = camera.get_camera_image()

            pixels = pixel_selector.run(color_image)
            waypoint = convert_a_pixel_to_waypoint(camera, pixels[0], wp_generator)
            # print(waypoint)
            finished = robot.go_to_waypoint(waypoint)

            if finished:
                break
    
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
