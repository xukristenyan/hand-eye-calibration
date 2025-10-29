from hand_eye_calibration.pixel_selection import PixelSelector
from hand_eye_calibration.robot import Kinova
import traceback
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
import torch
from hand_eye_calibration.utils.io import load_npy
from zed_toolbox import ZedCamera

class WaypointGenerator:
    def __init__(self, T):
        self.bTc = T

    def get_waypoint(self, p_cam):
        p_homo = np.hstack([p_cam, np.ones(1)])
        p_base = self.bTc @ p_homo
        target = [p_base[0], p_base[1], 0.10]
        waypoint = Waypoint(data = target)
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
    pixel_3d = camera.deproject_to_3d(pixel)
    print(f"point in camera: {pixel_3d}")
    waypoint = wp_generator.get_waypoint(pixel_3d)
    print(f"point in robot: {waypoint}")

    return waypoint

def tune_transform(transform, x=None, y=None, z=None):   # x, y, z are offsets in meters
    # tune the transform according to the specified offsets
    x = 0 if x is None else x
    y = 0 if y is None else y
    z = 0 if z is None else z
    transform[0,3] += x
    transform[1,3] += y
    transform[2,3] += z

    return transform

def main():
    # ===== YOUR CHANGES =====
    serial = "346522075401"
    # serial = "244622072715"

    # /home/necl/Projects/hand-eye-calibration/data/20251020_114301/T_346522075401.npy     # rotated, 90
    # /home/necl/Projects/hand-eye-calibration/data/relatively_good/T_346522075401.npy
    # /home/necl/Projects/hand-eye-calibration/data/bTc.npy
    # /home/necl/Projects/hand-eye-calibration/data/20251020_203203/T_346522075401.npy     # no filter, 150 trials (really bad lol)
    # /home/necl/Projects/hand-eye-calibration/data/transform_tests/T_346522075401.npy

    # file_name = "/home/necl/Projects/hand-eye-calibration/data/good_401/T_346522075401.npy"
    file_name = "/home/necl/Projects/hand-eye-calibration/data/20251029_114835/T_24944966.npy"
    # file_name = "/home/necl/Projects/hand-eye-calibration/data/Ts/T_346522075401.npy"
    transform = load_npy(file_name)
    transform = tune_transform(transform)

    # ========================
    # ===== YOUR CHANGES =====
    serial = 24944966

    # see readme for full configurations.
    specs = {"fps": 30}
    # ========================

    # see readme for full configurations.
    # specs = {
    #         "fps": 30,
    #         "color_auto_exposure": False,
    #         "depth_auto_exposure": False,
    #     }
    camera = None
    try:
        camera = ZedCamera(serial, specs)
        camera.launch()

        robot = Kinova(10, 1)
        robot.launch()
        if robot.connected:
            print("Connection successful")

        robot.go_home()

        pixel_selector = PixelSelector()

        wp_generator = WaypointGenerator(transform)

        color_image, _ = camera.get_rgbd()

        pixels = pixel_selector.run(color_image)
        waypoint = convert_a_pixel_to_waypoint(camera, pixels[0], wp_generator)
        robot.go_to_waypoint(waypoint)

        # save updated transform
        res = input('save the change?')
        if res == 'y' or res == 'Y':
            np.save(file_name, transform)
            print('SAVED')
        else:
            print('Transform UNCHANGED')
    
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
