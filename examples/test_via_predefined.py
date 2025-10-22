from hand_eye_calibration.pixel_selection import PixelSelector
from realsense_toolbox import RealSenseCamera
from realsense_toolbox import PointCloudGenerator
from hand_eye_calibration.robot import Kinova
import traceback
import numpy as np

from dataclasses import dataclass
import numpy as np
from typing_extensions import override
from numpy.typing import NDArray
import torch
from typing import Callable


class WaypointGenerator:
    def __init__(self, T):
        self.bTc = T

    def get_waypoint(self, p_cam):
        p_homo = np.hstack([p_cam, np.ones(1)])
        p_base = self.bTc @ p_homo
        p_base[2] = 0.15
        target = [p_base[0], p_base[1], 0.15]
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
        self.data[2] = max(0.25, self.data[2])

    @property
    def numpy(self) -> NDArray:
        return self.data

    @property
    def torch(self) -> torch.Tensor:
        raise ValueError("Why are you calling me?")

    def __repr__(self) -> str:
        return f"Waypoint({self.data[0]:.2f}, {self.data[1]:.2f}, {self.data[2]:.2f})"



def main():
    # ===== YOUR CHANGES =====
    serial = "346522075401"
# 234222302792

    # see readme for full configurations.
    specs = {
            "fps": 30,
            "color_auto_exposure": False,
            "depth_auto_exposure": False,
        }

    pcd_config = {
        "enable_depth_filter": True,
        "min_depth": 0.1,
        "max_depth": 2.0,
        "enable_prune": True,
        "bbox_min": [-0.8, -0.3, 0.01],     # x, y, z min ranges
        "bbox_max": [0.8, 0.5, 1.8],        # x, y, z max ranges
        "enable_downsample": False,
        "voxel_size": 0.01                  # unit: meter
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

        pixel_selector = PixelSelector()

        # transform = np.load("/home/necl/Projects/hand-eye-calibration/data/relatively_good/T_346522075401.npy")
        # transform = np.load('/home/necl/Projects/hand-eye-calibration/data/bTc.npy')
        transform = np.load("/home/necl/Projects/hand-eye-calibration/data/20251020_203203/T_346522075401.npy")     # no filter, 150 trials (really bad lol)


        wp_generator = WaypointGenerator(transform)

        _, _, _, depth_frame = camera.get_current_state()
        pixels = { # 0905
            (274, 250): [0.357, 0.003],
            (391, 220): [0.575, 0.079],
            (288, 376): [0.397, -0.207],
            (167, 309): [0.21, -0.10],
            # (389, 247): [0.587, -0.037],
            # (368, 316): [0.528, -0.124],

        }

        preds = []
        for pixel in pixels:

            pixel_3d = camera.deproject_pixel_to_point(pixel, depth_frame)
            print(f"{pixel_3d = }")

            waypoint = wp_generator.get_waypoint(pixel_3d)
            print(f"waypoint = {waypoint}")
            preds.append(waypoint.data[:2] * 100)
        

            # robot.go_to_waypoint(waypoint)

        trues = np.array(list(pixels.values())) * 100

        error = np.linalg.norm((preds - trues), axis=1).mean()
        print(error)
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
