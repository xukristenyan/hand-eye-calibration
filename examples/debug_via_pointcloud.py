import numpy as np
import open3d as o3d


def transform_point_cloud(points, transform_matrix):
    """
    Applies a 4x4 homogeneous transformation to an (N, 3) point cloud.
    """
    # Convert (N, 3) points to (N, 4) homogeneous coordinates
    num_points = points.shape[0]
    points_homogeneous = np.hstack((points, np.ones((num_points, 1))))
    
    # Apply transformation: T @ p.T
    # (4, 4) @ (4, N) -> (4, N)
    transformed_points_homo = (transform_matrix @ points_homogeneous.T).T
    
    # Convert back to (N, 3)
    transformed_points = transformed_points_homo[:, :3]
    
    return transformed_points

# def visualize_test(points_robot_frame, colors, T_robot_camera):
#     """
#     Visualizes the scene in the ROBOT frame.
#     """
#     # 1. Create the point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_robot_frame)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # 2. Create the ROBOT BASE coordinate frame (at origin)
#     robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.3, origin=[0, 0, 0])

#     # 3. Create the CAMERA'S coordinate frame
#     camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.1) # Smaller to distinguish it

#     camera_frame.transform(T_robot_camera)

#     print("Visualizing scene...")
#     print(" - Large axes = Robot Base Frame")
#     print(" - Small axes = Camera Pose")
#     print(" - Point Cloud = Data transformed to Robot Frame")
    
#     # 5. Visualize all geometries together
#     o3d.visualization.draw_geometries(
#         [pcd, robot_frame, camera_frame],
#         window_name="Camera to Robot Transform Test"
#     )
#     return pcd, robot_frame, camera_frame

# def visualize_test(points_robot_frame, colors, T_robot_camera, pcd_extra):
#     """
#     Visualizes the scene in the ROBOT frame.
#     """
#     # 1. Create the point cloud object from your manual deprojection
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_robot_frame)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # 2. Create the ROBOT BASE coordinate frame (at origin)
#     robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.3, origin=[0, 0, 0])

#     # 3. Create the CAMERA'S coordinate frame
#     camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.1) # Smaller to distinguish it

#     camera_frame.transform(T_robot_camera)

#     print("Visualizing scene...")
#     print(" - Large axes = Robot Base Frame")
#     print(" - Small axes = Camera Pose")
#     print(" - Point Cloud (Real Colors) = Your manual deprojection")
#     print(" - Point Cloud (Red) = Generator's 'ground truth' pcd")
    
#     # 5. Visualize all geometries together
#     #    *** THIS IS THE MAIN CHANGE ***
#     o3d.visualization.draw_geometries(
#         [pcd, pcd_extra, robot_frame, camera_frame],
#         window_name="Camera to Robot Transform Test"
#     )
#     return pcd, robot_frame, camera_frame

def visualize_test(points_robot_frame, points_camera_frame, colors, T_robot_camera):
    """
    Visualizes the scene by "connecting" the clouds at the camera pose.
    
    - The TRANSFORMED cloud is shown at the ROBOT BASE (0,0,0).
    - The ORIGINAL cloud is shown at the CAMERA POSE.
    """
    # 1. Create the TRANSFORMED point cloud (from points_robot_frame)
    #    We'll color this one RED and place it at the robot's origin (0,0,0)
    pcd_transformed_at_robot_base = o3d.geometry.PointCloud()
    pcd_transformed_at_robot_base.points = o3d.utility.Vector3dVector(points_robot_frame)
    pcd_transformed_at_robot_base.paint_uniform_color([1.0, 0.0, 0.0]) # Color it red
    
    # 2. Create the ORIGINAL point cloud (from points_camera_frame)
    #    This has the real colors. We must TRANSFORM it so it appears
    #    at the camera's pose.
    pcd_original_at_camera_pose = o3d.geometry.PointCloud()
    pcd_original_at_camera_pose.points = o3d.utility.Vector3dVector(points_camera_frame)
    pcd_original_at_camera_pose.colors = o3d.utility.Vector3dVector(colors)
    
    # Apply the transform to move it from the origin to the camera's pose
    pcd_original_at_camera_pose.transform(T_robot_camera)

    # 3. Create the ROBOT BASE coordinate frame (at origin)
    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0])

    # 4. Create the CAMERA'S coordinate frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1) # Smaller to distinguish it
    camera_frame.transform(T_robot_camera)

    print("Visualizing scene...")
    print(" - Large axes (at 0,0,0) = Robot Base Frame")
    print(" - Small axes = Camera Pose")
    print(" - Point Cloud (Red) = Transformed cloud (shown at Robot Base)")
    print(" - Point Cloud (Colors) = Original cloud (shown at Camera Pose)")
    
    # 5. Visualize all geometries together
    o3d.visualization.draw_geometries(
        [pcd_transformed_at_robot_base, pcd_original_at_camera_pose, robot_frame, camera_frame],
        window_name="Before vs. After Transformation (Camera-Anchored)"
    )
    return pcd_transformed_at_robot_base, pcd_original_at_camera_pose, robot_frame, camera_frame



import time
import cv2

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
        # target = [p_base[0], p_base[1], 0.15]
        # waypoint = Waypoint(data = target)
        waypoint = Waypoint(data = p_base[:3])
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
        
        transform = np.load("/home/necl/Projects/hand-eye-calibration/data/20251020_203203/T_346522075401.npy")     # no filter, 150 trials (really bad lol)

        pc_generator = PointCloudGenerator(pcd_config, id=serial[-3:])
        
        while True:
            color_image, depth_image, color_frame, depth_frame = camera.get_current_state()

            if color_image is not None and depth_image is not None:
                break

        # ground truth
        # pcd = pc_generator.get_pointcloud(color_image, color_frame, depth_frame, visualize=False)


        height, width, _ = color_image.shape
        colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        points_list = []
        colors_list = []

        for v in range(height):
            for u in range(width):
                pixel_coord = (u, v)

                point_3d = camera.deproject_pixel_to_point(pixel_coord, depth_frame)
                
                if point_3d is not None: 
                    points_list.append(point_3d)
                    
                    colors_list.append(colors[v, u])

        points_cam = np.array(points_list)
        colors_cam = np.array(colors_list) / 255.0
        
        print("Projection complete.")

        points_robot = transform_point_cloud(points_cam, transform)
        # visualize_test(points_robot, colors_cam, transform, pcd)
        visualize_test(points_robot, points_cam, colors_cam, transform)
        # waypoint = wp_generator.get_waypoint(pixel_3d)
        # print(f"waypoint = {waypoint}")

        # robot.go_to_waypoint(waypoint)

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

