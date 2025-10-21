import open3d as o3d
import os
from hand_eye_calibration import PointCloudMerger
import numpy as np



if __name__ == '__main__':

    # 2. Define file paths and transformation matrices
    pcd_files = ["cam1.pcd", "cam2.pcd"]

    # Transformation for cam1 (e.g., identity if it's the reference)
    T1 = np.identity(4)

    # Transformation for cam2 (e.g., translate 0.5m in x and rotate 45 deg around z)
    T2 = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
    T2 = np.hstack((T2, np.array([[0.5], [0], [0]])))
    T2 = np.vstack((T2, np.array([0, 0, 0, 1])))
    
    transformations = [T1, T2]
    
    # 3. Initialize the merger
    merger = PointCloudMerger(pcd_files=pcd_files, transformations=transformations)

    # 4. Get the merged point cloud with ICP and visualization enabled
    final_pcd = merger.get_merged_pointcloud(run_icp=True, visualize=True)

    # 5. Save the final result
    if final_pcd:
        merger.save_merged_pcd("merged_cloud.pcd")






from hand_eye_calibration import HandEyeCalibrator
from realsense_toolbox import RealSenseCamera
from realsense_toolbox import PointCloudGenerator
from hand_eye_calibration.robot import Kinova
import traceback

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

        # robot.go_home()


        calibrator = HandEyeCalibrator(camera, robot, data_dir='./data', save_images=True)

        pc_generator = PointCloudGenerator(pcd_config, id=serial[-3:])
        
        while True:
            color_image, depth_image, color_frame, depth_frame = camera.get_current_state()

            if color_image is not None and depth_image is not None:
                break

        pcd = pc_generator.get_pointcloud(color_image, color_frame, depth_frame, visualize=True)

        T_base_cam = calibrator.calibrate(marker_length=0.07, num_samples=10, filter=True)

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
