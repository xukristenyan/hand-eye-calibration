from realsense_toolbox import RealSenseCamera
from realsense_toolbox import PointCloudGenerator
import numpy as np
import open3d as o3d
from hand_eye_calibration.utils.io import load_ply


def main():
    # ===== YOUR CHANGES =====
    serial = "244622072715"  # side
    serial = "346522075401"  # main


    # see readme for full configurations.
    specs = {
            "fps": 30,
            "color_auto_exposure": False,
            "depth_auto_exposure": False,
        }

    pcd_config = {  # side
        "enable_depth_filter": True,
        "min_depth": 0.1,
        "max_depth": 2.3,
        "enable_prune": True,
        "bbox_min": [-0.7, -0.6, 0.00],     # x, y, z min ranges
        "bbox_max": [0.5, 0.5, 1.8],        # x, y, z max ranges
        "enable_downsample": False,
        "voxel_size": 0.01                  # unit: meter
    }


    pcd_config = {  # main
        "enable_depth_filter": True,
        "min_depth": 0.1,
        "max_depth": 2.3,
        "enable_prune": True,
        "bbox_min": [-0.7, -0.4, 0.01],     # x, y, z min ranges
        "bbox_max": [0.5, 0.5, 1.8],        # x, y, z max ranges
        "enable_downsample": True,
        "voxel_size": 0.01                  # unit: meter
    }

    # ========================

    camera = None
    try:
        camera = RealSenseCamera(serial, specs)
        camera.launch()

        pc_generator = PointCloudGenerator(pcd_config, id=serial[-3:])
        
        while True:
            color_image, depth_image, color_frame, depth_frame = camera.get_current_state()

            if color_image is not None and depth_image is not None:
                break

        pcd = pc_generator.get_pointcloud(color_image, color_frame, depth_frame, visualize=True)

        print(f"num points in point cloud: {np.asarray(pcd.points).shape}")

        # pc_generator.save(pcd, save_dir="./data/point_clouds", filename=f"{serial[-3:]}.pcd")

        # --- 2. Color them differently (IMPORTANT!) ---

        T_color_from_left_ir = camera.get_extrinsics_left_ir_to_color()

        pcd2 = load_ply("/home/necl/Projects/FoundationStereo/test_outputs/cloud_denoise.ply", visualize=True)
        pcd2.paint_uniform_color([0, 0, 1.0])  # Color pcd2 Blue
        o3d.visualization.draw_geometries([pcd, pcd2])
        print(T_color_from_left_ir)

        T_fake = np.identity(4)
        T_fake[0, 3] = 0.5  # Add 0.5 to the x-translation

        # pcd3 = pcd2.transform(T_color_from_left_ir)
        pcd3 = pcd2.transform(T_fake)

        # --- 3. Visualize them ---

        # Method 1: The simple one-liner
        print("Displaying both point clouds. Press 'q' to close the window.")
        o3d.visualization.draw_geometries([pcd, pcd3])


        # # Method 2: The visualizer (like in your script)
        # # This gives you more control over the window
        # print("Displaying again using the Visualizer class...")
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()

        # # Add both geometries to the scene
        # vis.add_geometry(pcd)
        # vis.add_geometry(pcd2)

        # vis.run()
        # vis.destroy_window()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")

    finally:
        if camera:
            camera.shutdown()



if __name__ == "__main__":
    main()