from hand_eye_calibration.utils.io import load_npy
import time
from realsense_toolbox.system import CameraSystem
from realsense_toolbox import PointCloudGenerator
from hand_eye_calibration import PointCloudMerger

# load in cam system
# get point cloud of each cam
# call point cloud merger with offset applied


def main():

    # ===== YOUR CHANGES =====
    serial1 = "346522075401"
    serial2 = "244622072715"

    T_base_main = load_npy("data/Ts/T_346522075401.npy")
    T_base_side = load_npy("data/Ts/T_244622072715.npy")
    T_offset = load_npy("data/Ts/manual_offset.npy")

    # see readme for full configurations.
    cam1_config = {
        "enable_viewer": True,
        "enable_recorder": False,
    }
    cam2_config = {
        "enable_viewer": True,
        "enable_recorder": False,
    }
    pcd_config2 = {  # side
        "enable_depth_filter": True,
        "min_depth": 0.1,
        "max_depth": 2.3,
        "enable_prune": True,
        "bbox_min": [-0.7, -0.6, 0.00],     # x, y, z min ranges
        "bbox_max": [0.5, 0.5, 1.8],        # x, y, z max ranges
        "enable_downsample": False,
        "voxel_size": 0.01                  # unit: meter
    }
    pcd_config1 = {  # main
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

    system_config = {
        serial1: cam1_config, 
        serial2: cam2_config
    }

    system = None
    try:
        system = CameraSystem(system_config=system_config)
        system.launch()

        pc_generator1 = PointCloudGenerator(pcd_config1, id=serial1[-3:])
        pc_generator2 = PointCloudGenerator(pcd_config2, id=serial2[-3:])

        
        while True:
            cur_frames = system.update()
            cam1_frame = cur_frames[serial1]
            cam2_frame = cur_frames[serial2]

            if cam1_frame[0] is not None and cam1_frame[1] is not None and cam2_frame[0] is not None and cam2_frame[1] is not None:
                break

        pcd1 = pc_generator1.get_pointcloud(cam1_frame[0], cam1_frame[2], cam1_frame[3], visualize=True)
        pcd2 = pc_generator2.get_pointcloud(cam2_frame[0], cam2_frame[2], cam2_frame[3], visualize=True)


        merger = PointCloudMerger(pcd1, pcd2, T_base_main, T_base_side, T_offset)

        final_merged_pcd = merger.get_merged_pcd(apply_manual_offset=True, visualize=True)


    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")

    finally:
        system.shutdown()



if __name__ == "__main__":
    main()