import open3d as o3d
import os
from hand_eye_calibration import PointCloudMerger
import numpy as np
from hand_eye_calibration.utils.io import load_pcd, load_npy


def main():
    pcd_main = load_pcd("data/401.pcd", visualize=False)
    pcd_side = load_pcd("data/715.pcd", visualize=False)

    T_base_main = load_npy("data/T_base_main.npy")
    T_base_side = load_npy("data/T_base_side.npy")

    offset_file = None      # "data/manual_offset.npy"
    initial_offset = load_npy(offset_file) if offset_file and os.path.exists(offset_file) else None

    merger = PointCloudMerger(pcd_main, pcd_side, T_base_main, T_base_side, initial_offset)
    final_offset = merger.run_adjustment(output_filepath=offset_file)

    if final_offset is not None:
        final_merged_pcd = merger.get_merged_pcd(apply_manual_offset=True, visualize=True)
    else:
        print("\nNo offset was saved.")



if __name__ == "__main__":
    main()
