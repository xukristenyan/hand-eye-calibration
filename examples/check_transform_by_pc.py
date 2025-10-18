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
