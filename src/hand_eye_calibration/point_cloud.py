

#     def get_merged_pointclouds(self, prune=False, run_icp=False, main_cam=0, vis=True, raw_vis=False, save_dir=None, object=False):

#         for serial in self.cameras:
#             self.generate_pointcloud(serial, visualize=raw_vis, object=object)
#             self.align_pointcloud_to_robot(serial, prune=prune, visualize=vis)

#         # ICP refinement (optional)
#         # WARNING: This piece is written by ChatGPT w/o modification
#         if run_icp:
#             if len(self.serials) >= 2:
#                 anchor_serial = self.serials[main_cam]
#                 for s in self.serials:
#                     if s == anchor_serial:
#                         continue
#                     self.icp_refine_between(
#                         src_serial=s,
#                         tgt_serial=anchor_serial,
#                     )
        
#         self.merged_pcds = self.fuse_scene_pointclouds(serials=self.serials, visualize=vis)

#         if save_dir is not None:
#             self.save_results(save_dir)

#         return self.merged_pcds


#     def align_pointcloud_to_robot(self, serial, prune=False, visualize=False):
#         pcd = self.cameras[serial]["pcd"]
#         color = self.cameras[serial]["color"]
#         TCR = self.cameras[serial]["TCR"]

#         pcd_homog = np.vstack((pcd.T, np.ones((1, pcd.shape[0]))))  # Shape: (4, N)
#         pcd_robot = (TCR @ pcd_homog).T[:, :3]
#         color_robot = color

#         if prune:
#             pcd_robot, color_robot = self.prune_pointcloud(pcd_robot, color_robot)

#         self.cameras[serial]["pcd_robot"] = pcd_robot
#         self.cameras[serial]["color_robot"] = color_robot

#         if visualize:
#             self.visualize_pointcloud(pcd_robot, color_robot, name=f"{serial} Aligned to Robot Frame")



# class PointCloudMerger():
#     '''
#     To merge multiple pointclouds of a static scene.
#     '''

# # [(pc1, T1), (pc2, T2), ...]
#     def __init__(self, pairs, ):
#         self.pairs = pairs


#     def transform_pointcloud(self, pc, T):
#         pcd_o3d = o3d.geometry.PointCloud()
#         pcd_o3d.points = o3d.utility.Vector3dVector(pc)
#         pcd_o3d.transform(T)
#         return pcd_o3d


#     def merge(self):
#         pass

#         # WARNING: This piece is written by ChatGPT w/o modification
#         serials = serials or list(self.cameras.keys())
#         clouds = []
#         for s in serials:
#             pts = self.cameras[s].get("pcd_robot", None)
#             cols = self.cameras[s].get("color_robot", None)
#             if pts is None or pts.shape[0] == 0:
#                 continue
#             clouds.append(self._to_o3d(pts, cols))
#         if len(clouds) == 0:
#             print("No clouds to fuse.")
#             return None
#         pcd_merged = clouds[0]
#         for c in clouds[1:]:
#             pcd_merged += c

#         if visualize:
#             frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#             o3d.visualization.draw_geometries([pcd_merged, frame], window_name="Fused Scene")
#         return pcd_merged
    
#     def get_merged_pointcloud(self, apply_icp=False, visualize=False):
#         for pc, T in self.pairs:
#             pcd_o3d = self.transform_pointcloud(pc, T)

#         # self.merge
#         # 
    
#     def visualize(self):
#         pass


#     def fuse_scene_pointclouds(self, serials=None, visualize=False):




import open3d as o3d
import numpy as np
import copy

class PointCloudMerger:
    def __init__(self, pcd_files, transformations):
        if len(pcd_files) != len(transformations):
            raise ValueError("The number of pcd_files must match the number of transformations.")

        self.pcd_files = pcd_files
        self.transformations = transformations
        self.original_pcds = self._load_pcds()
        self.aligned_pcds = []
        self.merged_pcd = None

    def _load_pcds(self):
        """Loads point clouds from the specified files."""
        pcds = []
        for f in self.pcd_files:
            pcd = o3d.io.read_point_cloud(f)
            if not pcd.has_points():
                print(f"Warning: Point cloud file is empty or could not be read: {f}")
            pcds.append(pcd)
        return pcds

    def align_pointclouds(self):
        """
        Aligns the point clouds to the robot base frame using the provided
        transformation matrices.
        """
        self.aligned_pcds = []
        for pcd, T in zip(self.original_pcds, self.transformations):
            # Create a deep copy to avoid modifying the original point cloud
            pcd_transformed = copy.deepcopy(pcd)
            pcd_transformed.transform(T)
            self.aligned_pcds.append(pcd_transformed)
        print("All point clouds have been transformed to the robot base frame.")

    def _run_icp_refinement(self, target_pcd_index=0, max_correspondence_distance=0.05, voxel_size=0.02):
        """
        Refines the alignment of point clouds using pairwise ICP.

        Args:
            target_pcd_index (int): The index of the point cloud to use as the
                                    stable target (anchor) for alignment.
            max_correspondence_distance (float): Maximum distance between two points
                                                 to be considered a correspondence.
            voxel_size (float): Voxel size used for downsampling before ICP.
        """
        if len(self.aligned_pcds) < 2:
            print("ICP refinement requires at least two point clouds. Skipping.")
            return

        print("Starting ICP refinement...")
        # The target point cloud remains fixed
        target_pcd = self.aligned_pcds[target_pcd_index]
        
        # Downsample for faster ICP
        target_down = target_pcd.voxel_down_sample(voxel_size)
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        for i in range(len(self.aligned_pcds)):
            if i == target_pcd_index:
                continue

            source_pcd = self.aligned_pcds[i]
            source_down = source_pcd.voxel_down_sample(voxel_size)
            source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

            print(f"Running ICP between point cloud {i} (source) and {target_pcd_index} (target)...")
            
            # Point-to-plane ICP is generally more robust
            icp_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, max_correspondence_distance, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            # Apply the refined transformation to the original (not downsampled) point cloud
            self.aligned_pcds[i].transform(icp_result.transformation)

        print("ICP refinement complete.")


    def merge(self):
        """Merges all the aligned point clouds into a single point cloud."""
        if not self.aligned_pcds:
            print("Warning: No aligned point clouds to merge. Run align_pointclouds() first.")
            return
            
        print("Fusing aligned point clouds...")
        # Combine all point clouds
        self.merged_pcd = o3d.geometry.PointCloud()
        for pcd in self.aligned_pcds:
            self.merged_pcd += pcd

        print(f"Fusion complete. Merged point cloud has {len(self.merged_pcd.points)} points.")
        return self.merged_pcd

    def get_merged_pointcloud(self, run_icp=False, target_pcd_index=0, visualize=False, voxel_size=0.01):
        """
        The main method to process and merge point clouds.

        Args:
            run_icp (bool): If True, run ICP refinement after initial alignment.
            target_pcd_index (int): Index of the point cloud to use as the ICP anchor.
            visualize (bool): If True, visualize the intermediate and final results.
            voxel_size (float): Voxel size for the final downsampling to clean up the merged cloud.

        Returns:
            open3d.geometry.PointCloud: The final merged and downsampled point cloud.
        """
        if visualize:
            print("Visualizing original point clouds (in their own frames)...")
            self.visualize(self.original_pcds, "Original Point Clouds")

        # Step 1: Align point clouds using transformation matrices
        self.align_pointclouds()

        if visualize:
            print("Visualizing point clouds after transforming to robot base...")
            self.visualize(self.aligned_pcds, "Aligned Point Clouds (Before ICP)")

        # Step 2: Optionally refine with ICP
        if run_icp:
            self._run_icp_refinement(target_pcd_index=target_pcd_index)
            if visualize:
                print("Visualizing point clouds after ICP refinement...")
                self.visualize(self.aligned_pcds, "Aligned Point Clouds (After ICP)")
        
        # Step 3: Merge the clouds
        self.merge()

        # Step 4: Downsample for a cleaner result
        if self.merged_pcd and voxel_size > 0:
            print(f"Downsampling merged cloud with voxel size: {voxel_size}")
            self.merged_pcd = self.merged_pcd.voxel_down_sample(voxel_size)
            print(f"Downsampled cloud has {len(self.merged_pcd.points)} points.")

        if visualize and self.merged_pcd:
            print("Visualizing final merged point cloud...")
            self.visualize([self.merged_pcd], "Final Merged Point Cloud")
            
        return self.merged_pcd

    def visualize(self, pcds, window_name="PointCloud Visualization"):
        """
        Visualizes a list of point clouds.

        Args:
            pcds (list): A list of open3d.geometry.PointCloud objects.
            window_name (str): The title of the visualization window.
        """
        # Assign a different color to each point cloud for clarity
        vis_pcds = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
        for i, pcd in enumerate(pcds):
            pcd_vis = copy.deepcopy(pcd)
            pcd_vis.paint_uniform_color(colors[i % len(colors)])
            vis_pcds.append(pcd_vis)
            
        # Add a coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis_pcds.append(coord_frame)
        
        o3d.visualization.draw_geometries(vis_pcds, window_name=window_name)

    def save_merged_pcd(self, filepath):
        """
        Saves the merged point cloud to a file.

        Args:
            filepath (str): The path where the .pcd file will be saved.
        """
        if self.merged_pcd:
            o3d.io.write_point_cloud(filepath, self.merged_pcd)
            print(f"Merged point cloud saved to {filepath}")
        else:
            print("No merged point cloud available to save.")

