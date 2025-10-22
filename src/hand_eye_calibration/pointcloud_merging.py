import open3d as o3d
import numpy as np
import copy
import sys
from .utils.pointcloud import get_transform_from_params, get_params_from_transform


class PointCloudMerger:
    '''
    Manage merging and manually adjusting point clouds.
    
    This class transforms a main and side point cloud into the robot base frame,
    then allows for fine-grained manual adjustment of the side camera's
    point cloud to perfect the alignment.
    '''
    def __init__(self, pcd_main, pcd_side, T_main, T_side, initial_offset=None):
        '''
        pcd_main: Open3D PointCloud from the main camera (in camera frame).
        pcd_side: Open3D PointCloud from the side camera (in camera frame).
        T_main: 4x4 NumPy array (T_base_cam) for the main camera.
        T_side: 4x4 NumPy array (T_base_cam) for the side camera.
        initial_offset: 4x4 NumPy array for an existing manual offset
                        to apply to the side camera. Defaults to identity.
        '''
        self.pcd_main = pcd_main
        self.pcd_side = pcd_side
        self.T_main = T_main
        self.T_side = T_side

        # transform both point clouds into the robot base frame
        self.pcd_main_base = copy.deepcopy(pcd_main).transform(T_main)
        self.pcd_side_base = copy.deepcopy(pcd_side).transform(T_side)
        self.pcd_side_base_original = copy.deepcopy(self.pcd_side_base)

        # initialize manual offset
        if initial_offset is None:
            self.manual_offset = np.eye(4)
            self.adjustment_params = {'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 
                                      'rx': 0.0, 'ry': 0.0, 'rz': 0.0}
        else:
            self.manual_offset = initial_offset
            self.adjustment_params = get_params_from_transform(initial_offset)

        # initialize visualizer
        self.vis = None
        self.pcd_side_vis = None
        self.should_save = False

        # initialize step sizes
        self.trans_step = 0.001  # 1 mm
        self.rot_step = 0.5      # 0.5 degrees
        

    def _print_controls(self):
        print("\n" + "="*70)
        print("           Interactive Point Cloud Alignment Tool")
        print("="*70)
        print("  Controls (focus the Open3D window):")
        print("\n  Translation (WASD + RF):")
        print("    W/S : Translate +X / -X (Forward/Back)")
        print("    A/D : Translate +Y / -Y (Left/Right)")
        print("    R/F : Translate +Z / -Z (Up/Down)")
        print("\n  Rotation (Arrow Keys + QE):")
        print("    ↑/↓ : Rotate +X / -X (Pitch)")
        print("    ←/→ : Rotate +Y / -Y (Yaw)")
        print("    Q/E : Rotate +Z / -Z (Roll)")
        print("\n  Other:")
        print("    +/- : Double/Halve step size (Trans & Rot)")
        print("    Z   : Reset adjustment to original (zero offset)")
        print("    ENTER : Save current offset and exit")
        print("    ESC : Exit without saving")
        print("="*70 + "\n")


    def _update_visualization(self):
        self.manual_offset = get_transform_from_params(self.adjustment_params)
        
        pcd_new = copy.deepcopy(self.pcd_side_base_original)
        pcd_new.transform(self.manual_offset)

        # update the visualizer geometry
        self.pcd_side_vis.points = pcd_new.points
        self.pcd_side_vis.colors = pcd_new.colors
        self.vis.update_geometry(self.pcd_side_vis)


    def _make_param_callback(self, param_key, delta):
        def callback(vis):
            self.adjustment_params[param_key] += delta
            self._update_visualization()
            return False
        
        return callback


    def _make_step_callback(self, multiplier):
        def callback(vis):
            self.trans_step *= multiplier
            self.rot_step *= multiplier
            self._update_visualization()
            return False

        return callback


    def _reset_callback(self, vis):
        for key in self.adjustment_params:
            self.adjustment_params[key] = 0.0

        self._update_visualization()

        return False


    def _save_and_exit_callback(self, vis):
        self.should_save = True
        self.vis.close()

        return False


    def _exit_callback(self, vis):
        self.should_save = False
        self.vis.close()

        return False


    def run_adjustment(self, output_filepath):
        '''
        Launch the Open3D visualizer for manual adjustment.

        Args:
            output_filepath: Path to save the .npy file (e.g., "manual_offset.npy").

        Returns:
            The final 4x4 manual offset matrix if saved, otherwise None.
        '''
        self._print_controls()

        # initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="PointCloud Alignment Tool", width=1920, height=1080)
        
        # load in point clouds and frame
        self.vis.add_geometry(self.pcd_main_base)
        self.pcd_side_vis = copy.deepcopy(self.pcd_side_base_original)
        self.vis.add_geometry(self.pcd_side_vis)
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(coord_frame)
        
        # register KeyCallbacks        
        # Translation (WASD + RF)
        self.vis.register_key_callback(ord('W'), self._make_param_callback('tx', self.trans_step))
        self.vis.register_key_callback(ord('S'), self._make_param_callback('tx', -self.trans_step))
        self.vis.register_key_callback(ord('A'), self._make_param_callback('ty', self.trans_step))
        self.vis.register_key_callback(ord('D'), self._make_param_callback('ty', -self.trans_step))
        self.vis.register_key_callback(ord('R'), self._make_param_callback('tz', self.trans_step))
        self.vis.register_key_callback(ord('F'), self._make_param_callback('tz', -self.trans_step))

        # Rotation (Arrow Keys + QE)
        self.vis.register_key_callback(265, self._make_param_callback('rx', self.rot_step))  # Up
        self.vis.register_key_callback(264, self._make_param_callback('rx', -self.rot_step)) # Down
        self.vis.register_key_callback(263, self._make_param_callback('ry', self.rot_step))  # Left
        self.vis.register_key_callback(262, self._make_param_callback('ry', -self.rot_step)) # Right
        self.vis.register_key_callback(ord('Q'), self._make_param_callback('rz', self.rot_step))
        self.vis.register_key_callback(ord('E'), self._make_param_callback('rz', -self.rot_step))

        # Step Size (+/-)
        self.vis.register_key_callback(ord('='), self._make_step_callback(2.0)) # +
        self.vis.register_key_callback(ord('+'), self._make_step_callback(2.0)) # (Numpad)
        self.vis.register_key_callback(ord('-'), self._make_step_callback(0.5)) # -
        
        # Meta Controls
        self.vis.register_key_callback(ord('Z'), self._reset_callback)          # Reset
        self.vis.register_key_callback(257, self._save_and_exit_callback)     # Enter
        self.vis.register_key_callback(256, self._exit_callback)              # ESC

        # run
        self._update_visualization()
        self.vis.run()
        self.vis.destroy_window()

        # post-run
        if self.should_save:
            self.save_offset(output_filepath)
            return self.manual_offset
        else:
            print("[PointCloudMerger] Adjustment cancelled.")
            return None


    def save_offset(self, output_filepath):
        print(f"[PointCloudMerger] Saved final offset to {output_filepath}")
        np.save(output_filepath, self.manual_offset)


    def get_merged_pcd(self, apply_manual_offset=True, visualize=False):
        if apply_manual_offset:
            pcd_side_final = copy.deepcopy(self.pcd_side_base_original)
            pcd_side_final.transform(self.manual_offset)
            merged_pcd = self.pcd_main_base + pcd_side_final

        else:
            merged_pcd = self.pcd_main_base + self.pcd_side_base

        if visualize:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([merged_pcd, coord_frame], window_name="Merged Point Cloud", width=1920, height=1080)

        return merged_pcd
