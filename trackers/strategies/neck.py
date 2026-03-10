# trackers/strategies/neck.py
import numpy as np
import cv2
import time
import math
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter
from trackers.occluder_shared import OccluderManager

class NeckStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # Smooth the center position to prevent jitter
        self.filter_pos = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        
        self.settings = {
            "Scale": 150,      
            "Up_Down": 0,      
            "Fwd_Back": 0,     
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
            # Occluder settings handled by Manager
        }

    def get_slider_definitions(self):
        sliders = [
            # 1. EXPONENTIAL SCALER
            ("Scale", "Size", 1, 500, 150, 1.0),
            
            # 2. POSITION CONTROLS
            ("Up_Down", "Height", -500, 500, 0, 1.0),
            ("Fwd_Back", "Depth", -500, 500, 0, 1.0), 
            
            # 3. ROTATION
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
        ]
        
        # Only draw occluder sliders if we are in 3D mode
        if getattr(self, 'mode', '3d') == "3d":
            if hasattr(OccluderManager, 'get_body_sliders'):
                sliders.extend(OccluderManager.get_body_sliders())
            elif hasattr(OccluderManager, 'get_shared_sliders'):
                sliders.extend(OccluderManager.get_shared_sliders())
        return sliders

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        
        if not results.pose_landmarks: return []

        lm_l = results.pose_landmarks.landmark[11] # Left Shoulder
        lm_r = results.pose_landmarks.landmark[12] # Right Shoulder
        
        if lm_l.visibility < 0.5 or lm_r.visibility < 0.5: return []

        # 1. Pixel Coordinates (Ultra Stable)
        px_l = np.array(self._get_px(lm_l, w, h))
        px_r = np.array(self._get_px(lm_r, w, h))
        
        # 2. Estimate Depth (Z) based purely on 2D pixel width
        REAL_SHOULDER_WIDTH = 0.40
        px_width = np.linalg.norm(px_l - px_r)
        if px_width < 10: return []
        
        f = self.camera_matrix[0,0]
        z_est = (f * REAL_SHOULDER_WIDTH) / px_width
        
        # 3. Get 3D Points using identical Z for both shoulders (Removes yaw jitter entirely)
        p_l = self._unproject(px_l, z_est)
        p_r = self._unproject(px_r, z_est)
        
        p_center = (p_l + p_r) / 2.0
        p_smooth = self.filter_pos.update(p_center, time.time())
        
        # 4. Construct Rotation Matrix (Pure 2D Chest Plane)
        vec_x = p_r - p_l
        vec_x /= np.linalg.norm(vec_x)
        
        vec_cam_fwd = np.array([0, 0, 1], dtype=np.float32)
        vec_y = np.cross(vec_cam_fwd, vec_x) 
        vec_y /= np.linalg.norm(vec_y)
        
        vec_z = np.cross(vec_x, vec_y)
        
        R_chest = np.eye(4, dtype=np.float32)
        R_chest[:3, 0] = vec_x
        R_chest[:3, 1] = vec_y 
        R_chest[:3, 2] = vec_z

        # 5. Build Matrices
        # A. Necklace (Jewelry)
        mat_jewel = self._build_matrix(p_smooth, R_chest)
        render_commands.append({'type': 'mesh', 'matrix': mat_jewel})
        
        # B. Shared Body Occluder
        mat_occ = OccluderManager.get_body_matrix(self.settings, p_smooth, R_chest)
        render_commands.append({
            'type': 'occluder', 
            'matrix': mat_occ,
            'file_path': OccluderManager.PATH_BODY,
            'mesh_key': 'occ_body'
        })

        return render_commands

    def _build_matrix(self, pos, rot_mat):
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        
        up_down = self.settings.get("Up_Down", 0) * 0.0001
        fwd_back = self.settings.get("Fwd_Back", 0) * 0.0001
        
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[1, 3] = up_down
        T_offset[2, 3] = fwd_back

        # --- ROTATION FIX ---
        # "Gravity Slope" default rotation
        base_x = -125
        base_y = 0
        base_z = 180
        
        rx = np.radians(base_x + self.settings.get("Rot_X", 0))
        ry = np.radians(base_y + self.settings.get("Rot_Y", 0))
        rz = np.radians(base_z + self.settings.get("Rot_Z", 0))
        
        def make_rot(a, ax):
            c, s = math.cos(a), math.sin(a)
            M = np.eye(4, dtype=np.float32); 
            if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
            elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
            else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
            return M
        
        M_rot = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)

        # Scale
        raw_scale = self.settings.get("Scale", 150)
        s = 0.00001 * (raw_scale ** 2.0)
        S = np.diag([s, s, s, 1.0])

        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ rot_mat @ T_offset @ M_rot @ S

    def _get_px(self, lm, w, h): return [int(lm.x * w), int(lm.y * h)]

    def _unproject(self, point_2d, z):
        u, v = point_2d
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)