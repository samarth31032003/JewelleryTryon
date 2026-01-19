# trackers/strategies/forehead.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter

class ForeheadStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.filter_pos = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        self.settings = {
            "Scale": 100,
            "Up_Down": 50,     # 0=Brows, 100=Hairline
            "Side": 0,         # -50=Left Temple, 0=Center, 50=Right Temple
            "Fwd_Back": 0,     # Depth (Off skin)
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
            
            # OCCLUDER (Forehead Plate)
            "Occ_Size": 100,   # Size of the blocker
            "Occ_Curve": 50    # Curvature (Depth)
        }

    def get_slider_definitions(self):
        return [
            # JEWELRY
            ("Scale", "Size", 0, 150, 100, 0.01),
            ("Up_Down", "Pos (Brows-Hair)", 0, 100, 50, 0.01), # Default Middle
            ("Side", "Side (Passa)", -50, 50, 0, 0.01),
            ("Fwd_Back", "Depth", -50, 50, 0, 0.002),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
            
            # OCCLUDER
            ("Occ_Size", "Mask Size", 0, 200, 100, 0.01),
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.face_landmarks: return []
        lms = results.face_landmarks.landmark
        
        # --- 1. Z-Depth Estimation (Shared Logic) ---
        # Estimate depth based on cheek-to-cheek distance (approx 14cm real width)
        px_l = np.array([lms[234].x * w, lms[234].y * h])
        px_r = np.array([lms[454].x * w, lms[454].y * h])
        width_px = np.linalg.norm(px_l - px_r)
        if width_px < 1: return []
        
        REAL_FACE_WIDTH = 0.14 
        f = self.camera_matrix[0,0]
        z_est = (f * REAL_FACE_WIDTH) / width_px
        
        def get_p(idx):
            return self._unproject([lms[idx].x * w, lms[idx].y * h], z_est)
            
        # --- 2. Head Rotation (Standard) ---
        p_tip = get_p(1)    # Nose Tip
        p_top = get_p(168)  # Bridge Top
        p_left = get_p(234) # Left Cheek
        p_right = get_p(454)# Right Cheek
        
        vec_up = p_top - p_tip; vec_up /= np.linalg.norm(vec_up)
        vec_right = p_right - p_left; vec_right /= np.linalg.norm(vec_right)
        vec_fwd = np.cross(vec_right, vec_up); vec_fwd /= np.linalg.norm(vec_fwd)
        vec_up = np.cross(vec_fwd, vec_right)
        
        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right
        R_head[:3, 1] = vec_up
        R_head[:3, 2] = vec_fwd
        
        # --- 3. Calculate Position ---
        # Landmarks: 10 (Hairline), 151 (Brows Center)
        # 127 (Left Temple), 356 (Right Temple) - for Side Passa
        
        p_hair = get_p(10)
        p_brow = get_p(151)
        
        # A. Vertical Interpolation (Brows <-> Hairline)
        v_ratio = self.settings.get("Up_Down", 50) / 100.0
        pos_v = p_brow * (1 - v_ratio) + p_hair * v_ratio
        
        # B. Horizontal Interpolation (Center <-> Temples)
        side_val = self.settings.get("Side", 0)
        
        if abs(side_val) < 1:
            pos_final = pos_v # Center
        elif side_val < 0:
            # Move Left (towards 127)
            p_temple_L = get_p(127)
            ratio = abs(side_val) / 50.0
            pos_final = pos_v * (1 - ratio) + p_temple_L * ratio
        else:
            # Move Right (towards 356)
            p_temple_R = get_p(356)
            ratio = side_val / 50.0
            pos_final = pos_v * (1 - ratio) + p_temple_R * ratio

        # Smooth
        pos_smooth = self.filter_pos.update(pos_final, time.time())
        
        cmds = []
        
        # --- 4. Matrices ---
        # Jewelry
        cmds.append({'type': 'mesh', 'matrix': self._build_matrix(pos_smooth, R_head, False)})
        
        # Occluder (Placed slightly behind position)
        cmds.append({'type': 'occluder', 'matrix': self._build_matrix(pos_smooth, R_head, True)})
        
        return cmds

    def _build_matrix(self, pos, rot_mat, is_occluder):
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        T_offset = np.eye(4, dtype=np.float32)
        
        if is_occluder:
            # Move slightly back (-Z) and up (+Y) to match forehead curve
            T_offset[2, 3] = -0.02 
            
            # Scale (Flattened Sphere)
            size = self.settings.get("Occ_Size", 100) * 0.001
            # Width x Height x Depth (Depth is small to make it a "Plate")
            S = np.diag([size, size * 0.8, size * 0.2, 1.0])
        else:
            # User offsets
            fwd = self.settings.get("Fwd_Back", 0) * 0.01
            T_offset[2, 3] = fwd
            
            # User Rotation
            rx = np.radians(self.settings.get("Rot_X", 0))
            ry = np.radians(self.settings.get("Rot_Y", 0))
            rz = np.radians(self.settings.get("Rot_Z", 0))
            
            import math
            def make_rot(a, ax):
                c, s = math.cos(a), math.sin(a)
                M = np.eye(4, dtype=np.float32); 
                if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
                elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
                else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
                return M
            M_user = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)
            
            # Scale
            s = self.settings.get("Scale", 100) * 0.01
            S = np.diag([s, s, s, 1.0])

        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        if is_occluder:
            return cv_to_gl @ T @ rot_mat @ T_offset @ S
        else:
            return cv_to_gl @ T @ rot_mat @ T_offset @ M_user @ S

    def _unproject(self, point_2d, z):
        u, v = point_2d
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)