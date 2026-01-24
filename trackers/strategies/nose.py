# trackers/strategies/nose.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter

class NoseStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.filter_pos = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        # Default Settings
        self.settings = {
            "Scale": 100,
            "Side": 50,        # 0=Left, 50=Center, 100=Right
            "Up_Down": 0,      # Adjust vertical placement
            "Fwd_Back": 0,     # Depth (Piercing effect)
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
            
            # OCCLUDER (The Nose Bulb)
            "Occ_Size": 100,   # Overall size of the nose bulb
            "Occ_Y": 0         # Move mask Up/Down independently of jewelry
        }

    def get_slider_definitions(self):
        return [
            # JEWELRY
            ("Side", "L <-> R", 0, 100, 20, 0.01),  # Default 20 (Left-ish)
            ("Scale", "Size", 0, 150, 100, 0.01),
            ("Up_Down", "Height", -50, 50, 0, 0.002),
            ("Fwd_Back", "Depth", -50, 50, 0, 0.002),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
            
            # OCCLUDER
            ("Occ_Size", "Mask Size", 0, 200, 100, 0.01),
            ("Occ_Y", "Mask Height", -50, 50, 0, 0.002)
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        
        # Check Face Landmarks
        if not results.face_landmarks: 
            return []
        
        lms = results.face_landmarks.landmark
        
        # 1. Calc Depth (Z Estimation) using Face Width
        # 234 = Left Ear/Cheek, 454 = Right Ear/Cheek
        px_l = np.array([lms[234].x * w, lms[234].y * h])
        px_r = np.array([lms[454].x * w, lms[454].y * h])
        width_px = np.linalg.norm(px_l - px_r)
        
        if width_px < 1: return []
        
        REAL_FACE_WIDTH = 0.14 # Approx 14cm
        f = self.camera_matrix[0,0]
        z_est = (f * REAL_FACE_WIDTH) / width_px
        
        # Helper to get 3D point from landmark index
        def get_p(idx):
            u, v = lms[idx].x * w, lms[idx].y * h
            return self._unproject([u,v], z_est)

        # Key Landmarks
        # 1 = Nose Tip, 168 = Top Bridge, 49 = Left Alar, 279 = Right Alar
        p_tip = get_p(1)
        p_top = get_p(168)
        p_left = get_p(49)
        p_right = get_p(279)
        
        # 2. Calculate Head Rotation Matrix
        vec_up = p_top - p_tip
        vec_up /= np.linalg.norm(vec_up)
        
        vec_right = p_right - p_left
        vec_right /= np.linalg.norm(vec_right)
        
        # Normal (Forward)
        vec_fwd = np.cross(vec_right, vec_up)
        vec_fwd /= np.linalg.norm(vec_fwd)
        
        # Re-orthogonalize Up
        vec_up = np.cross(vec_fwd, vec_right)
        
        # Build Rotation Matrix
        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right
        R_head[:3, 1] = vec_up
        R_head[:3, 2] = vec_fwd

        # 3. Calculate Base Position (Interpolate Left <-> Right)
        side_val = self.settings.get("Side", 50)
        
        if side_val < 50:
            # Left side (0 to 50 maps to Left -> Tip)
            ratio = side_val / 50.0
            pos_base = p_left * (1 - ratio) + p_tip * ratio
        else:
            # Right side (50 to 100 maps to Tip -> Right)
            ratio = (side_val - 50) / 50.0
            pos_base = p_tip * (1 - ratio) + p_right * ratio

        # Smoothing
        pos_smooth = self.filter_pos.update(pos_base, time.time())

        # 4. Build Render Commands
        cmds = []
        
        # A. Jewelry Matrix
        mat_jewel = self._build_matrix(pos_smooth, R_head, is_occluder=False)
        cmds.append({'type': 'mesh', 'matrix': mat_jewel})
        
        # B. Occluder (Nose Bulb) - Placed at TIP, not jewelry pos
        mat_occ = self._build_matrix(p_tip, R_head, is_occluder=True)
        cmds.append({'type': 'occluder', 'matrix': mat_occ})

        return cmds

    def _build_matrix(self, pos, rot_mat, is_occluder):
        # 1. Base Position
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # 2. Offsets
        T_offset = np.eye(4, dtype=np.float32)
        
        if is_occluder:
            # Occluder specific adjustments
            occ_y = self.settings.get("Occ_Y", 0) * 0.01
            T_offset[1, 3] = occ_y
            
            # Scale (The "Bulb Size")
            scale = self.settings.get("Occ_Size", 100) * 0.015 
            # Make it slightly flattened (wider than tall)
            S = np.diag([scale * 1.2, scale, scale, 1.0])
            
        else:
            # Jewelry adjustments
            up_down = self.settings.get("Up_Down", 0) * 0.01
            fwd_back = self.settings.get("Fwd_Back", 0) * 0.01
            
            T_offset[1, 3] = up_down
            T_offset[2, 3] = fwd_back
            
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