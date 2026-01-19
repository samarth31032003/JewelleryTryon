# trackers/strategies/waist.py
import numpy as np
import cv2
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter, RotationFilter

class WaistStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # Filter for the main center position
        self.filter_pos = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        
        # Default Settings
        self.settings = {
            "Scale": 100,      # Overall Size
            "Up_Down": 0,      # Vertical offset (Waist vs Hips)
            "Rot_X": 0,
            "Rot_Y": 0,
            "Rot_Z": 0,
            "Occ_Width": 100,  # Torso Width %
            "Occ_Depth": 100   # Torso Depth %
        }

    def get_slider_definitions(self):
        return [
            # JEWELRY
            ("Scale", "Size", 0, 200, 100, 0.01),
            ("Up_Down", "Height", -200, 200, 0, 0.005), # Moves up/down torso
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            
            # OCCLUDER (Torso Mask)
            ("Occ_Width", "Body Width", 0, 150, 100, 0.01),
            ("Occ_Depth", "Body Thick", 0, 150, 70, 0.01) # Default 70% of width
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        
        if not results.pose_landmarks: 
            return []

        # MediaPipe Pose Landmarks: 23 = Left Hip, 24 = Right Hip
        lm_l = results.pose_landmarks.landmark[23]
        lm_r = results.pose_landmarks.landmark[24]
        
        # Visibility Check
        if lm_l.visibility < 0.5 or lm_r.visibility < 0.5:
            return []

        # 1. Get 2D Pixels
        px_l = np.array(self._get_px(lm_l, w, h))
        px_r = np.array(self._get_px(lm_r, w, h))
        
        # 2. Calculate Depth (Z) Estimation
        # We assume an average human hip width is ~30cm (0.3m)
        REAL_HIP_WIDTH = 0.30 
        px_width = np.linalg.norm(px_l - px_r)
        if px_width < 10: return [] # Too far/noise
        
        focal_length = self.camera_matrix[0,0]
        z_est = (focal_length * REAL_HIP_WIDTH) / px_width
        
        # 3. Get 3D Center Position
        # Midpoint in 2D
        px_center = (px_l + px_r) / 2.0
        # Project to 3D
        p_center = self._unproject(px_center, z_est)
        
        # 4. Smooth Position
        # We assume current time is roughly monotonic for smoothing
        import time
        p_smooth = self.filter_pos.update(p_center, time.time())

        # 5. Construct Rotation Matrix
        # X-Axis: From Left Hip to Right Hip
        vec_x = self._unproject(px_r, z_est) - self._unproject(px_l, z_est)
        vec_x /= np.linalg.norm(vec_x)
        
        # Y-Axis: Up (Approximate -0.1 in Y is up in Camera space usually, but simple Up is (0,-1,0))
        # Better: Cross product of X and Forward(0,0,1) gives Up
        vec_z = np.array([0, 0, 1], dtype=np.float32) # Camera looks down Z
        vec_y = np.cross(vec_z, vec_x)
        vec_y /= np.linalg.norm(vec_y)
        
        # Re-orthogonalize Z
        vec_z = np.cross(vec_x, vec_y)
        
        # Build Rotation Matrix
        R_body = np.eye(4, dtype=np.float32)
        R_body[:3, 0] = vec_x
        R_body[:3, 1] = vec_y # Note: In GL Y is Up, in CV Y is Down. 
        R_body[:3, 2] = vec_z

        # 6. Apply Sliders & Create Matrices
        
        # --- A. Jewelry Matrix ---
        mat_jewel = self._build_matrix(p_smooth, R_body, is_occluder=False)
        render_commands.append({'type': 'mesh', 'matrix': mat_jewel})
        
        # --- B. Occluder Matrix ---
        mat_occ = self._build_matrix(p_smooth, R_body, is_occluder=True)
        render_commands.append({'type': 'occluder', 'matrix': mat_occ})

        return render_commands

    def _build_matrix(self, pos, rot_mat, is_occluder):
        # 1. Base Transformation
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # 2. Slider Offsets
        up_down = self.settings.get("Up_Down", 0) * 0.01 # Meters
        T_offset = np.eye(4, dtype=np.float32)
        # Move along Body Y axis (Vertical relative to hips)
        T_offset[1, 3] = -up_down # Negative because Y is flipped in CV/GL mix usually
        
        # 3. User Rotation
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
        M_user_rot = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)
        
        # 4. Scaling
        if is_occluder:
            # Occluder is a Cylinder. 
            # Default Cylinder is Height=1 (Y), Radius=1 (X/Z)
            # We want it: Width=HipWidth, Depth=TorsoThick
            w_scale = self.settings.get("Occ_Width", 100) * 0.0035 
            d_scale = self.settings.get("Occ_Depth", 70) * 0.0035
            h_scale = 0.4 # Fixed height for torso occluder
            
            S = np.diag([w_scale, h_scale, d_scale, 1.0])
            
            # Fix Orientation: Cylinder default is Y-up. 
            # Our R_body aligns X to hips. We need to rotate Cylinder 90deg Z?
            # Actually, standard cylinder is Y-up. Our R_body Y is Up. 
            # So just Scale is enough.
        else:
            s_val = self.settings.get("Scale", 100) * 0.01
            S = np.diag([s_val, s_val, s_val, 1.0])

        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        # Pipeline: Scale -> UserRot -> BodyRot -> Offset -> Position -> CV2GL
        if is_occluder:
             # Occluder doesn't rotate with UserRot, only BodyRot
             return cv_to_gl @ T @ rot_mat @ T_offset @ S
        else:
             return cv_to_gl @ T @ rot_mat @ T_offset @ M_user_rot @ S

    def _get_px(self, lm, w, h): return [int(lm.x * w), int(lm.y * h)]

    def _unproject(self, point_2d, z):
        u, v = point_2d
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)