# trackers/strategies/nose.py
import numpy as np
import cv2
import time
import math
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter
from trackers.occluder_shared import OccluderManager

class NoseStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.filter_pos = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        self.settings = {
            "Scale": 150,
            "Side": 0, "Up_Down": 0, "Fwd_Back": 0,
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
        }

    def get_slider_definitions(self):
        sliders = [
            ("Scale", "Size", 1, 500, 150, 1.0),
            ("Side", "L <-> R", -500, 500, 0, 1.0),  
            ("Up_Down", "Height", -500, 500, 0, 1.0),
            ("Fwd_Back", "Depth", -500, 500, 0, 1.0),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
        ]
        # Append Shared Occluder Sliders
        sliders.extend(OccluderManager.get_head_sliders())
        return sliders

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.face_landmarks: return []
        lms = results.face_landmarks.landmark

        # --- 1. Get Deep 3D Landmarks ---
        # Uses lm.z * 0.4 intensity logic (Same as Ear/Forehead)
        # This prevents shrinking when turning head
        p_tip = self._get_vec(lms[1], w, h)     # Nose Tip
        p_top = self._get_vec(lms[168], w, h)   # Bridge Top
        p_left_cheek = self._get_vec(lms[234], w, h) # Left Cheek
        p_right_cheek = self._get_vec(lms[454], w, h)# Right Cheek
        
        # Nostril Landmarks for sliding
        p_left_nostril = self._get_vec(lms[49], w, h)
        p_right_nostril = self._get_vec(lms[279], w, h)

        # --- 2. Calculate Head Rotation ---
        # Up: Bridge to Tip
        vec_up = p_top - p_tip
        vec_up /= np.linalg.norm(vec_up)
        
        # Right: Left Cheek to Right Cheek
        vec_right = p_right_cheek - p_left_cheek
        vec_right /= np.linalg.norm(vec_right)
        
        # Forward: Cross Product (Right x Up)
        vec_fwd = np.cross(vec_right, vec_up)
        vec_fwd /= np.linalg.norm(vec_fwd)
        
        # Re-orthogonalize Up
        vec_up = np.cross(vec_fwd, vec_right)
        
        # Build Rotation Matrix
        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right
        R_head[:3, 1] = vec_up
        R_head[:3, 2] = vec_fwd

        # --- 3. Calculate Base Position (Curved Interpolation) ---
        side_val = self.settings.get("Side", 0)
        if side_val < 0:
            # Left side (0 to -500 maps to Tip -> Left Nostril)
            ratio = abs(side_val) / 500.0
            pos_base = p_tip * (1 - ratio) + p_left_nostril * ratio
        else:
            # Right side (0 to 500 maps to Tip -> Right Nostril)
            ratio = side_val / 500.0
            pos_base = p_tip * (1 - ratio) + p_right_nostril * ratio

        # Smoothing
        pos_smooth = self.filter_pos.update(pos_base, time.time())
        # --------------------------------------

        cmds = []
        
        # A. Jewelry Matrix
        mat_jewel = self._build_matrix(pos_smooth, R_head)
        cmds.append({'type': 'mesh', 'matrix': mat_jewel})
        
        # B. OCCLUDER (NEW SHARED LOGIC)
        # Let's use Head Center (Midpoint of cheeks + offset) to be safe for all strategies.
        p_head_center = (p_left_cheek + p_right_cheek) / 2.0
        
        mat_occ = OccluderManager.get_head_matrix(self.settings, p_head_center, R_head)
        
        cmds.append({
            'type': 'occluder', 
            'matrix': mat_occ,
            'file_path': OccluderManager.PATH_HEAD,
            'mesh_key': 'occ_head'
        })

        return cmds

    def _build_matrix(self, pos, rot_mat):
        # 1. Base Position
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        up_down = self.settings.get("Up_Down", 0) * 0.0001
        fwd_back = self.settings.get("Fwd_Back", 0) * 0.0001
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[1, 3] = up_down; T_offset[2, 3] = fwd_back
        
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
        
        # Scale (Exponential)
        raw_scale = self.settings.get("Scale", 150)
        s = 0.00001 * (raw_scale ** 2.0)
        S = np.diag([s, s, s, 1.0])
        
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        return cv_to_gl @ T @ rot_mat @ T_offset @ M_user @ S

    def _get_vec(self, lm, w, h):
        """
        Converts MediaPipe landmark to 3D vector with Volume.
        Intensity Multiplier: 0.4
        """
        u, v = lm.x * w, lm.y * h
        z_base = 0.5 
        
        # Relative Depth (Volume Intensity)
        z_rel = lm.z * 0.4 
        z_final = z_base + z_rel
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z_final / fx
        y = (v - cy) * z_final / fy
        return np.array([x, y, z_final], dtype=np.float32)