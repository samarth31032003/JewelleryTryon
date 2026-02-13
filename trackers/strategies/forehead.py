# trackers/strategies/forehead.py
import numpy as np
import cv2
import time
import math
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter
from trackers.occluder_shared import OccluderManager

class ForeheadStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.filter_pos = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        self.settings = {
            "Scale": 150,
            "Up_Down": 500,    
            "Side": 0,         
            "Fwd_Back": 0,     
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
        }

    def get_slider_definitions(self):
        sliders = [
            # 1. EXPONENTIAL SCALER
            ("Scale", "Size", 1, 500, 150, 1.0),

            # 2. POSITION RATIOS
            ("Up_Down", "Pos (Brows-Hair)", 0, 1000, 500, 1.0), 
            ("Side", "Side (Passa)", -1000, 1000, 0, 1.0),
            ("Fwd_Back", "Depth", -500, 500, 0, 1.0),

            # 3. ROTATION
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
        ]
        
        # 4. SHARED OCCLUDER SLIDERS
        sliders.extend(OccluderManager.get_head_sliders())
        return sliders

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.face_landmarks: return []
        lms = results.face_landmarks.landmark
        
        # --- 1. Get Deep 3D Landmarks ---
        p_tip = self._get_vec(lms[1], w, h)
        p_top = self._get_vec(lms[168], w, h)
        p_left = self._get_vec(lms[234], w, h)
        p_right = self._get_vec(lms[454], w, h)
        
        # --- 2. Calculate Head Rotation ---
        # Standard "3-Vector" Logic
        vec_up = p_top - p_tip; vec_up /= np.linalg.norm(vec_up)
        vec_right = p_right - p_left; vec_right /= np.linalg.norm(vec_right)
        
        vec_fwd = np.cross(vec_right, vec_up); vec_fwd /= np.linalg.norm(vec_fwd)
        vec_up = np.cross(vec_fwd, vec_right)
        
        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right
        R_head[:3, 1] = vec_up
        R_head[:3, 2] = vec_fwd
        
        # --- 3. Calculate Position (Interpolation) ---
        p_hair = self._get_vec(lms[10], w, h)
        p_brow = self._get_vec(lms[151], w, h)
        
        v_ratio = self.settings.get("Up_Down", 500) / 1000.0
        pos_v = p_brow * (1 - v_ratio) + p_hair * v_ratio
        
        side_val = self.settings.get("Side", 0)
        
        if abs(side_val) < 1:
            pos_final = pos_v 
        elif side_val < 0:
            p_temple_L = self._get_vec(lms[127], w, h)
            ratio = abs(side_val) / 1000.0
            pos_final = pos_v * (1 - ratio) + p_temple_L * ratio
        else:
            p_temple_R = self._get_vec(lms[356], w, h)
            ratio = side_val / 1000.0
            pos_final = pos_v * (1 - ratio) + p_temple_R * ratio

        pos_smooth = self.filter_pos.update(pos_final, time.time())
        
        cmds = []
        
        # A. Jewelry Matrix
        cmds.append({'type': 'mesh', 'matrix': self._build_matrix(pos_smooth, R_head)})
        
        # B. SHARED OCCLUDER (Skull)
        # Use center of face so the skull stays stable regardless of where the tikka is
        p_head_center = (p_left + p_right) / 2.0
        mat_occ = OccluderManager.get_head_matrix(self.settings, p_head_center, R_head)
        
        cmds.append({
            'type': 'occluder', 
            'matrix': mat_occ,
            'file_path': OccluderManager.PATH_HEAD,
            'mesh_key': 'occ_head'
        })
        
        return cmds

    def _build_matrix(self, pos, rot_mat):
        # Simplified: No longer handles occluder logic
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        T_offset = np.eye(4, dtype=np.float32)
        
        fwd = self.settings.get("Fwd_Back", 0) * 0.0001
        T_offset[2, 3] = fwd
        
        rx = np.radians(self.settings.get("Rot_X", 0))
        ry = np.radians(self.settings.get("Rot_Y", 0))
        rz = np.radians(self.settings.get("Rot_Z", 0))
        
        def make_rot(a, ax):
            c, s = math.cos(a), math.sin(a)
            M = np.eye(4, dtype=np.float32); 
            if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
            elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
            else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
            return M
        M_user = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)
        
        raw_scale = self.settings.get("Scale", 150)
        s = 0.00001 * (raw_scale ** 2.0)
        S = np.diag([s, s, s, 1.0])

        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        return cv_to_gl @ T @ rot_mat @ T_offset @ M_user @ S

    def _get_vec(self, lm, w, h):
        """
        Converts a MediaPipe landmark into a 3D vector with volume.
        """
        u, v = lm.x * w, lm.y * h
        z_base = 0.5 
        z_rel = lm.z * 0.4 
        z_final = z_base + z_rel
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z_final / fx
        y = (v - cy) * z_final / fy
        return np.array([x, y, z_final], dtype=np.float32)