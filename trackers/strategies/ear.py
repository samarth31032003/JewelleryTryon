# trackers/strategies/ear.py
import numpy as np
import cv2
import time
import math
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter
from trackers.occluder_shared import OccluderManager

class EarringStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.filter_left = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        self.filter_right = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        self.settings = {
            "Scale": 150,      
            "Up_Down": 0,    
            "Side": 0,         
            "Fwd_Back": 0,     
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
        }

    def get_slider_definitions(self):
        sliders = [
            # 1. EXPONENTIAL SCALER
            ("Scale", "Size", 1, 500, 150, 1.0), 

            # 2. HIGH-RES POSITION
            ("Up_Down", "Height", -500, 500, 0, 1.0),
            ("Side", "In/Out", -500, 500, 0, 1.0),
            ("Fwd_Back", "Depth", -500, 500, 0, 1.0),

            # 3. ROTATION
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
        ]
        
        # 4. SHARED OCCLUDER SLIDERS
        # MUST BE HEAD SLIDERS
        if getattr(self, 'mode', '3d') == "3d":
            sliders.extend(OccluderManager.get_head_sliders())
        return sliders

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.face_landmarks: return []
        lms = results.face_landmarks.landmark
        
        # --- 1. Calculate Head Rotation ---
        p_tip = self._get_vec(lms[1], w, h)
        p_top = self._get_vec(lms[168], w, h)
        p_left = self._get_vec(lms[234], w, h)
        p_right = self._get_vec(lms[454], w, h)

        # --- 2. Calculate Ear Positions ---
        p_lobe_L = self._get_vec(lms[177], w, h)
        p_lobe_R = self._get_vec(lms[401], w, h)
        
        now = time.time()
        pos_L = self.filter_left.update(p_lobe_L, now)
        pos_R = self.filter_right.update(p_lobe_R, now)
        
        cmds = []

        # 2d flat tracking.
        if getattr(self, 'mode', '3d') == "2d":
            # Calculate 2D Roll using left and right sides of face
            dy = p_right[1] - p_left[1]
            dx = p_right[0] - p_left[0]
            roll_angle = math.atan2(dy, dx)
            
            user_spin = math.radians(self.settings.get("Rot_Z", 0))
            scale_2d = self.settings.get("Scale", 150) * 0.001 
            
            # Offsets
            off_y = self.settings.get("Up_Down", 0) * 0.0001
            side_off = self.settings.get("Side", 0) * 0.0001

            # Left Earring
            mat_L_2d = self._build_2d_matrix(pos_L, roll_angle + user_spin, scale_2d, offset_x=-side_off, offset_y=off_y)
            cmds.append({'type': 'mesh', 'matrix': mat_L_2d})
            
            # Right Earring
            mat_R_2d = self._build_2d_matrix(pos_R, roll_angle + user_spin, scale_2d, offset_x=side_off, offset_y=off_y)
            cmds.append({'type': 'mesh', 'matrix': mat_R_2d})
            
            return cmds

        # Calculate Axes
        vec_up = p_top - p_tip; vec_up /= np.linalg.norm(vec_up)
        vec_right = p_right - p_left; vec_right /= np.linalg.norm(vec_right)
        
        # Cross Product gives the "True Normal" out of the face
        vec_fwd = np.cross(vec_right, vec_up); vec_fwd /= np.linalg.norm(vec_fwd)
        vec_up = np.cross(vec_fwd, vec_right)

        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right
        R_head[:3, 1] = vec_up
        R_head[:3, 2] = vec_fwd

        # Left Earring
        mat_L = self._build_matrix(pos_L, R_head, is_left=True)
        cmds.append({'type': 'mesh', 'matrix': mat_L})
        
        # Right Earring
        mat_R = self._build_matrix(pos_R, R_head, is_left=False)
        cmds.append({'type': 'mesh', 'matrix': mat_R})
        
        # --- 3. SHARED OCCLUDER ---
        # Calculate Head Center (Midpoint of cheeks)
        p_head_center = (p_left + p_right) / 2.0      
        mat_occ = OccluderManager.get_head_matrix(self.settings, p_head_center, R_head)
        cmds.append({
            'type': 'occluder', 
            'matrix': mat_occ,
            'file_path': OccluderManager.PATH_HEAD,
            'mesh_key': 'occ_head'
        })
        return cmds

    def _build_matrix(self, pos, rot_mat, is_left):
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        
        up_down = self.settings.get("Up_Down", 0) * 0.0001
        side = self.settings.get("Side", 0) * 0.0001
        fwd_back = self.settings.get("Fwd_Back", 0) * 0.0001
        
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[1, 3] = up_down
        T_offset[2, 3] = fwd_back
        
        if is_left: T_offset[0, 3] = -side 
        else: T_offset[0, 3] = side

        # Rotation Controls
        base_y = -90 if is_left else 90
        user_x = self.settings.get("Rot_X", 0)
        user_y = self.settings.get("Rot_Y", 0)
        user_z = self.settings.get("Rot_Z", 0)

        if not is_left: user_y = -user_y

        rx = np.radians(user_x)
        ry = np.radians(base_y + user_y) 
        rz = np.radians(user_z)
        
        def make_rot(a, ax):
            c, s = math.cos(a), math.sin(a)
            M = np.eye(4, dtype=np.float32); 
            if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
            elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
            else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
            return M
        M_rot = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)
        
        # Exponential Scale
        raw_scale = self.settings.get("Scale", 150)
        s = 0.00001 * (raw_scale ** 2.0)
        S = np.diag([s, s, s, 1.0])
        
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        return cv_to_gl @ T @ rot_mat @ T_offset @ M_rot @ S

    def _get_vec(self, lm, w, h):
        u, v = lm.x * w, lm.y * h
        z_base = 0.5 
        
        # Relative Depth from MediaPipe
        # lm.z is normalized. We multiply by 0.4 to give it "Physical Volume".
        # If this is too small, face looks flat. If too big, face looks pointy.
        # 0.4 is a sweet spot for head rotation responsiveness.
        z_rel = lm.z * 0.4 
        z_final = z_base + z_rel
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z_final / fx
        y = (v - cy) * z_final / fy
        return np.array([x, y, z_final], dtype=np.float32)
