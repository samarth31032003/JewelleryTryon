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
        # 1. Position Smoother (Center of Necklace)
        self.filter_pos = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        
        # 2. Angle Smoothers (The "Steadicam")
        # We smooth the Angles directly, not the vectors.
        # This prevents the "Shaking" when MediaPipe flickers.
        self.filter_yaw = OneEuroFilter(min_cutoff=0.1, beta=2.0)
        self.filter_roll = OneEuroFilter(min_cutoff=0.1, beta=2.0)
        
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
        # MUST BE BODY SLIDERS
        sliders.extend(OccluderManager.get_body_sliders())
        return sliders

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        
        if not results.pose_landmarks: return []

        lm_l = results.pose_landmarks.landmark[11] # Left Shoulder
        lm_r = results.pose_landmarks.landmark[12] # Right Shoulder
        
        if lm_l.visibility < 0.5 or lm_r.visibility < 0.5: return []

        # 1. Pixel Coordinates
        px_l = np.array(self._get_px(lm_l, w, h))
        px_r = np.array(self._get_px(lm_r, w, h))
        
        # 2. Independent Depth Calculation
        Z_BASE = 0.5 
        # REDUCED INTENSITY: 0.3 is the sweet spot. 
        # 0.5 was swinging too wide. 0.3 is tighter.
        Z_INTENSITY = 0.3 
        
        z_l = Z_BASE + (lm_l.z * Z_INTENSITY)
        z_r = Z_BASE + (lm_r.z * Z_INTENSITY)
        
        # 3. Get 3D Points
        p_l = self._unproject(px_l, z_l)
        p_r = self._unproject(px_r, z_r)
        
        # 4. Smooth Position (Neck Base)
        p_center = (p_l + p_r) / 2.0
        p_smooth = self.filter_pos.update(p_center, time.time())
        
        # 5. ANGLE-BASED TRACKING (The Fix)
        # Instead of Vector Cross Products (which shake), we calculate Euler Angles.
        
        # Vector from Left Shoulder to Right Shoulder
        vec = p_r - p_l
        dx, dy, dz = vec[0], vec[1], vec[2]
        
        # Calculate Raw Angles
        # Yaw (Turning Head): Rotation around Y-axis (XZ plane)
        # We use -dz because OpenGL Z is backwards relative to our math sometimes, 
        # but atan2(dz, dx) is standard. Let's stick to standard trig.
        raw_yaw = math.atan2(dz, dx)
        
        # Roll (Tilting Head Sideways): Rotation around Z-axis (XY plane)
        raw_roll = math.atan2(dy, dx)
        
        # Smooth the Angles
        now = time.time()
        yaw_smooth = self.filter_yaw.update(raw_yaw, now)
        roll_smooth = self.filter_roll.update(raw_roll, now)
        
        # Reconstruct Rotation Matrix from Smoothed Angles
        # We construct the "Shoulder Rotation" matrix manually
        # Y-Axis Rotation (Yaw)
        cy, sy = math.cos(yaw_smooth), math.sin(yaw_smooth)
        Ry = np.eye(4, dtype=np.float32)
        Ry[:3, :3] = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
        
        # Z-Axis Rotation (Roll)
        cr, sr = math.cos(roll_smooth), math.sin(roll_smooth)
        Rz = np.eye(4, dtype=np.float32)
        Rz[:3, :3] = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
        
        # Combine: First Yaw (Turn), then Roll (Tilt)
        R_chest = Ry @ Rz

        # 6. Build Matrices
        
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
        # "Gravity Slope" Method (Method A)
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