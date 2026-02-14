# trackers/strategies/waist.py
import numpy as np
import cv2
import math
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter

class WaistStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # Position Smoother
        self.filter_pos = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        # Rotation Smoother (Steadicam logic)
        self.filter_yaw = OneEuroFilter(min_cutoff=0.1, beta=1.0)
        
        self.settings = {
            "Scale": 150,      # Overall Size
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
            ("Scale", "Size", 0, 300, 150, 1.0),
            ("Up_Down", "Height", -200, 200, 0, 1.0),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
            
            # OCCLUDER (Procedural Cylinder)
            ("Occ_Width", "Body Width", 0, 200, 100, 1.0),
            ("Occ_Depth", "Body Thick", 0, 200, 100, 1.0)
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.pose_landmarks: return []

        # MediaPipe Pose Landmarks: 23 = Left Hip, 24 = Right Hip
        lm_l = results.pose_landmarks.landmark[23] # Left Hip
        lm_r = results.pose_landmarks.landmark[24] # Right Hip
        
        # Visibility Check
        if lm_l.visibility < 0.5 or lm_r.visibility < 0.5: return []

        # 1. Pixel Coordinates
        px_l = np.array(self._get_px(lm_l, w, h))
        px_r = np.array(self._get_px(lm_r, w, h))
        
        # 2. INDEPENDENT DEPTH
        # We use a fixed base depth + the AI's relative Z.
        # This decouples depth from rotation.
        Z_BASE = 2.5  # Hips are usually ~2.5m away
        Z_INTENSITY = 0.5 
        
        z_l = Z_BASE + (lm_l.z * Z_INTENSITY)
        z_r = Z_BASE + (lm_r.z * Z_INTENSITY)
        
        # 3. Unproject to 3D Space
        p_l = self._unproject(px_l, z_l)
        p_r = self._unproject(px_r, z_r)
        
        # 4. Position
        p_center = (p_l + p_r) / 2.0
        p_smooth = self.filter_pos.update(p_center, time.time())
        
        # 5. Rotation (Yaw Calculation)
        # Instead of raw vectors, we calculate the Turn Angle (Yaw)
        vec = p_r - p_l
        dx, dz = vec[0], vec[2]
        
        # flip the rotation direction
        raw_yaw = -math.atan2(dz, dx) 
        yaw = self.filter_yaw.update(raw_yaw, time.time())
        
        # Construct Body Matrix from Yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_body = np.eye(4, dtype=np.float32)
        R_body[:3, :3] = [
            [cy,  0, sy],
            [0,   1, 0],
            [-sy, 0, cy]
        ]

        # 6. Build Commands
        cmds = []
        
        # A. Jewelry Mesh
        cmds.append({
            'type': 'mesh', 
            'matrix': self._build_matrix(p_smooth, R_body, False)
        })
        
        # B. Procedural Occluder (No mesh_key = Cylinder)
        cmds.append({
            'type': 'occluder',
            'matrix': self._build_matrix(p_smooth, R_body, True)
        })

        return cmds

    def _build_matrix(self, pos, rot_mat, is_occluder):
        # 1. Base Position
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # 2. Height Offset
        # We use negative Y because in Computer Vision Y is Down, but we want Up.
        up_down = self.settings.get("Up_Down", 0) * 0.002
        T_off = np.eye(4, dtype=np.float32)
        T_off[1, 3] = -up_down 
        
        # 3. Scaling
        if is_occluder:
            # Cylinder Scaling (Width vs Depth)
            # 0.0035 is calibrated to match human scale at Z=2.5m
            w = self.settings.get("Occ_Width", 100) * 0.0035
            d = self.settings.get("Occ_Depth", 100) * 0.0035
            h = 0.4 # Fixed height for torso band
            S = np.diag([w, h, d, 1.0])
            
            # Occluders don't use user rotation (Tilt/Spin), they stick to the body
            cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
            return cv_to_gl @ T @ rot_mat @ T_off @ S
        else:
            # Jewelry Scaling
            # s = self.settings.get("Scale", 150) * 0.0001 * (1.5 ** 2) # Exponential-ish curve
            # Or simpler:
            s = self.settings.get("Scale", 150) * 0.002
            S = np.diag([s, s, s, 1.0])
            
            # User Rotation (Tilt/Spin)
            rx = np.radians(self.settings.get("Rot_X", 0))
            ry = np.radians(self.settings.get("Rot_Y", 0))
            rz = np.radians(self.settings.get("Rot_Z", 0))
            
            def make_rot(a, ax):
                c, s = math.cos(a), math.sin(a)
                M = np.eye(4, dtype=np.float32)
                if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
                elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
                else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
                return M
            
            M_user = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)
            
            cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
            return cv_to_gl @ T @ rot_mat @ T_off @ M_user @ S

    def _get_px(self, lm, w, h): return [int(lm.x * w), int(lm.y * h)]

    def _unproject(self, point_2d, z):
        u, v = point_2d
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)