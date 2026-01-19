# trackers/strategies/neck.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter, RotationFilter

class NeckStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # Smooth the center position to prevent jitter
        self.filter_pos = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        
        self.settings = {
            "Scale": 100,      # Necklace Size
            "Up_Down": 0,      # Vertical Adjust (Neck vs Chest)
            "Fwd_Back": 0,     # Z-Offset (Prevent clipping into skin)
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
            
            # OCCLUDER (The "Bib" Plane)
            "Occ_Width": 100,  # Width of the masking plane
            "Occ_Height": 100  # Height of the masking plane
        }

    def get_slider_definitions(self):
        return [
            # JEWELRY
            ("Scale", "Size", 0, 200, 100, 0.01),
            ("Up_Down", "Height", -100, 100, 0, 0.005),
            ("Fwd_Back", "Depth", -50, 50, 0, 0.005), # Move away/into chest
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            
            # OCCLUDER (Flattened Cylinder -> Plane)
            ("Occ_Width", "Mask W", 0, 150, 100, 0.01),
            ("Occ_Height", "Mask H", 50, 150, 100, 0.01)
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        
        if not results.pose_landmarks: return []

        # MediaPipe Pose: 11=Left Shoulder, 12=Right Shoulder
        lm_l = results.pose_landmarks.landmark[11]
        lm_r = results.pose_landmarks.landmark[12]
        
        if lm_l.visibility < 0.5 or lm_r.visibility < 0.5: return []

        # 1. Pixel Coordinates
        px_l = np.array(self._get_px(lm_l, w, h))
        px_r = np.array(self._get_px(lm_r, w, h))
        
        # 2. Estimate Depth (Z)
        # Avg shoulder width ~ 40cm (0.4m)
        REAL_SHOULDER_WIDTH = 0.40
        px_width = np.linalg.norm(px_l - px_r)
        if px_width < 10: return []
        
        f = self.camera_matrix[0,0]
        z_est = (f * REAL_SHOULDER_WIDTH) / px_width
        
        # 3. Get 3D Points
        p_l = self._unproject(px_l, z_est)
        p_r = self._unproject(px_r, z_est)
        
        # Midpoint (The "Neck Base")
        p_center = (p_l + p_r) / 2.0
        p_smooth = self.filter_pos.update(p_center, time.time())
        
        # 4. Construct Rotation Matrix (Chest Plane)
        # X-Axis: Left to Right Shoulder
        vec_x = p_r - p_l
        vec_x /= np.linalg.norm(vec_x)
        
        # Y-Axis: Down spine. Cross X with Forward(Z) to get Down-ish
        vec_cam_fwd = np.array([0, 0, 1], dtype=np.float32)
        vec_y = np.cross(vec_cam_fwd, vec_x) 
        vec_y /= np.linalg.norm(vec_y)
        
        # Z-Axis: Normal out of chest
        vec_z = np.cross(vec_x, vec_y)
        
        # Build Matrix
        R_chest = np.eye(4, dtype=np.float32)
        R_chest[:3, 0] = vec_x
        R_chest[:3, 1] = vec_y # Pointing Down
        R_chest[:3, 2] = vec_z

        # 5. Build Matrices
        
        # A. Necklace
        mat_jewel = self._build_matrix(p_smooth, R_chest, is_occluder=False)
        render_commands.append({'type': 'mesh', 'matrix': mat_jewel})
        
        # B. Occluder (The Bib)
        # We push it slightly behind the necklace (-Z) so it doesn't hide the front
        mat_occ = self._build_matrix(p_smooth, R_chest, is_occluder=True)
        render_commands.append({'type': 'occluder', 'matrix': mat_occ})

        return render_commands

    def _build_matrix(self, pos, rot_mat, is_occluder):
        # Base Position
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # Offset Logic
        up_down = self.settings.get("Up_Down", 0) * 0.01
        fwd_back = self.settings.get("Fwd_Back", 0) * 0.01
        
        T_offset = np.eye(4, dtype=np.float32)
        # Y is Down, so -Y moves Up towards neck
        # Z is Forward, so +Z moves away from chest
        if is_occluder:
             # Occluder sits slightly behind necklace
             T_offset[1, 3] = up_down 
             T_offset[2, 3] = fwd_back - 0.02 # 2cm behind
        else:
             T_offset[1, 3] = up_down
             T_offset[2, 3] = fwd_back

        # Scale Logic
        if is_occluder:
            # Flatten the cylinder to make a Plane ("The Bib")
            w = self.settings.get("Occ_Width", 100) * 0.002
            h = self.settings.get("Occ_Height", 100) * 0.002
            d = 0.01 # Very thin Z (The Plane effect)
            S = np.diag([w, h, d, 1.0])
        else:
            s = self.settings.get("Scale", 100) * 0.01
            S = np.diag([s, s, s, 1.0])

        # Rotation Logic
        rx = np.radians(self.settings.get("Rot_X", 0))
        # Necklace usually needs -90 X rotation because models come lying flat
        # But we start with 0 for now
        
        import math
        c, s = math.cos(rx), math.sin(rx)
        M_rot = np.eye(4, dtype=np.float32)
        M_rot[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]

        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        if is_occluder:
            return cv_to_gl @ T @ rot_mat @ T_offset @ S
        else:
            return cv_to_gl @ T @ rot_mat @ T_offset @ M_rot @ S

    def _get_px(self, lm, w, h): return [int(lm.x * w), int(lm.y * h)]

    def _unproject(self, point_2d, z):
        u, v = point_2d
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)