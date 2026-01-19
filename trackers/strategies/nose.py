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
        render_commands = []
        
        # Check Face Landmarks
        if not results.face_landmarks: 
            return []
        
        lms = results.face_landmarks.landmark
        
        # Key Landmarks (MediaPipe Face Mesh)
        # 1 = Nose Tip
        # 168 = Midpoint between eyes (Top of nose bridge)
        # 49 = Left Alar (Left Nostril outer edge)
        # 279 = Right Alar (Right Nostril outer edge)
        
        # 1. Calculate Head Rotation Matrix
        # Y-Axis (Up): Chin(152) to Forehead(10) ?? No, let's use Nose Bridge
        # Up Vector: Tip(1) to TopBridge(168)
        p_tip = self._get_vec(lms[1], w, h)
        p_top = self._get_vec(lms[168], w, h)
        p_left = self._get_vec(lms[49], w, h)
        p_right = self._get_vec(lms[279], w, h)
        
        # Calculate Vectors
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

        # 2. Calculate Base Position (Interpolate Left <-> Right)
        # We create a line from Left Nostril -> Center -> Right Nostril
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

        # 3. Build Matrices
        
        # A. Jewelry Matrix
        mat_jewel = self._build_matrix(pos_smooth, R_head, is_occluder=False)
        render_commands.append({'type': 'mesh', 'matrix': mat_jewel})
        
        # B. Occluder (Nose Bulb)
        # We place this at the TIP (p_tip), not the jewelry position
        # Because the nose is at the center, even if jewelry is on the side.
        mat_occ = self._build_matrix(p_tip, R_head, is_occluder=True)
        render_commands.append({'type': 'occluder', 'matrix': mat_occ})

        return render_commands

    def _build_matrix(self, pos, rot_mat, is_occluder):
        # 1. Base Position
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # 2. Offsets
        T_offset = np.eye(4, dtype=np.float32)
        
        if is_occluder:
            # Occluder specific adjustments
            occ_y = self.settings.get("Occ_Y", 0) * 0.01
            # Move occluder slightly UP/DOWN relative to nose tip
            T_offset[1, 3] = occ_y
            
            # Scale (The "Bulb Size")
            scale = self.settings.get("Occ_Size", 100) * 0.015 # Base size multiplier
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

    def _get_vec(self, lm, w, h):
        """Converts landmark to 3D vector with Z estimation"""
        # Estimate Z based on face width or use landmark Z?
        # MediaPipe FaceMesh Z is relative to head center.
        # We need Camera Space Z.
        # Simple hack: Use PnP-like depth estimation or fixed depth.
        # Better: FaceMesh landmarks[1] (Tip) vs landmarks[234/454] (Cheeks)
        
        # Quick Z estimation using Iris distance or Face Width?
        # Let's use a standard approximation similar to Waist/Neck
        # Assuming Face Width is ~14cm
        
        px_x = int(lm.x * w)
        px_y = int(lm.y * h)
        
        # We can pass the calculated Z from outside, but for now let's approx:
        # We really should calculate Z once per frame in process_frame, but...
        # Let's rely on the relative Z from mediapipe scaled by a factor
        
        # Just use unproject with a fixed Z for now, 
        # relying on the matrix math to handle relative positions.
        # Or better: Calculate "z_est" in process_frame and pass it down.
        # For simplicity in this snippet, I'll recalculate z_est inside process_frame 
        # and pass it here? No, let's keep it simple.
        
        # Let's grab Z from the Landmark itself (it has x,y,z)
        # But LM Z is not in meters. It's relative scale.
        
        # REVISIT: Let's do the standard depth calc we did for other trackers.
        return np.array([0,0,0]) # Placeholder, see updated process_frame logic

    # OVERRIDING process_frame to handle Z properly
    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.face_landmarks: return []
        lms = results.face_landmarks.landmark
        
        # 1. Calc Depth
        px_l = np.array([lms[234].x * w, lms[234].y * h])
        px_r = np.array([lms[454].x * w, lms[454].y * h])
        width_px = np.linalg.norm(px_l - px_r)
        if width_px < 1: return []
        
        REAL_FACE_WIDTH = 0.14 # 14cm
        f = self.camera_matrix[0,0]
        z_est = (f * REAL_FACE_WIDTH) / width_px
        
        # Helper to get 3D point
        def get_p(idx):
            u, v = lms[idx].x * w, lms[idx].y * h
            # MediaPipe Z is roughly same scale as X (relative to image width)
            # We add it to our Z_est
            # rel_z = lms[idx].z * w # Scale relative Z by image width to get pixels?
            # Actually, let's just use the unproject logic, ignoring local Z for now
            # as it's flat enough for nose pins.
            return self._unproject([u,v], z_est)
            
        p_tip = get_p(1)
        p_top = get_p(168)
        p_left = get_p(49)
        p_right = get_p(279)
        
        # ... (Rest of rotation logic uses these p_ variables) ...
        # (Copy the Rotation/Matrix logic from above, using these p_ vectors)
        
        # [REPEATING LOGIC FROM PREVIOUS SNIPPET WITH CORRECT POINTS]
        vec_up = p_top - p_tip; vec_up /= np.linalg.norm(vec_up)
        vec_right = p_right - p_left; vec_right /= np.linalg.norm(vec_right)
        vec_fwd = np.cross(vec_right, vec_up); vec_fwd /= np.linalg.norm(vec_fwd)
        vec_up = np.cross(vec_fwd, vec_right)
        
        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right; R_head[:3, 1] = vec_up; R_head[:3, 2] = vec_fwd

        side_val = self.settings.get("Side", 50)
        if side_val < 50:
            ratio = side_val / 50.0
            pos_base = p_left * (1 - ratio) + p_tip * ratio
        else:
            ratio = (side_val - 50) / 50.0
            pos_base = p_tip * (1 - ratio) + p_right * ratio
            
        pos_smooth = self.filter_pos.update(pos_base, time.time())
        
        cmds = []
        cmds.append({'type': 'mesh', 'matrix': self._build_matrix(pos_smooth, R_head, False)})
        cmds.append({'type': 'occluder', 'matrix': self._build_matrix(p_tip, R_head, True)}) # Occluder at TIP
        return cmds

    def _unproject(self, point_2d, z):
        u, v = point_2d
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)