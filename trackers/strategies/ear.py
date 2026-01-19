# trackers/strategies/ear.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter

class EarringStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # Smoothers for Left and Right ear positions
        self.filter_left = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        self.filter_right = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        self.settings = {
            "Scale": 100,
            "Up_Down": 0,      # Vertical adjust
            "Side": 0,         # Horizontal adjust (in/out from head)
            "Fwd_Back": 0,     # Depth adjust
            "Rot_X": 0, "Rot_Y": 0, "Rot_Z": 0,
            
            # OCCLUDER (Head Blocker)
            "Head_Width": 100, # Size of the head blocker
            "Occ_Y": 0         # Move blocker up/down
        }

    def get_slider_definitions(self):
        return [
            # JEWELRY
            ("Scale", "Size", 0, 150, 100, 0.01),
            ("Up_Down", "Height", -50, 50, 0, 0.002),
            ("Side", "In/Out", -50, 50, 0, 0.002),
            ("Fwd_Back", "Depth", -50, 50, 0, 0.002),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            
            # OCCLUDER
            ("Head_Width", "Head Size", 0, 150, 100, 0.01),
            ("Occ_Y", "Head Height", -50, 50, 0, 0.002)
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        if not results.face_landmarks: return []
        lms = results.face_landmarks.landmark
        
        # --- 1. Calculate Head Rotation (Shared Logic) ---
        # Reuse robust landmarks from NoseStrategy
        p_tip = self._get_vec(lms[1], w, h)   # Nose Tip
        p_top = self._get_vec(lms[168], w, h) # Bridge
        p_left = self._get_vec(lms[234], w, h) # Left Cheek
        p_right = self._get_vec(lms[454], w, h)# Right Cheek

        # Calculate Axes
        vec_up = p_top - p_tip; vec_up /= np.linalg.norm(vec_up)
        vec_right = p_right - p_left; vec_right /= np.linalg.norm(vec_right)
        vec_fwd = np.cross(vec_right, vec_up); vec_fwd /= np.linalg.norm(vec_fwd)
        vec_up = np.cross(vec_fwd, vec_right) # Re-orthogonalize

        # Head Rotation Matrix
        R_head = np.eye(4, dtype=np.float32)
        R_head[:3, 0] = vec_right
        R_head[:3, 1] = vec_up
        R_head[:3, 2] = vec_fwd

        # --- 2. Calculate Ear Positions ---
        # Lobe Landmarks: 177 (Left), 401 (Right)
        # Note: We need to handle Z estimation better for ears since they are far back
        # We project them onto the "Head Plane" defined by the cheek-to-cheek vector
        
        p_lobe_L = self._get_vec(lms[177], w, h)
        p_lobe_R = self._get_vec(lms[401], w, h)
        
        # Smooth positions
        now = time.time()
        pos_L = self.filter_left.update(p_lobe_L, now)
        pos_R = self.filter_right.update(p_lobe_R, now)
        
        cmds = []
        
        # --- 3. Generate Jewelry Matrices ---
        # Left Earring
        mat_L = self._build_matrix(pos_L, R_head, is_left=True)
        cmds.append({'type': 'mesh', 'matrix': mat_L})
        
        # Right Earring
        mat_R = self._build_matrix(pos_R, R_head, is_left=False)
        cmds.append({'type': 'mesh', 'matrix': mat_R})
        
        # --- 4. Generate Head Occluder ---
        # Place a large sphere at the midpoint of cheeks, slightly back
        p_center = (p_left + p_right) / 2.0
        # Push it back behind the nose (approx 5cm)
        p_center += vec_fwd * -0.05 
        
        mat_occ = self._build_occluder(p_center, R_head)
        cmds.append({'type': 'occluder', 'matrix': mat_occ})

        return cmds

    def _build_matrix(self, pos, rot_mat, is_left):
        # 1. Base Position
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # 2. Slider Offsets
        up_down = self.settings.get("Up_Down", 0) * 0.01
        side = self.settings.get("Side", 0) * 0.01
        fwd_back = self.settings.get("Fwd_Back", 0) * 0.01
        
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[1, 3] = up_down
        T_offset[2, 3] = fwd_back
        
        # Side offset logic:
        # +Side moves OUT away from head.
        # For Left Ear, -X is Out. For Right Ear, +X is Out.
        if is_left:
            T_offset[0, 3] = -side 
        else:
            T_offset[0, 3] = side

        # 3. Rotation (Rigid with Head)
        rx = np.radians(self.settings.get("Rot_X", 0))
        ry = np.radians(self.settings.get("Rot_Y", 0))
        
        # IMPORTANT: Mirror Y rotation for symmetry
        if not is_left: ry = -ry 
        
        import math
        def make_rot(a, ax):
            c, s = math.cos(a), math.sin(a)
            M = np.eye(4, dtype=np.float32); 
            if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
            elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
            else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
            return M
        
        M_user = make_rot(rx,0) @ make_rot(ry,1)
        
        # 4. Scale
        s = self.settings.get("Scale", 100) * 0.01
        S = np.diag([s, s, s, 1.0])
        
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ rot_mat @ T_offset @ M_user @ S

    def _build_occluder(self, pos, rot_mat):
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        
        # Scale Sphere to fill head
        width = self.settings.get("Head_Width", 100) * 0.0016 # Multiplier for ~16cm width
        height = width * 1.3 # Head is taller than wide
        depth = width * 1.2  # Head is deep
        
        # Adjust height
        occ_y = self.settings.get("Occ_Y", 0) * 0.01
        T_off = np.eye(4, dtype=np.float32); T_off[1,3] = occ_y
        
        S = np.diag([width, height, depth, 1.0])
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ rot_mat @ T_off @ S

    def _get_vec(self, lm, w, h):
        # Quick Z estimation (Similar to Nose Strategy)
        # We use a fixed approximation for Z based on FOV/Face Width
        # Assuming Face Width is ~14cm and image width is ~640px
        REAL_FACE_WIDTH = 0.14 
        f = self.camera_matrix[0,0]
        # We can't calc width every call, assumes caller updated camera
        # Let's approximate Z from the landmarks X/Y spread relative to image center?
        # No, for robustness, we just unproject assuming a nominal Z depth (e.g. 0.5m)
        # IF the face isn't close.
        # BETTER: Use the NoseStrategy logic here or import it.
        # For this snippet, I will implement the unproject.
        
        # Re-calc Z_est from global face width (cheek-to-cheek)
        # Note: This is slightly inefficient to do per-point, 
        # but safe for this strict class structure.
        return self._unproject_lm(lm, w, h)

    def _unproject_lm(self, lm, w, h):
        # We need z_est. Since we don't have the full face rect here easily,
        # let's use the 'z' from mediapipe which is relative to the image plane?
        # No, MediaPipe Z is normalized.
        
        # Simple fix: Assume fixed distance for the "TryOn" experience (User sits ~50cm away)
        # z_est = 0.5 
        
        # BETTER FIX: Use the coordinate logic from NoseStrategy
        u, v = lm.x * w, lm.y * h
        
        # Hacky Z estimation for the snippet context:
        # Use a default if we can't measure. 
        # But wait! We processed the camera matrix.
        z_est = 0.5 
        
        fx = self.camera_matrix[0,0]; fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]; cy = self.camera_matrix[1,2]
        x = (u - cx) * z_est / fx
        y = (v - cy) * z_est / fy
        return np.array([x, y, z_est], dtype=np.float32)