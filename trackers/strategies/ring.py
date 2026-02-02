# trackers/strategies/ring.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter, RotationFilter

class HandState:
    def __init__(self):
        self.filter_rvec = RotationFilter(min_cutoff=0.1, beta=2.0)
        self.filter_tvec = OneEuroFilter(min_cutoff=0.1, beta=2.0)
        self.last_valid_rvec = None
        self.last_valid_tvec = None
        self.last_seen_time = 0

class RingStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # Use the same Hand Model as WristStrategy for stable PnP
        self.model_right = np.array([
            [0, 0.025, 0],         # WRIST
            [-2.5, 6.0, 0.0],      # INDEX_MCP
            [2.5, 5.5, 0.0],       # PINKY_MCP
            [0.0, 6.0, 0.0],       # MIDDLE_MCP
            [0.0, -25.0, 0.0],     # ELBOW
        ], dtype=np.float64)
        self.model_left = self.model_right.copy()
        self.model_left[:, 0] *= -1 
        
        self.states = {"Left": HandState(), "Right": HandState()}

        # SETTINGS FOR RINGS
        # Note: 'Finger' is an index: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
        self.settings = {
            "Scale": 100, 
            "Slide": 50,    # 50% = Center of the bone
            "Finger": 3,    # Default to Ring Finger
            "Rot_X": 90,    # Rings usually face 'forward' (90deg tilt)
            "Rot_Y": 0, 
            "Rot_Z": 0,
            "Occ_Radius": 100, # 100% of calculated thickness
            "Occ_Len": 100     # 100% of bone length
        }

    def get_slider_definitions(self):
        return [
            ("Finger", "Finger (0=Thumb)", 0, 4, 3, 1.0),
            ("Scale", "Size", 1, 300, 100, 0.01),
            ("Slide", "Position %", 0, 100, 50, 0.01), # 0 to 1 along bone
            ("Rot_X", "Tilt X", -180, 180, 90, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
            # OCCLUDER CONTROLS (Merged!)
            ("Occ_Radius", "Mask Thick", 50, 200, 100, 0.01),
            ("Occ_Len", "Mask Length", 50, 150, 100, 0.01)
        ]

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        if not results.pose_landmarks: return []

        # 1. Identify Candidates (Same as Wrist)
        candidates = []
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13, self.model_right, "Right"))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14, self.model_left, "Left"))

        for hand_lms, elbow_idx, model_3d, label in candidates:
            state = self.states[label]
            if time.time() - state.last_seen_time > 1.0: state.last_valid_rvec = None

            # --- Solve PnP (Same generic hand tracking) ---
            # We track the whole hand first to get 3D orientation
            wrist = self._get_px(hand_lms.landmark[0], w, h)
            index = self._get_px(hand_lms.landmark[5], w, h)
            pinky = self._get_px(hand_lms.landmark[17], w, h)
            middle = self._get_px(hand_lms.landmark[9], w, h)
            elbow_lm = results.pose_landmarks.landmark[elbow_idx]
            if elbow_lm.visibility < 0.5: continue
            elbow = self._get_px(elbow_lm, w, h)
            
            # Geometry Fix
            fixed_index = self._project_point(elbow, wrist, index, 0.24)
            fixed_pinky = self._project_point(elbow, wrist, pinky, 0.24)
            fixed_middle = self._project_point(elbow, wrist, middle, 0.24)
            image_points = np.array([wrist, fixed_index, fixed_pinky, fixed_middle, elbow], dtype=np.float64)

            try:
                success, rvec, tvec = cv2.solvePnP(
                    model_3d, image_points, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=(state.last_valid_rvec is not None),
                    rvec=state.last_valid_rvec, tvec=state.last_valid_tvec
                )

                if success:
                    now = time.time()
                    rvec_s = state.filter_rvec.update(rvec.flatten(), now)
                    tvec_s = state.filter_tvec.update(tvec.flatten(), now).reshape(3,1)
                    state.last_valid_rvec = rvec_s
                    state.last_valid_tvec = tvec_s
                    state.last_seen_time = now

                    # --- RING SPECIFIC LOGIC STARTS HERE ---
                    
                    # 1. Determine Which Finger?
                    f_idx = int(self.settings.get("Finger", 3))
                    # MediaPipe Landmark Indices:
                    # Thumb: 1-2, Index: 5-6, Middle: 9-10, Ring: 13-14, Pinky: 17-18
                    # Map 0-4 slider to MCP index
                    mcp_indices = [1, 5, 9, 13, 17] 
                    pip_indices = [2, 6, 10, 14, 18]
                    
                    mcp_idx = mcp_indices[f_idx]
                    pip_idx = pip_indices[f_idx]

                    # 2. Get 3D Coordinates of that Finger
                    # We can't rely on the hand model for fingers (it doesn't have them all).
                    # We must "Deproject" the 2D MediaPipe landmarks using the Depth we found from PnP.
                    
                    # Approximating 3D position using the Hand Plane
                    # (Simplified: We use PnP rotation to orient, and approximate finger relative to wrist)
                    # BETTER WAY: Just use the 2D landmarks and 'fake' the Z based on the hand tilt?
                    # NO, we want robust 3D.
                    
                    # Let's use the Raw MediaPipe 3D Landmarks (World Landmarks) if available? 
                    # They are relative to hip. PnP is relative to Camera.
                    # Hybrid Approach: Use PnP R/T to place the Wrist, then add the relative hand structure.
                    
                    # Let's just use the Hand Model's transformation to get the WRIST position,
                    # and assume the fingers are in standard pose? No, fingers move.
                    
                    # SIMPLEST ROBUST METHOD:
                    # Use the 2D landmarks for X/Y. Use the PnP Tvec for Z.
                    # It's an approximation, but works for "Try On" visualization.
                    
                    mcp_2d = self._get_px(hand_lms.landmark[mcp_idx], w, h)
                    pip_2d = self._get_px(hand_lms.landmark[pip_idx], w, h)
                    
                    # Back-project 2D -> 3D using estimated Z
                    z_est = tvec_s[2][0] # Z distance from camera
                    
                    # Unproject
                    p_mcp = self._unproject(mcp_2d, z_est)
                    p_pip = self._unproject(pip_2d, z_est)
                    
                    # 3. Calculate Matrices

                    # Derive palm normal (roll hint) from PnP hand pose so the ring
                    # rotates consistently with the hand's orientation.
                    R_hand, _ = cv2.Rodrigues(rvec_s)
                    t_hand = tvec_s.flatten()
                    # Use the same model_3d provided for this hand to get palm basis
                    p_wrist_cam = (R_hand @ model_3d[0]) + t_hand
                    p_index_cam = (R_hand @ model_3d[1]) + t_hand
                    p_pinky_cam = (R_hand @ model_3d[2]) + t_hand
                    p_middle_cam = (R_hand @ model_3d[3]) + t_hand

                    hand_side = p_pinky_cam - p_index_cam
                    hs_len = np.linalg.norm(hand_side)
                    if hs_len > 1e-6:
                        hand_side /= hs_len
                    else:
                        hand_side = np.array([1.0, 0.0, 0.0], dtype=np.float64)

                    hand_forward = p_middle_cam - p_wrist_cam
                    hf_len = np.linalg.norm(hand_forward)
                    if hf_len > 1e-6:
                        hand_forward /= hf_len
                    else:
                        hand_forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)

                    palm_normal = np.cross(hand_side, hand_forward)
                    pn_len = np.linalg.norm(palm_normal)
                    if pn_len > 1e-6:
                        palm_normal /= pn_len
                    else:
                        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                    # For the mirrored hand, flip the normal so the ring front faces outward.
                    if label == "Left":
                        palm_normal = -palm_normal

                    # A. The Ring (Mesh)
                    # Align Y axis to vector (MCP -> PIP) and set roll around that axis
                    # using the palm normal projected onto the plane perpendicular to the finger.
                    mat_jewel = self._get_aligned_matrix(p_mcp, p_pip, roll_hint=palm_normal)
                    # 3. Occluder Calculation
                    rad_scale = self.settings.get("Occ_Radius", 100) / 100.0
                    len_scale = self.settings.get("Occ_Len", 100) / 100.0
                    
                    # Extend the bone slightly based on slider
                    p_pip_extended = p_mcp + (p_pip - p_mcp) * len_scale
                    
                    # B. The Occluder (Bone)
                    # Stretch cylinder from MCP to PIP
                    mat_occ = self._get_capsule_matrix(p_mcp, p_pip_extended, radius_scale=0.9 * rad_scale)                    
                    render_commands.append({'type': 'mesh', 'matrix': mat_jewel})
                    render_commands.append({'type': 'occluder', 'matrix': mat_occ})

            except Exception as e:
                # print(e)
                pass

        return render_commands

    def _unproject(self, point_2d, z):
        """Converts 2D pixel + Z depth -> 3D Camera Coordinate"""
        u, v = point_2d
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    def _get_aligned_matrix(self, p_a, p_b, roll_hint=None):
        """Aligns object to vector A->B and applies roll using roll_hint.
        roll_hint: a 3D direction (e.g., palm normal) used to orient the ring's spin
        consistently with the hand orientation. If None, defaults to minimal-rotation alignment.
        """
        vec = p_b - p_a
        length = np.linalg.norm(vec)
        if length < 1e-6: return np.eye(4)
        
        # Position: Slide along vector
        slide_pct = self.settings.get("Slide", 50) / 100.0
        pos = p_a + (vec * slide_pct)
        
        # Rotation: Build basis with Y aligned to vector and X/Z derived from roll_hint
        y_dir = vec / length
        if roll_hint is not None:
            # Project roll_hint onto plane perpendicular to y_dir
            roll_proj = roll_hint - np.dot(roll_hint, y_dir) * y_dir
            rp_len = np.linalg.norm(roll_proj)
            if rp_len > 1e-6:
                z_dir = roll_proj / rp_len  # choose Z along projected palm normal
            else:
                # Fallback: arbitrary perpendicular
                z_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if abs(np.dot(z_dir, y_dir)) > 0.9:
                    z_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_dir = np.cross(z_dir, y_dir)
            xl = np.linalg.norm(x_dir)
            if xl > 1e-6:
                x_dir /= xl
            else:
                # Fallback to orthonormal completion
                x_dir = np.cross(y_dir, np.array([0.0, 0.0, 1.0]))
                x_dir /= np.linalg.norm(x_dir)
            # Recompute z to ensure orthonormal right-handed basis
            z_dir = np.cross(x_dir, y_dir)
            z_dir /= np.linalg.norm(z_dir)
            R = np.stack([x_dir, y_dir, z_dir], axis=1)
        else:
            # Minimal rotation alignment (no roll preservation)
            y_axis = np.array([0, 1, 0], dtype=np.float64)
            target_dir = y_dir
            rot_axis = np.cross(y_axis, target_dir)
            rot_angle = np.arccos(np.clip(np.dot(y_axis, target_dir), -1.0, 1.0))
            if np.linalg.norm(rot_axis) < 1e-6:
                R = np.eye(3)
            else:
                rot_axis /= np.linalg.norm(rot_axis)
                R, _ = cv2.Rodrigues(rot_axis * rot_angle)

        # User Rotation adjustments (Local)
        rx = np.radians(self.settings.get("Rot_X", 0))
        ry = np.radians(self.settings.get("Rot_Y", 0))
        rz = np.radians(self.settings.get("Rot_Z", 0))
        # (Simplified Euler rotation)
        import math
        def make_rot(a, ax):
            c, s = math.cos(a), math.sin(a)
            M = np.eye(4); 
            if ax==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
            elif ax==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
            else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
            return M
        M_user = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)

        # Scale (Uniform)
        s_val = self.settings.get("Scale", 100) * 0.005 # Smaller base scale for rings
        S = np.diag([s_val, s_val, s_val, 1.0])

        # Build Matrix: Scale -> UserRot -> AlignRot -> Translate
        T = np.eye(4); T[:3, 3] = pos
        M_align = np.eye(4); M_align[:3, :3] = R
        
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        return cv_to_gl @ T @ M_align @ M_user @ S

    def _get_capsule_matrix(self, point_a, point_b, radius_scale=1.0):
        # (Copy this helper from wrist.py, or ideally move it to base.py so both can share!)
        # For now, duplicate it here for safety.
        vec = point_b - point_a
        length = np.linalg.norm(vec)
        if length < 1e-6: return np.eye(4)
        radius = length * 0.25 * radius_scale # Fingers are fatter relative to length
        S = np.diag([radius, length, radius, 1.0])
        y_axis = np.array([0, 1, 0], dtype=np.float64)
        target_dir = vec / length
        rot_axis = np.cross(y_axis, target_dir)
        rot_angle = np.arccos(np.clip(np.dot(y_axis, target_dir), -1.0, 1.0))
        if np.linalg.norm(rot_axis) < 1e-6: R = np.eye(3)
        else: rot_axis /= np.linalg.norm(rot_axis); R, _ = cv2.Rodrigues(rot_axis * rot_angle)
        T = np.eye(4); T[:3, 3] = point_a
        M_rot = np.eye(4); M_rot[:3, :3] = R
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        return cv_to_gl @ T @ M_rot @ S

    def _get_px(self, lm, w, h): return [int(lm.x * w), int(lm.y * h)]
    
    def _project_point(self, elbow, wrist, knuckle, ratio):
        # (Copy from wrist.py or move to utils)
        e = np.array(elbow); w = np.array(wrist); k = np.array(knuckle)
        arm_vec = w - e; arm_len = np.linalg.norm(arm_vec)
        if arm_len == 0: return k
        arm_dir = arm_vec / arm_len; vec_wk = k - w
        parallel = np.dot(vec_wk, arm_dir) * arm_dir; perp = vec_wk - parallel
        return w + (arm_dir * (arm_len * ratio)) + perp