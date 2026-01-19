# trackers/strategies/wrist.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter, RotationFilter

class HandState:
    """Helper to store memory for a specific hand"""
    def __init__(self):
        self.filter_rvec = RotationFilter(min_cutoff=0.1, beta=2.0)
        self.filter_tvec = OneEuroFilter(min_cutoff=0.1, beta=2.0)
        self.last_valid_rvec = None
        self.last_valid_tvec = None
        self.last_seen_time = 0

class WristStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        # 3D Models
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
        
        # Initialize Default Settings
        # These match the sliders defined below
        self.settings = {
            "Scale": 100,
            "Slide": 0,
            "Rot_X": 0,
            "Rot_Y": 0,
            "Rot_Z": 0,
            "Occ_Radius": 100, # 100% of calculated thickness
            "Occ_Len": 100     # 100% of bone length
        }

    def get_slider_definitions(self):
        # Key, Label, Min, Max, Default, ScaleFactor
        return [
            ("Scale", "Size", 1, 500, 100, 0.01),
            ("Slide", "Position", -500, 500, 0, 0.5),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
            # OCCLUDER CONTROLS (Merged!)
            ("Occ_Radius", "Mask Thick", 50, 200, 100, 0.01),
            ("Occ_Len", "Mask Length", 50, 150, 100, 0.01)
        ]
    
    def _get_capsule_matrix(self, point_a, point_b, radius_scale=1.0):
        """
        Creates a matrix that stretches a unit cylinder from Point A to Point B.
        point_a, point_b: (3,) numpy arrays (Coordinates in Camera Space)
        """
        # 1. Vector from A to B
        vec = point_b - point_a
        length = np.linalg.norm(vec)
        if length < 1e-6: return np.eye(4)

        # 2. Scale Matrix
        # Scale X/Z by radius, Scale Y by length
        # Base radius is arbitrary, we tune it with radius_scale
        radius = length * 0.15 * radius_scale # Heuristic: Arm is ~15% as thick as it is long
        S = np.diag([radius, length, radius, 1.0])

        # 3. Rotation Matrix (Align Y-axis to Vector)
        # We want to rotate (0,1,0) to match 'vec' direction
        y_axis = np.array([0, 1, 0], dtype=np.float64)
        target_dir = vec / length
        
        # Axis-Angle rotation logic
        rot_axis = np.cross(y_axis, target_dir)
        # rot_angle = np.arccos(np.dot(y_axis, target_dir)) # is it right?
        rot_angle = np.arccos(np.clip(np.dot(y_axis, target_dir), -1.0, 1.0))
        
        if np.linalg.norm(rot_axis) < 1e-6:
            R = np.eye(3)
        else:
            rot_axis /= np.linalg.norm(rot_axis)
            R, _ = cv2.Rodrigues(rot_axis * rot_angle)

        # 4. Translation (Move to Start Point A)
        T = np.eye(4)
        T[:3, 3] = point_a

        # 5. Build Final Matrix
        # M = Translate * Rotate * Scale
        M_rot = np.eye(4); M_rot[:3, :3] = R
        
        # Convert to GL Coords at the very end
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ M_rot @ S

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        
        if not results.pose_landmarks: 
            return []

        # 1. Identify Candidates
        candidates = []
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13, self.model_right, "Right"))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14, self.model_left, "Left"))

        # 2. Process Each Hand
        for hand_lms, elbow_idx, model_3d, label in candidates:
            state = self.states[label]
            
            # --- Check Timeout (Reset if hand was lost for > 1 sec) ---
            if time.time() - state.last_seen_time > 1.0:
                state.last_valid_rvec = None

            # --- Tracking Logic (Same as before) ---
            wrist = self._get_px(hand_lms.landmark[0], w, h)
            index = self._get_px(hand_lms.landmark[5], w, h)
            pinky = self._get_px(hand_lms.landmark[17], w, h)
            middle = self._get_px(hand_lms.landmark[9], w, h)
            elbow_lm = results.pose_landmarks.landmark[elbow_idx]
            if elbow_lm.visibility < 0.5: continue
            elbow = self._get_px(elbow_lm, w, h)

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
                    
                    # --- NEW: Generate 4x4 Matrices Internally ---
                    mat_jewel = self._compute_matrix(rvec_s, tvec_s, is_occluder=False)
                    
                    render_commands.append({
                        'type': 'mesh', 
                        'matrix': mat_jewel,
                        'info': f"{label} Hand"
                    })
        
                    # 1. Reconstruct 3D points of Wrist and Elbow in Camera Space
                    # We have tvec (Wrist Position). We need Elbow.
                    # Since we used PnP, we can just project the 3D Model points using the rvec/tvec!
                    
                    # Get the 3D coordinates of the model joints
                    # Index 0 = Wrist, Index 4 = Elbow (in our self.model_right array)
                    R_mat, _ = cv2.Rodrigues(state.last_valid_rvec)
                    t_vec = state.last_valid_tvec.flatten()
                    
                    p_wrist = (R_mat @ self.model_right[0]) + t_vec
                    p_elbow = (R_mat @ self.model_right[4]) + t_vec
                    
                    # 2. Calculate Capsule Matrix
                    rad_scale = self.settings.get("Occ_Radius", 100) / 100.0

                    # We extend it slightly past the elbow to be safe
                    p_elbow_ext = p_wrist + (p_elbow - p_wrist) * 1.1 
                    len_scale = self.settings.get("Occ_Len", 100) / 100.0
                    
                    # Apply Length Scale to the endpoint
                    p_pip_extended = p_wrist + (p_elbow_ext - p_wrist) * len_scale

                    mat_occ = self._get_capsule_matrix(p_wrist, p_pip_extended, radius_scale=1.0 * rad_scale)
                    
                    # 3. Add to commands
                    render_commands.append({
                        'type': 'occluder',
                        'matrix': mat_occ
                    })

            except cv2.error:
                # Catch the crash if solvePnP fails, reset state to be safe
                state.last_valid_rvec = None
                print(f"Warning: Tracking lost/reset for {label} hand")

        return render_commands

    def _compute_matrix(self, rvec, tvec, is_occluder=False):
        """Converts rvec/tvec + Settings -> 4x4 OpenGL Matrix"""
        # 1. Base Pose (OpenCV)
        R, _ = cv2.Rodrigues(rvec)
        T_base = np.eye(4, dtype=np.float32)
        T_base[:3, :3] = R
        T_base[:3, 3] = tvec.flatten()

        # 2. Coord Swap (CV -> GL)
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        # 3. Apply Sliders (Scale, Slide, Rotate)
        s_val = self.settings.get("Scale", 100) * 0.01
        S = np.diag([s_val, s_val, s_val, 1.0]).astype(np.float32)

        # Slide (Only affects Mesh, not Occluder usually... but let's apply to both for now)
        # Actually, Occluders usually stay fixed to the body part. 
        # Jewelry slides.
        if not is_occluder:
            slide_dist = self.settings.get("Slide", 0) * 0.5
            T_slide = np.eye(4, dtype=np.float32)
            T_slide[:3, 3] = [0, slide_dist, 0] # Y-Axis Slide for Bracelets
            
            # Manual Rotations
            rx = np.radians(self.settings.get("Rot_X", 0))
            ry = np.radians(self.settings.get("Rot_Y", 0))
            rz = np.radians(self.settings.get("Rot_Z", 0))
            # ... (Rotation Matrix Math) ...
            # Simplified rotation construction for brevity:
            import math
            def make_rot(angle, axis):
                c, s = math.cos(angle), math.sin(angle)
                M = np.eye(4, dtype=np.float32)
                if axis==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
                elif axis==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
                else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
                return M
            
            M_rot = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)
            
            # Order: Scale -> UserRot -> Slide -> BodyPose -> FixCoords
            return cv_to_gl @ T_base @ T_slide @ M_rot @ S
        
        else:
            # Occluder just follows the body pose + coord fix
            return cv_to_gl @ T_base

    def _get_px(self, lm, w, h):
        return [int(lm.x * w), int(lm.y * h)]

    def _project_point(self, elbow, wrist, knuckle, ratio):
        e = np.array(elbow); w = np.array(wrist); k = np.array(knuckle)
        arm_vec = w - e
        arm_len = np.linalg.norm(arm_vec)
        if arm_len == 0: return k
        arm_dir = arm_vec / arm_len
        vec_wk = k - w
        parallel = np.dot(vec_wk, arm_dir) * arm_dir
        perp = vec_wk - parallel
        return w + (arm_dir * (arm_len * ratio)) + perp