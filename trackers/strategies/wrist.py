# trackers/strategies/wrist.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter, RotationFilter

class HandState:
    """Helper to store memory for a specific hand (Left or Right)"""
    def __init__(self):
        self.filter_rvec = RotationFilter(min_cutoff=0.1, beta=2.0)
        self.filter_tvec = OneEuroFilter(min_cutoff=0.1, beta=2.0)
        self.last_valid_rvec = None
        self.last_valid_tvec = None
        self.last_seen_time = 0

class WristStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        
        # 3D Models (Vertices)
        self.model_right = np.array([
            [0, 0.025, 0],         # WRIST
            [-2.5, 6.0, 0.0],      # INDEX_MCP
            [2.5, 5.5, 0.0],       # PINKY_MCP
            [0.0, 6.0, 0.0],       # MIDDLE_MCP
            [0.0, -25.0, 0.0],     # ELBOW
        ], dtype=np.float64)

        self.model_left = self.model_right.copy()
        self.model_left[:, 0] *= -1 
        
        # State Container: Keys are "Left" and "Right"
        self.states = {
            "Left": HandState(),
            "Right": HandState()
        }

    def calculate_pose(self, results, w, h):
        """Returns a LIST of results: [(rvec, tvec, info), ...]"""
        self.update_camera(w, h)
        active_poses = []
        
        if not results.pose_landmarks: 
            return []

        # 1. Define Candidates to process
        candidates = []
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13, self.model_right, "Right"))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14, self.model_left, "Left"))

        # 2. Loop through ALL candidates (Supports 2 hands at once)
        for hand_lms, elbow_idx, model_3d, label in candidates:
            state = self.states[label]
            
            # --- Check Timeout (Reset if hand was lost for > 1 sec) ---
            if time.time() - state.last_seen_time > 1.0:
                state.last_valid_rvec = None
                state.last_valid_tvec = None

            # --- Extract Points ---
            wrist = self._get_px(hand_lms.landmark[0], w, h)
            index = self._get_px(hand_lms.landmark[5], w, h)
            pinky = self._get_px(hand_lms.landmark[17], w, h)
            middle = self._get_px(hand_lms.landmark[9], w, h)
            
            # Elbow from Pose
            elbow_lm = results.pose_landmarks.landmark[elbow_idx]
            if elbow_lm.visibility < 0.5: continue
            elbow = self._get_px(elbow_lm, w, h)

            # --- Fix Geometry custom straightening logic ---
            fixed_index = self._project_point(elbow, wrist, index, 0.24)
            fixed_pinky = self._project_point(elbow, wrist, pinky, 0.24)
            fixed_middle = self._project_point(elbow, wrist, middle, 0.24)
            
            image_points = np.array([wrist, fixed_index, fixed_pinky, fixed_middle, elbow], dtype=np.float64)

            # --- Solve PnP (Using THIS hand's memory) ---
            try:
                success, rvec, tvec = cv2.solvePnP(
                    model_3d, image_points, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=(state.last_valid_rvec is not None),
                    rvec=state.last_valid_rvec, 
                    tvec=state.last_valid_tvec
                )

                if success:
                    # Apply Filter
                    now = time.time()
                    rvec_s = state.filter_rvec.update(rvec.flatten(), now)
                    tvec_s = state.filter_tvec.update(tvec.flatten(), now).reshape(3,1)
                    
                    # Update State
                    state.last_valid_rvec = rvec
                    state.last_valid_tvec = tvec
                    state.last_seen_time = now
                    
                    # Add to results
                    info = {"found": True, "hand": label, "z_depth": tvec_s[2][0]}
                    active_poses.append((rvec_s, tvec_s, info))
            
            except cv2.error:
                # Catch the crash if solvePnP fails, reset state to be safe
                state.last_valid_rvec = None
                print(f"Warning: Tracking lost/reset for {label} hand")

        return active_poses

    def _get_px(self, lm, w, h):
        return [int(lm.x * w), int(lm.y * h)]

    def _project_point(self, elbow, wrist, knuckle, ratio):
        # (Same math as before)
        e = np.array(elbow); w = np.array(wrist); k = np.array(knuckle)
        arm_vec = w - e
        arm_len = np.linalg.norm(arm_vec)
        if arm_len == 0: return k
        arm_dir = arm_vec / arm_len
        vec_wk = k - w
        parallel = np.dot(vec_wk, arm_dir) * arm_dir
        perp = vec_wk - parallel
        return w + (arm_dir * (arm_len * ratio)) + perp