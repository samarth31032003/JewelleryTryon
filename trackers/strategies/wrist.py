# trackers/strategies/wrist.py
import numpy as np
import cv2
import time
from .base import TrackingStrategy
from utils.smoothing import OneEuroFilter, RotationFilter
from utils.logger import logger

log = logger.bind(component="trackers")

class HandState:
    """Helper to store memory for a specific hand"""
    def __init__(self):
        # High Beta = Fast response when moving. High min_cutoff = No jitter when resting.
        self.filter_rvec = RotationFilter(min_cutoff=0.4, beta=0.8)
        self.filter_tvec = OneEuroFilter(min_cutoff=0.5, beta=0.7)
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
        return [
            ("Scale", "Size", 1, 500, 100, 0.01),
            ("Slide", "Position", -500, 500, 0, 0.5),
            ("Rot_X", "Tilt X", -180, 180, 0, 1.0),
            ("Rot_Y", "Tilt Y", -180, 180, 0, 1.0),
            ("Rot_Z", "Spin", -180, 180, 0, 1.0),
            ("Occ_Radius", "Mask Thick", 50, 200, 100, 0.01),
            ("Occ_Len", "Mask Length", 50, 150, 100, 0.01)
        ]
    
    def _get_capsule_matrix(self, point_a, point_b, radius_scale=1.0):
        vec = point_b - point_a
        length = np.linalg.norm(vec)
        if length < 1e-6: return np.eye(4)

        radius = length * 0.15 * radius_scale
        S = np.diag([radius, length, radius, 1.0])

        y_axis = np.array([0, 1, 0], dtype=np.float64)
        target_dir = vec / length
        
        rot_axis = np.cross(y_axis, target_dir)
        rot_angle = np.arccos(np.clip(np.dot(y_axis, target_dir), -1.0, 1.0))
        
        if np.linalg.norm(rot_axis) < 1e-6:
            R = np.eye(3)
        else:
            rot_axis /= np.linalg.norm(rot_axis)
            R, _ = cv2.Rodrigues(rot_axis * rot_angle)

        T = np.eye(4)
        T[:3, 3] = point_a

        M_rot = np.eye(4); M_rot[:3, :3] = R
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ M_rot @ S

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        render_commands = []
        
        if not results.pose_landmarks: 
            return []

        candidates = []
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13, self.model_right, "Right"))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14, self.model_left, "Left"))

        for hand_lms, elbow_idx, model_3d, label in candidates:
            state = self.states[label]
            
            if time.time() - state.last_seen_time > 1.0:
                state.last_valid_rvec = None

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
                    # Directly use the high-quality EuroFilter values without lagging them!
                    rvec_s = state.filter_rvec.update(rvec.flatten(), now)
                    tvec_s = state.filter_tvec.update(tvec.flatten(), now).reshape(3,1)
                    
                    state.last_valid_rvec = rvec_s
                    state.last_valid_tvec = tvec_s
                    state.last_seen_time = now
                    
                    mat_jewel = self._compute_matrix(rvec_s, tvec_s, is_occluder=False)
                    
                    render_commands.append({
                        'type': 'mesh', 
                        'matrix': mat_jewel,
                        'info': f"{label} Hand"
                    })
        
                    R_mat, _ = cv2.Rodrigues(rvec_s)
                    t_vec = tvec_s.flatten()
                    
                    p_wrist = (R_mat @ self.model_right[0]) + t_vec
                    p_elbow = (R_mat @ self.model_right[4]) + t_vec
                    
                    rad_scale = self.settings.get("Occ_Radius", 100) / 100.0
                    p_elbow_ext = p_wrist + (p_elbow - p_wrist) * 1.1 
                    len_scale = self.settings.get("Occ_Len", 100) / 100.0
                    p_pip_extended = p_wrist + (p_elbow_ext - p_wrist) * len_scale

                    mat_occ = self._get_capsule_matrix(p_wrist, p_pip_extended, radius_scale=1.0 * rad_scale)
                    
                    render_commands.append({
                        'type': 'occluder',
                        'matrix': mat_occ
                    })

            except cv2.error:
                state.last_valid_rvec = None
                log.warning(f"Warning: Tracking lost/reset for {label} hand")

        return render_commands

    def _compute_matrix(self, rvec, tvec, is_occluder=False):
        R, _ = cv2.Rodrigues(rvec)
        T_base = np.eye(4, dtype=np.float32)
        T_base[:3, :3] = R
        T_base[:3, 3] = tvec.flatten()

        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        s_val = self.settings.get("Scale", 100) * 0.01
        S = np.diag([s_val, s_val, s_val, 1.0]).astype(np.float32)

        slide_dist = self.settings.get("Slide", 0) * 0.5 if not is_occluder else 0
        T_slide = np.eye(4, dtype=np.float32)
        T_slide[:3, 3] = [0, slide_dist, 0]

        rx = np.radians(self.settings.get("Rot_X", 0))
        ry = np.radians(self.settings.get("Rot_Y", 0))
        rz = np.radians(self.settings.get("Rot_Z", 0))
        import math
        def make_rot(angle, axis):
            c, s = math.cos(angle), math.sin(angle)
            M = np.eye(4, dtype=np.float32)
            if axis==0: M[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
            elif axis==1: M[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
            else: M[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
            return M
        M_rot = make_rot(rx,0) @ make_rot(ry,1) @ make_rot(rz,2)

        return cv_to_gl @ T_base @ T_slide @ M_rot @ S if not is_occluder else cv_to_gl @ T_base @ S

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