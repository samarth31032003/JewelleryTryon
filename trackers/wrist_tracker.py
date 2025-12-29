# trackers/wrist_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import time # Needed for smoothing timestamps
from utils.smoothing import OneEuroFilter, RotationFilter

class WristTracker:
    def __init__(self):
        # 1. Setup MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            refine_face_landmarks=False
        )
        
        # Drawing Utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # --- THE FIX: Define Separate Models for Left and Right ---
        # RIGHT HAND: Index (-2.5) is to the "Left" of wrist, Pinky (2.5) to "Right"
        # left right swapped because of inverse selfie camera
        self.model_left = np.array([
            [0, 0.025, 0],         # WRIST
            [-2.5, 6.0, 0.0],      # INDEX_MCP
            [2.5, 5.5, 0.0],       # PINKY_MCP
            [0.0, 6.0, 0.0],       # MIDDLE_MCP
            [0.0, -25.0, 0.0],     # ELBOW
        ], dtype=np.float64)

        # LEFT HAND: Mirror the X values (Multiply X by -1)
        self.model_right = self.model_left.copy()
        self.model_right[:, 0] *= -1 

        # Camera Matrix Cache
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.last_results = None

        # --- FILTERS ---
        self._init_filters()
        self.MAX_JUMP_THRESHOLD = 50.0  # Increased slightly
        self.SIDE_VIEW_THRESHOLD = 25.0 
        # Safety: Store last valid Z to prevent crashing into screen
        self.last_valid_z = 50.0 
        # SAFETY MEMORY
        self.last_valid_tvec = None
        self.last_valid_rvec = None
        self.last_hand_type = None  # <--- NEW: Remembers "Left" or "Right"

    def _init_filters(self):
        """Helper to reset filters from scratch"""
        # --- SMOOTHING FILTERS ---
        # min_cutoff: 0.01 (Very smooth when still)
        # beta: 10.0 (Very responsive when moving)
        # Jittery when holding hand still? Decrease min_cutoff (e.g., to 0.001).
        # Laggy when waving hand fast? Increase beta (e.g., to 20.0 or 50.0).
        self.filter_rvec = RotationFilter(min_cutoff=0.5, beta=0.5)
        self.filter_tvec = OneEuroFilter(min_cutoff=0.1, beta=0.5)

    def process(self, image):
        """
        Input: BGR Image from OpenCV
        Output: rvec, tvec, debug_data (dict)
        """
        h, w, c = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.holistic.process(img_rgb)
        self.last_results = results 
        
        # Init Camera Matrix if needed
        if self.camera_matrix is None or self.camera_matrix[0,2] != w/2:
            focal_length = w 
            center = (w / 2, h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )

        debug_info = {"found": False, "error": 0.0, "z_depth": 0.0, "hand": "None"}

        if not results.pose_landmarks:
            return None, None, debug_info

        # --- HAND SELECTION ---
        # Note: We rely on the "Mirror" logic. 
        # If user sees "Left Hand" on screen (Mirrored Right), we process it as Left.
        candidates = []
        # Tuple format: (Landmarks, Elbow_Index, 3D_Model, Label)
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13, self.model_right, "Right"))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14, self.model_left, "Left"))

        for hand_landmarks, elbow_idx, model_3d, label in candidates:
            # --- THE FIX: HAND SWAP RESET ---
            # If we switched from Right to Left (or vice versa), RESET EVERYTHING.
            if label != self.last_hand_type:
                # print(f"Hand Changed: {self.last_hand_type} -> {label}. Resetting State.")
                self.last_valid_tvec = None
                self.last_valid_rvec = None
                self._init_filters() # Wipe the smoothing memory
                self.last_hand_type = label
            # 1. Get Raw 2D Points
            # Indices: Wrist(0), Index(5), Pinky(17), Middle(9)
            wrist = np.array(self._get_px(hand_landmarks.landmark[0], w, h))
            index = np.array(self._get_px(hand_landmarks.landmark[5], w, h))
            pinky = np.array(self._get_px(hand_landmarks.landmark[17], w, h))
            middle = np.array(self._get_px(hand_landmarks.landmark[9], w, h))
            
            # Elbow from Pose
            elbow_lm = results.pose_landmarks.landmark[elbow_idx]
            if elbow_lm.visibility < 0.5: continue
            elbow = np.array(self._get_px(elbow_lm, w, h))

            # --- CHECK 1: SIDE VIEW DETECTION ---
            # Calculate visible width (Index <-> Pinky distance in pixels)
            hand_width_px = np.linalg.norm(index - pinky)
            is_side_view = hand_width_px < self.SIDE_VIEW_THRESHOLD
            # --- THE FIX: Straighten the Hand ---
            # We virtually move the Index, Pinky, and Middle points to align 
            # with the Wrist-Elbow axis, ignoring wrist flexion.
            
            # Ratio based on your 3D Model:
            # Elbow->Wrist is 25.0 units.
            # Wrist->Knuckles is approx 6.0 units.
            # So Knuckles should be (6/25) = 0.24 units further along the arm line.
            arm_ratio = 6.0 / 25.0 
            
            fixed_index = self._project_point(elbow, wrist, index, arm_ratio)
            fixed_pinky = self._project_point(elbow, wrist, pinky, arm_ratio)
            fixed_middle = self._project_point(elbow, wrist, middle, arm_ratio)
            
            # Pack points for PnP
            # Order must match self.wrist_3d_model: [WRIST, INDEX, PINKY, MIDDLE, ELBOW]
            image_points = np.array([
                wrist, 
                fixed_index, 
                fixed_pinky, 
                fixed_middle, 
                elbow
            ], dtype=np.float64)

            # 2. Solve PnP (Now solving for a "Straight Forearm")
            success_pnp, rvec, tvec = cv2.solvePnP(
                model_3d, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=(self.last_valid_rvec is not None),
                rvec=self.last_valid_rvec if self.last_valid_rvec is not None else None,
                tvec=self.last_valid_tvec if self.last_valid_tvec is not None else None
            )

            if success_pnp:
                # --- APPLY SMOOTHING ---
                current_time = time.time()

                # --- CHECK 2: TELEPORT GUARD ---
                if self.last_valid_tvec is not None:
                    # How far did it move?
                    dist = np.linalg.norm(tvec - self.last_valid_tvec)
                    if dist > self.MAX_JUMP_THRESHOLD:
                        # IMPOSSIBLE JUMP -> Ignore this frame completely
                        # Just return the last known good smoothed data
                        # print("JUMP DETECTED - REJECTING")
                        return self.filter_rvec.x_prev.reshape(3,1), self.filter_tvec.x_prev.reshape(3,1), debug_info

                # --- CHECK 3: LOCK Z-DEPTH IN SIDE VIEW ---
                if is_side_view and self.last_valid_tvec is not None:
                    # Trust X and Y, but Force Z to be the same as before
                    # This prevents the "Zoom to Ear" bug
                    tvec[2] = self.last_valid_tvec[2]
                
                # We flatten rvec/tvec to 1D arrays for the filter, then reshape back
                rvec_smooth = self.filter_rvec.update(rvec.flatten(), current_time)
                tvec_smooth = self.filter_tvec.update(tvec.flatten(), current_time).reshape(3, 1)

                # --- SAFETY GUARD: Z-Depth ---
                # If Z is too small (e.g. < 5cm), it means PnP failed/collapsed.
                # We ignore this bad frame and keep the old Z depth.
                z_raw = tvec_smooth[2][0]
                if z_raw < 5.0: 
                    # Use last known good Z, but keep X/Y updates
                    tvec_smooth[2][0] = self.last_valid_z
                else:
                    self.last_valid_z = z_raw
                
                # Save valid raw state for next frame guards
                self.last_valid_rvec = rvec
                self.last_valid_tvec = tvec
                # Calculate Error (Optional: Use smoothed or raw for debug?)
                # Usually calculate error on RAW data to see tracking quality
                projected, _ = cv2.projectPoints(model_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                projected = projected.reshape(-1, 2)
                error = np.linalg.norm(image_points - projected, axis=1).mean()

                debug_info = {
                    "found": True,
                    "error": error,
                    "z_depth": tvec_smooth[2][0], # Return smoothed Z
                    "rvec": rvec_smooth,          # Return smoothed R
                    "tvec": tvec_smooth,          # Return smoothed T
                    "hand": label
                }
                return rvec_smooth, tvec_smooth, debug_info

        return None, None, debug_info

    def _get_px(self, lm, w, h):
        return [int(lm.x * w), int(lm.y * h)]

    def _project_point(self, elbow, wrist, knuckle, ratio):
        """
        Projects the 'knuckle' onto the local coordinate system of the arm 
        to ignore wrist bending (pitch), while preserving width (roll/yaw).
        """
        e = np.array(elbow)
        w = np.array(wrist)
        k = np.array(knuckle)
        
        # Vector along the arm (Elbow -> Wrist)
        arm_vec = w - e
        arm_len = np.linalg.norm(arm_vec)
        if arm_len == 0: return k # Safety
        
        arm_dir = arm_vec / arm_len
        
        # 1. Calculate ideal "height" along the arm
        # We enforce the knuckle to be at a fixed distance from the wrist
        # extending OUTWARDS from the elbow.
        ideal_pos_on_line = w + (arm_dir * (arm_len * ratio))
        
        # 2. Calculate "width" (deviation from the center line)
        # We project the REAL knuckle onto the arm line to find its current 'height'
        # vector_to_k = k - e
        # projection_len = np.dot(vector_to_k, arm_dir)
        # projected_point = e + (arm_dir * projection_len)
        
        # The deviation is the perpendicular vector from the line to the real knuckle
        # deviation_vec = k - projected_point
        
        # -- SIMPLIFIED MATH --
        # Actually, simpler: 
        # We want the knuckle to be at 'ideal_pos_on_line' + 'deviation_vec'.
        # But wait, if we wave "Goodbye" (Flexion), the 'deviation_vec' changes?
        # No, flexion moves the knuckle parallel to the arm line (mostly).
        # Deviation (Yaw) moves it perpendicular.
        # We want to keep Perpendicular movement, kill Parallel movement.
        
        # Re-calc with cleaner math:
        # A. Find perpendicular distance of K from Line(E, W)
        vec_wk = k - w
        # Component parallel to arm
        parallel_component = np.dot(vec_wk, arm_dir) * arm_dir
        # Component perpendicular to arm (The Width/Twist)
        perp_component = vec_wk - parallel_component
        
        # B. Construct new point
        # Start at Wrist -> Add Fixed Length along arm -> Add Original Width
        # Note: 'ratio' is relative to Arm Length. 
        # Wrist->Knuckle length = arm_len * ratio
        new_point = w + (arm_dir * (arm_len * ratio)) + perp_component
        
        return new_point
    
    def draw_debug(self, image):
        """ Draws MediaPipe landmarks on the image """
        if self.last_results:
            # Draw Hands
            if self.last_results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    self.last_results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS)
            if self.last_results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    self.last_results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS)
            if self.last_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    self.last_results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS)
        return image