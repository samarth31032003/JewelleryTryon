import cv2
import mediapipe as mp
import numpy as np

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

        # 2. Define the "Extended Hand Model" (Anchor)
        self.wrist_3d_model = np.array([
            # (0.0, 0.0, 0.0),       # 0: WRIST
            [0, 0.025, 0],
            (-2.5, 6.0, 0.0),      # 1: INDEX_MCP
            (2.5, 5.5, 0.0),       # 2: PINKY_MCP
            (0.0, 6.0, 0.0),       # 3: MIDDLE_MCP
            (0.0, -25.0, 0.0),     # 4: ELBOW (The Stabilizer)
        ], dtype=np.float64)

        # Camera Matrix Cache
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Store last results for drawing
        self.last_results = None

    def process(self, image):
        """
        Input: BGR Image from OpenCV
        Output: rvec, tvec, debug_data (dict)
        """
        h, w, c = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run Holistic
        results = self.holistic.process(img_rgb)
        self.last_results = results # Save for drawing later
        
        if self.camera_matrix is None or self.camera_matrix[0,2] != w/2:
            focal_length = w 
            center = (w / 2, h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )

        debug_info = {"found": False, "error": 0.0, "z_depth": 0.0}

        if not results.pose_landmarks:
            return None, None, debug_info

        candidates = []
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14))

        for hand_landmarks, elbow_idx in candidates:
            # --- STEP A: Get 2D Pixels for Hand (4 Points) ---
            key_indices = [0, 5, 17, 9] 
            image_points = []
            
            for k in key_indices:
                lm = hand_landmarks.landmark[k]
                px, py = int(lm.x * w), int(lm.y * h)
                image_points.append([px, py])

            # --- STEP B: Get 2D Pixel for Elbow (1 Point) ---
            elbow_lm = results.pose_landmarks.landmark[elbow_idx]
            if elbow_lm.visibility < 0.5: continue 

            e_px, e_py = int(elbow_lm.x * w), int(elbow_lm.y * h)
            image_points.append([e_px, e_py])

            image_points = np.array(image_points, dtype=np.float64)

            # --- STEP C: Solve PnP (5 Points) ---
            success_pnp, rvec, tvec = cv2.solvePnP(
                self.wrist_3d_model, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                projected, _ = cv2.projectPoints(
                    self.wrist_3d_model, rvec, tvec, self.camera_matrix, self.dist_coeffs
                )
                projected = projected.reshape(-1, 2)
                error = np.linalg.norm(image_points - projected, axis=1).mean()

                debug_info = {
                    "found": True,
                    "error": error,
                    "z_depth": tvec[2][0],
                    "rvec": rvec,
                    "tvec": tvec
                }
                return rvec, tvec, debug_info

        return None, None, debug_info

    def draw_debug(self, image):
        """ Draws MediaPipe landmarks on the image """
        if self.last_results:
            # Draw Hands
            if self.last_results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    self.last_results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            
            if self.last_results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    self.last_results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            
            # Draw Pose (Arm only ideally, but full pose for now to see elbows)
            if self.last_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    self.last_results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image