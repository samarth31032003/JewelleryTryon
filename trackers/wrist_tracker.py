# trackers/wrist_tracker.py
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

        # --- THE FIX: Check both hands with labels ---
        candidates = []
        
        # Tuple format: (Landmarks, Elbow_Index, 3D_Model, Label)
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, 13, self.model_right, "Right"))
            
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, 14, self.model_left, "Left"))

        for hand_landmarks, elbow_idx, model_3d, label in candidates:
            # 1. Get 2D Pixels (Hand)
            key_indices = [0, 5, 17, 9] 
            image_points = []
            
            for k in key_indices:
                lm = hand_landmarks.landmark[k]
                px, py = int(lm.x * w), int(lm.y * h)
                image_points.append([px, py])

            # 2. Get 2D Pixel (Elbow)
            elbow_lm = results.pose_landmarks.landmark[elbow_idx]
            if elbow_lm.visibility < 0.5: continue 

            e_px, e_py = int(elbow_lm.x * w), int(elbow_lm.y * h)
            image_points.append([e_px, e_py])

            image_points = np.array(image_points, dtype=np.float64)

            # 3. Solve PnP using the CORRECT Model (Left vs Right)
            success_pnp, rvec, tvec = cv2.solvePnP(
                model_3d,   # <--- Uses self.model_right or self.model_left
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                # Calculate Error
                projected, _ = cv2.projectPoints(model_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                projected = projected.reshape(-1, 2)
                error = np.linalg.norm(image_points - projected, axis=1).mean()

                debug_info = {
                    "found": True,
                    "error": error,
                    "z_depth": tvec[2][0],
                    "rvec": rvec,
                    "tvec": tvec,
                    "hand": label # Tell UI which hand it is
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