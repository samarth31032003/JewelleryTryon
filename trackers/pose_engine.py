# trackers/pose_engine.py
import cv2
import mediapipe as mp

class PoseEngine:
    _instance = None # Singleton pattern to prevent reloading heavy models

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoseEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized: return
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            refine_face_landmarks=True # Enable Iris tracking (good for earrings)
        )
        
        # Drawing utilities (Optional, for debug)
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_results = None
        self.initialized = True

    def process(self, image):
        """
        Input: BGR Image
        Output: MediaPipe Results Object
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.last_results = self.holistic.process(img_rgb)
        return self.last_results