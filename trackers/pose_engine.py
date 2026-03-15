# trackers/pose_engine.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from utils.paths import FACE_LANDMARKER_PATH, POSE_LANDMARKER_PATH, HAND_LANDMARKER_PATH
from utils.logger import logger

log = logger.bind(component="ai")


class LandmarkListAdapter:
    """
    Adapts new Tasks API landmark lists to match the old Holistic API interface.

    Old API:  results.face_landmarks.landmark[i].x
    New API:  face_result.face_landmarks[0][i].x

    This wrapper makes the new format accessible via .landmark[i]
    """
    def __init__(self, landmarks):
        self.landmark = landmarks

    def __bool__(self):
        return bool(self.landmark)


class CombinedResults:
    """
    Wraps results from 3 separate Tasks API landmarkers into a single object
    that matches the old MediaPipe Holistic results interface.

    All existing strategy code works unchanged because it accesses:
      results.face_landmarks.landmark[i]
      results.pose_landmarks.landmark[i]
      results.left_hand_landmarks.landmark[i]
      results.right_hand_landmarks.landmark[i]
    """
    def __init__(self, face_result=None, pose_result=None, hand_result=None):
        # Face landmarks (478 points: 468 face + 10 iris)
        if face_result and face_result.face_landmarks:
            self.face_landmarks = LandmarkListAdapter(face_result.face_landmarks[0])
        else:
            self.face_landmarks = None

        # Pose landmarks (33 points)
        if pose_result and pose_result.pose_landmarks:
            self.pose_landmarks = LandmarkListAdapter(pose_result.pose_landmarks[0])
        else:
            self.pose_landmarks = None

        # Hand landmarks — map handedness labels to left/right attributes
        # For mirrored camera input, "Left" = user's left hand (direct mapping)
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        if hand_result and hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                if i < len(hand_result.handedness):
                    label = hand_result.handedness[i][0].category_name
                    adapter = LandmarkListAdapter(hand_lms)
                    if label == "Left":
                        self.left_hand_landmarks = adapter
                    elif label == "Right":
                        self.right_hand_landmarks = adapter


class PoseEngine:
    _instance = None  # Singleton to prevent reloading heavy models

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoseEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        log.info("[PoseEngine] Initializing MediaPipe Tasks API...")

        # Validate model files exist
        for path, name in [
            (FACE_LANDMARKER_PATH, "Face Landmarker"),
            (POSE_LANDMARKER_PATH, "Pose Landmarker"),
            (HAND_LANDMARKER_PATH, "Hand Landmarker"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"[PoseEngine] {name} model not found at: {path}\n"
                    f"Run 'python utils/download_models.py' to download required models."
                )

        # --- Face Landmarker (478 landmarks) ---
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(FACE_LANDMARKER_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # --- Pose Landmarker (33 landmarks) ---
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(POSE_LANDMARKER_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # --- Hand Landmarker (21 landmarks x 2 hands) ---
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(HAND_LANDMARKER_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        self._frame_count = 0
        self.last_results = None
        self.initialized = True
        log.info("[PoseEngine] All 3 landmarkers initialized successfully.")

    def process(self, image):
        """
        Input: BGR Image
        Output: CombinedResults (compatible with old Holistic API)
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Monotonically increasing timestamp for VIDEO mode
        self._frame_count += 1
        timestamp_ms = self._frame_count * 33  # ~30fps interval

        # Run all three landmarkers
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        self.last_results = CombinedResults(face_result, pose_result, hand_result)
        return self.last_results