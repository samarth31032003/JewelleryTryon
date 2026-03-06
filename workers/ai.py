# workers/ai.py
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QObject
from trackers.pose_engine import PoseEngine
from utils.logger import logger

log = logger.bind(component="ai")

class AIWorker(QThread):
    """
    Runs MediaPipe and Tracking Strategies in a background thread.
    Receives frames -> Calculates Matrices -> Emits Render Commands.
    """
    # Signal: Emits (list_of_commands, is_face_detected)
    results_ready = pyqtSignal(list, bool)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.latest_frame = None
        self.strategy = None
        
        # Mutex to prevent reading/writing frame at same time
        self.mutex = QMutex()
        self.settings_mutex = QMutex()
        
        # Initialize Engine (created here to stay on this thread)
        self.ai_engine = None 

    def run(self):
        """Main Loop: Waits for a frame, processes it, sleeps."""
        self.is_running = True
        log.info("[AIWorker] Starting AI Engine...")
        if self.ai_engine is None:
            self.ai_engine = PoseEngine()

        while self.is_running:
            # 1. Get the latest frame safely
            frame = None
            self.mutex.lock()
            if self.latest_frame is not None:
                frame = self.latest_frame
                self.latest_frame = None # Consume it
            self.mutex.unlock()

            # 2. Process if we have a frame AND a strategy
            if frame is not None and self.strategy:
                # Run MediaPipe
                results = self.ai_engine.process(frame)
                h, w, _ = frame.shape
                
                # Run Strategy Logic (Math)
                self.settings_mutex.lock()
                try:
                    commands = self.strategy.process_frame(results, w, h)
                except Exception as e:
                    log.error(f"[AIWorker] Strategy Error: {e}")
                    commands = []
                self.settings_mutex.unlock()

                # Check if face was found
                face_found = bool(results.face_landmarks)
                
                # 3. Send results back to UI
                self.results_ready.emit(commands, face_found)
            
            # Sleep to prevent CPU hogging (1ms)
            self.msleep(1)

        log.info("[AIWorker] Stopped.")

    def process_frame(self, frame):
        """Called by CameraWorker (via Main) to update the input."""
        self.mutex.lock()
        self.latest_frame = frame
        self.mutex.unlock()

    def set_strategy(self, strategy):
        """Updates the active tracking strategy safely."""
        self.settings_mutex.lock()
        self.strategy = strategy
        self.settings_mutex.unlock()

    def update_settings(self, new_settings):
        """Updates slider values safely."""
        self.settings_mutex.lock()
        if self.strategy:
            # Handle CollectionStrategy vs Standard Strategy
            if hasattr(self.strategy, 'update_settings'):
                self.strategy.update_settings(new_settings)
            else:
                self.strategy.settings.update(new_settings)
        self.settings_mutex.unlock()

    def stop(self):
        self.is_running = False
        self.wait()