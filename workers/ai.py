# workers/ai.py
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QObject
from trackers.pose_engine import PoseEngine
from utils.logger import logger
import traceback

log = logger.bind(component="ai")

class AIWorker(QThread):
    """
    Runs MediaPipe and Tracking Strategies in a background thread.
    Receives frames -> Calculates Matrices -> Emits Render Commands.
    """
    results_ready = pyqtSignal(list, bool)
    frame_processed = pyqtSignal(np.ndarray) # Emits the composited frame

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.latest_frame = None
        self.strategy = None
        self.active_mode = "3d" # Default routing mode
        
        # Instantiate the 2D pipeline manager and structural strategy
        from graphics.overlay_2d import CollectionManager2D
        from trackers.strategies.strategy_2d import Strategy2D
        self.manager_2d = CollectionManager2D()
        self.strategy_2d = Strategy2D(self.manager_2d)
        
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
            try:
                # 1. Get the latest frame safely
                frame = None
                self.mutex.lock()
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None # Consume it
                self.mutex.unlock()

                # 2. Process if we have a frame AND (a strategy OR we are in 2D mode)
                if frame is not None and (self.strategy or self.active_mode == '2d'):
                    # Run MediaPipe
                    results = self.ai_engine.process(frame)
                    h, w, _ = frame.shape
                    
                    # Run Strategy Logic (Math) - Only if it exists (3D mode)
                    commands = []
                    if self.strategy:
                        self.settings_mutex.lock()
                        try:
                            commands = self.strategy.process_frame(results, w, h)
                        except Exception as e:
                            log.error(f"[AIWorker] Strategy Error: {e}")
                        self.settings_mutex.unlock()

                    face_found = bool(results.face_landmarks)
                    
                    # Routing Logic
                    if self.active_mode == '2d':
                        try:
                            # Flat 2D Mode: Compose the layout via OpenCV overlay
                            composited = self.manager_2d.process_frame(frame, results, w, h)
                            self.frame_processed.emit(composited)
                            self.results_ready.emit([], face_found) # Empty 3D draw cmds
                        except Exception as e:
                            log.error(f"FATAL 2D ERROR: {e}")
                            log.error(traceback.format_exc())
                    else:
                        try:
                            # True 3D Mode: Emit original frame to background, compute 3D matrices
                            self.frame_processed.emit(frame)
                            self.results_ready.emit(commands, face_found)
                        except Exception as e:
                            log.error(f"FATAL 3D ERROR: {e}")
                            log.error(traceback.format_exc())         
                
                # Sleep to prevent CPU hogging (1ms)
                self.msleep(1)
            
            except Exception as e:
                # This will catch absolutely any error in the entire thread
                log.error(f"🚨 FATAL AI THREAD CRASH: {e}")
                log.error(traceback.format_exc())
                
                # Sleep for 1 second so it doesn't spam your logs 10,000 times a second
                import time
                time.sleep(1)           

        log.info("[AIWorker] Stopped.")

    def process_frame(self, frame):
        """Called by CameraWorker (via Main) to update the input."""
        self.mutex.lock()
        self.latest_frame = frame
        self.mutex.unlock()

    def set_active_collection(self, items: list, mode: str):
        """Prepares the AIWorker with new items depending on the mode."""
        self.settings_mutex.lock()
        self.active_mode = mode
        
        if mode == '2d':
            self.manager_2d.clear()
            key_real = "necklace" # Default fallback
            for item in items:
                if item.image_2d_path:
                    key_hint = item.name.lower() + " " + item.image_2d_path.lower()
                    if "ear" in key_hint: key_real = "ear"
                    elif "neck" in key_hint: key_real = "necklace"
                    elif "nose" in key_hint: key_real = "nosepin"
                    else: key_real = "forehead"  
                    
                    self.manager_2d.load_item(key_real, item.image_2d_path)
            
            # Create the strategy and explicitly tell it what type of sliders to draw!
            from trackers.strategies.strategy_2d import Strategy2D
            self.strategy = Strategy2D(self.manager_2d, active_type=key_real)
            # Load saved slider values from the Database
            for item in items:
                if hasattr(item, 'settings') and isinstance(item.settings, dict):
                    self.strategy.update_settings(item.settings)

        self.settings_mutex.unlock()

    def set_strategy(self, strategy):
        """Updates the active tracking strategy safely."""
        self.settings_mutex.lock()
        self.strategy = strategy
        self.settings_mutex.unlock()

    def update_settings(self, new_settings):
        """Updates slider values safely."""
        log.warning(f"🚨 [TRIPWIRE 1] AIWorker received: {new_settings}")
        
        self.settings_mutex.lock()
        if self.strategy:
            if hasattr(self.strategy, 'update_settings'):
                self.strategy.update_settings(new_settings)
            else:
                self.strategy.settings.update(new_settings)
        self.settings_mutex.unlock()

    def stop(self):
        self.is_running = False
        self.wait()