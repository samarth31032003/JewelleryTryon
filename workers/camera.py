# workers/camera.py
import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class CameraWorker(QThread):
    """
    Runs in a background thread. Captures frames from the webcam.
    Emits 'frame_captured' signal with the new image.
    """
    frame_captured = pyqtSignal(np.ndarray) # Signal to send data to UI
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.is_running = True
        self.cap = None

    def run(self):
        """The main loop of the thread."""
        print(f"[CameraWorker] Starting Camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Optimize Camera (Optional, depends on hardware)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("[CameraWorker] Error: Could not open camera.")
            return

        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Flip mirrored if needed (selfie mode)
                frame = cv2.flip(frame, 1)
                
                # Emit the frame to the Main Thread
                # (This copies the data safely across threads)
                self.frame_captured.emit(frame)
            else:
                print("[CameraWorker] Warning: Empty frame.")
                time.sleep(0.1) # Prevent busy loop on failure
            
            # Tiny sleep to yield execution (prevents 100% CPU usage)
            # 1ms is enough for 60+ FPS
            self.msleep(1)

        # Cleanup when loop breaks
        self.cap.release()
        print("[CameraWorker] Camera released.")

    def stop(self):
        """Safely stops the thread loop."""
        self.is_running = False
        self.wait() # Block until run() finishes