# workers/camera.py
import cv2
import time
import os
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class CameraWorker(QThread):
    """
    Smart Camera Worker.
    - Runs in background thread.
    - Attempts HD (1280x720) by default but adapts to hardware limits.
    - Emits 'frame_captured' with whatever aspect ratio the camera provides.
    """
    frame_captured = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0, width=1280, height=720):
        super().__init__()
        self.camera_index = camera_index
        # We request HD, but will accept whatever the camera gives
        self.req_w = width
        self.req_h = height
        self.is_running = True
        self.cap = None

    def run(self):
        """The main loop of the thread."""
        print(f"[CameraWorker] Opening Camera {self.camera_index}...")
        
        # Optimization: Use DirectShow on Windows for faster startup/switching
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        
        # 1. Request High Resolution (e.g., 16:9 HD)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("[CameraWorker] CRITICAL: Could not open camera.")
            return

        # 2. Check what resolution we ACTUALLY got (Adaptive Logic)
        real_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[CameraWorker] Camera Active @ {int(real_w)}x{int(real_h)}")
        
        # (This resolution is implicitly passed to the Viewer via the frame itself,
        # so the Viewer will automatically adjust its aspect ratio bars).

        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame (Standard Mirror behavior)
                frame = cv2.flip(frame, 1)
                self.frame_captured.emit(frame)
            else:
                # Camera busy or disconnected
                time.sleep(0.1)
            
            # Tiny sleep to share CPU
            self.msleep(1)

        self.cap.release()
        print("[CameraWorker] Camera released.")

    def stop(self):
        """Safely stops the thread loop."""
        self.is_running = False
        self.wait()