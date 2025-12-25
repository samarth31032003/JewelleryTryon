# ui/tryon_view.py
import cv2
import json
import numpy as np
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QSlider, QGroupBox, QGridLayout, 
                             QCheckBox, QTextEdit, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt, QTimer

from graphics.renderer import ARViewerWidget
from trackers.wrist_tracker import WristTracker
from utils.paths import CONFIG_PATH

class TryOnWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR Studio - Modular")
        self.resize(1200, 800)
        
        self.tracker = WristTracker()
        self.use_ai = False
        self.show_landmarks = False 
        self.sliders = {} 
        
        self.setup_ui()
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)
        self.timer.start(30)
        
        self.load_settings()
        
        # Get Native Resolution
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"CAMERA NATIVE RES: {w} x {h} (Ratio: {w/h:.2f})")
        
        # schedule it to run immediately *after* the event loop starts and window is ready.
        QTimer.singleShot(100, self.load_default_occluder)

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left: OpenGL View
        self.viewer = ARViewerWidget()
        layout.addWidget(self.viewer, stretch=3)
        
        # Right: Controls
        panel = QWidget(); p_layout = QVBoxLayout(panel); layout.addWidget(panel, stretch=1)
        tabs = QTabWidget()
        
        # TAB 1: BRACELET
        tab_trans = QWidget(); t_lay = QVBoxLayout(tab_trans)
        btn_load = QPushButton("Load Bracelet"); btn_load.clicked.connect(self.load_model_dialog); t_lay.addWidget(btn_load)
        
        self.btn_ai = QPushButton("Enable AI"); self.btn_ai.setCheckable(True)
        self.btn_ai.setStyleSheet("background-color: #555555; color: white; padding: 10px;")
        self.btn_ai.toggled.connect(self.toggle_ai); t_lay.addWidget(self.btn_ai)
        
        self.chk_track = QCheckBox("Show Tracking Points")
        self.chk_track.toggled.connect(self.toggle_tracking_viz); t_lay.addWidget(self.chk_track)
        
        chk_grid = QCheckBox("Show Grid"); 
        chk_grid.toggled.connect(lambda c: setattr(self.viewer, 'show_grid', c)); t_lay.addWidget(chk_grid)
        
        self.txt = QTextEdit(); self.txt.setMaximumHeight(80); t_lay.addWidget(self.txt)
        
        grid = QGridLayout()
        params = [
            ("Scale", 1, 500, 100, 0.01),      # 1 to 5x scale
            ("Pos X", -500, 500, 0, 0.1),      # Moves -50 to +50 units (Was -5 to +5)
            ("Pos Y", -500, 500, 0, 0.1),      # Moves -50 to +50 units
            ("Pos Z", -500, 500, 0, 0.1),      # Moves -50 to +50 units
            ("Rot X", -180, 180, 0, 1.0),      # Rotation is usually fine with 1.0 step
            ("Rot Y", -180, 180, 0, 1.0),
            ("Rot Z", -180, 180, 0, 1.0)
        ]
        self.add_sliders("B", params, grid) 
        grp = QGroupBox("Bracelet Transform"); grp.setLayout(grid); t_lay.addWidget(grp)
        t_lay.addStretch(); tabs.addTab(tab_trans, "Bracelet")
        
        # TAB 2: OCCLUDER
        tab_occ = QWidget(); o_lay = QVBoxLayout(tab_occ)
        btn_reload = QPushButton("Reload Hand Model"); btn_reload.clicked.connect(self.load_default_occluder); o_lay.addWidget(btn_reload)
        chk_vis = QCheckBox("Debug: Show Hand Mesh (Red)"); chk_vis.setChecked(True)
        chk_vis.toggled.connect(lambda c: setattr(self.viewer, 'debug_occluder', c)); o_lay.addWidget(chk_vis)
        
        occ_grid = QGridLayout()
        # 2. UPDATED OCCLUDER PARAMETERS AS WELL
        params_occ = [
            ("Scale", 1, 1000, 100, 0.01),
            ("Pos X", -500, 500, 0, 0.1),      # Increased Range
            ("Pos Y", -500, 500, 0, 0.1),      # Increased Range
            ("Pos Z", -500, 500, 0, 0.1),      # Increased Range
            ("Rot X", -180, 180, 0, 1.0),
            ("Rot Y", -180, 180, 0, 1.0),
            ("Rot Z", -180, 180, 0, 1.0)
        ]
        self.add_sliders("H", params_occ, occ_grid) 
        grp_occ = QGroupBox("Hand Model Transform"); grp_occ.setLayout(occ_grid); o_lay.addWidget(grp_occ)
        o_lay.addStretch(); tabs.addTab(tab_occ, "Occluder")

        p_layout.addWidget(tabs)
        
        # --- TAB 3: CAMERA / LENS ---
        tab_cam = QWidget(); c_lay = QVBoxLayout(tab_cam)
        
        # FOV Slider (The Focal Length Tuner)
        # Range: 20 to 120 degrees. Default for webcams is usually around 40-60.
        cam_grid = QGridLayout()
        cam_params = [
            ("FOV", 20, 120, 45, 1.0), 
            # Note: 45 is a safe starting guess for most webcams
        ]
        self.add_sliders("Cam", cam_params, cam_grid, self.update_camera_params)
        
        grp_cam = QGroupBox("Lens Tuning"); grp_cam.setLayout(cam_grid)
        c_lay.addWidget(grp_cam)
        c_lay.addStretch()
        tabs.addTab(tab_cam, "Camera")
        
    def update_camera_params(self):
        # Callback when slider moves
        fov_val = self.sliders["Cam_FOV"]['obj'].value()
        self.viewer.fov = fov_val
        self.viewer.update() # Trigger repaint        

    def add_sliders(self, prefix, params, layout, callback=None):
        for i, (n, min_v, max_v, def_v, scale) in enumerate(params):
            key = f"{prefix}_{n}"
            l = QLabel(n); s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v)
            if callback: s.valueChanged.connect(callback)
            self.sliders[key] = {'obj': s, 'scale': scale, 'default': def_v}
            layout.addWidget(l, i, 0); layout.addWidget(s, i, 1)

    def load_model_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open", "", "3D (*.glb *.obj)")
        if f: self.viewer.load_object(f, is_occluder=False)

    def load_default_occluder(self):
        # We need to find the file relative to the project root
        # Assuming we run from main.py in root
        hand_path = os.path.abspath("data/3d_models/3d_hand.obj") 
        # Or wherever you store it
        if os.path.exists(hand_path):
            self.viewer.load_object(hand_path, is_occluder=True)
        else:
            print(f"Warning: Hand model not found at {hand_path}")

    def toggle_ai(self, c):
        self.use_ai = c
        self.btn_ai.setText("AI TRACKING: ON" if c else "Enable AI")
        self.btn_ai.setStyleSheet(f"background-color: {'#00AA00' if c else '#555555'}; color: white; padding: 10px;")

    def toggle_tracking_viz(self, c):
        self.show_landmarks = c

    def loop(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame_disp = cv2.flip(frame, 0) # Flip for OpenGL Texturing
        
        # Tracker Logic
        if self.use_ai:
            rvec, tvec, info = self.tracker.process(frame_disp) # Note: Tracker expects flipped or standard? 
            # Check if tracker expects standard:
            # The original code passed `cv2.flip(frame, 0)` to tracker.
            
            if self.show_landmarks:
                frame_disp = self.tracker.draw_debug(frame_disp)
                
            if info['found']:
                self.viewer.model_bracelet = self.compute_matrix(rvec, tvec, "B", ai=True)
                self.viewer.model_occluder = self.compute_matrix(rvec, tvec, "H", ai=True)
                self.txt.setText(f"Tracking: {info['hand']} Hand\nDepth: {info['z_depth']:.1f}")
            else: self.txt.setText("Searching...")
        else:
            self.viewer.model_bracelet = self.compute_matrix(None, None, "B", ai=False)
            self.viewer.model_occluder = self.compute_matrix(None, None, "H", ai=False)
            self.txt.setText("Manual Mode")
        
        self.viewer.update_bg(frame_disp)
        self.viewer.update()

    def compute_matrix(self, rvec, tvec, prefix, ai=False):
        # Helper to get value from slider * scale
        def val(n): 
            return self.sliders[f"{prefix}_{n}"]['obj'].value() * self.sliders[f"{prefix}_{n}"]['scale']
        
        # 1. Base Transformation (From AI or Identity)
        if ai:
            R, _ = cv2.Rodrigues(rvec)
            T_base = np.eye(4, dtype=np.float32)
            T_base[:3, :3] = R
            T_base[:3, 3] = tvec.flatten()
        else: 
            T_base = np.eye(4, dtype=np.float32)
            
        # 2. Coordinate Conversion (OpenCV -> OpenGL)
        # OpenCV looks down +Z, OpenGL looks down -Z. We flip Y and Z.
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        # 3. Manual Scaling
        scale_val = val("Scale")
        S = np.diag([scale_val, scale_val, scale_val, 1.0]).astype(np.float32)
        
        # 4. Manual Rotation
        rx, ry, rz = np.radians(val("Rot X")), np.radians(val("Rot Y")), np.radians(val("Rot Z"))
        
        # Rotation X
        c, s = np.cos(rx), np.sin(rx)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        
        # Rotation Y
        c, s = np.cos(ry), np.sin(ry)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        
        # Rotation Z
        c, s = np.cos(rz), np.sin(rz)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        
        # --- THE FIX: TRY SWAPPING THIS ORDER ---
        # Old: M_rot_3x3 = Rz @ Ry @ Rx
        # New: Rx @ Ry @ Rz 
        M_rot_3x3 = Rx @ Ry @ Rz 
        
        M_rot = np.eye(4, dtype=np.float32)
        M_rot[:3, :3] = M_rot_3x3
        
        # 5. Manual Translation (Position)
        tx, ty, tz = val("Pos X"), val("Pos Y"), val("Pos Z")
        if not ai: 
            tz -= 30.0 # Default push-back for manual mode so it's visible
        
        T_off = np.eye(4, dtype=np.float32)
        T_off[:3, 3] = [tx, ty, tz]
        
        # 6. Final Multiplication Order
        # Scale -> Rotate -> Translate -> Base(AI) -> FixCoords
        return cv_to_gl @ T_base @ T_off @ M_rot @ S
    
    def save_settings(self):
        """Iterates through all sliders and saves their raw integer values to JSON."""
        settings_data = {}
        
        # self.sliders is a dict like: {'B_Scale': {'obj': QSlider, ...}, ...}
        for key, info in self.sliders.items():
            # We save the raw integer value from the slider
            settings_data[key] = info['obj'].value()
            
        try:
            # Save to 'config.json' in the same directory as main.py
            with open(CONFIG_PATH, "w") as f:
                json.dump(settings_data, f, indent=4)
            print("Settings saved successfully.")
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def load_settings(self):
        """Reads JSON and updates slider positions."""
        if not os.path.exists(CONFIG_PATH):
            print("No config file found, using defaults.")
            return

        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                
            # Update UI from data
            for key, value in data.items():
                if key in self.sliders:
                    # Block signals briefly so we don't trigger 
                    # a render update for every single slider change
                    self.sliders[key]['obj'].blockSignals(True)
                    self.sliders[key]['obj'].setValue(value)
                    self.sliders[key]['obj'].blockSignals(False)
            
            # Trigger one final update to apply all changes to the 3D view
            self.viewer.update()
            print("Settings loaded.")
            
        except Exception as e:
            print(f"Error loading settings: {e}")

    def closeEvent(self, event):
        """Called when the window is being closed."""
        # 1. Save the state
        self.save_settings()
        
        # 2. Release the camera resource
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        # 3. Stop the timer to prevent background processing
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        event.accept()