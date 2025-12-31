# ui/tryon_view.py
import cv2
import json
import numpy as np
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QGroupBox, QGridLayout, 
                             QCheckBox, QTextEdit, QTabWidget)
from PyQt5.QtCore import Qt, QTimer

from graphics.renderer import ARViewerWidget
from trackers.wrist_tracker import WristTracker

class TryOnWindow(QWidget): # Inherit from QWidget for QStackedWidget compatibility
    def __init__(self, db):
        super().__init__()
        self.db = db  # Store DB reference
        self.tracker = WristTracker()
        self.current_item = None # Track the active jewelry item
        
        self.use_ai = False
        self.show_landmarks = False 
        self.sliders = {} 
        
        # --- NEW: State Management for Loading ---
        self.occluder_loaded = False
        self.pending_item = None # Changed from path to Item Object
        
        self.setup_ui()
        
        self.cap = cv2.VideoCapture() 
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)
        

    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: OpenGL View
        self.viewer = ARViewerWidget()
        layout.addWidget(self.viewer, stretch=3)
        
        # Right: Controls
        panel = QWidget(); p_layout = QVBoxLayout(panel); layout.addWidget(panel, stretch=1)
        tabs = QTabWidget()
        
        # --- TAB 1: JEWELRY TRANSFORM ---
        tab_trans = QWidget(); t_lay = QVBoxLayout(tab_trans)
        # Removed "Load Bracelet" button since Catalogue handles this now
        
        self.btn_ai = QPushButton("Enable AI"); self.btn_ai.setCheckable(True)
        self.btn_ai.setStyleSheet("background-color: #555555; color: white; padding: 10px;")
        self.btn_ai.toggled.connect(self.toggle_ai); t_lay.addWidget(self.btn_ai)
        
        self.chk_track = QCheckBox("Show Tracking Points")
        self.chk_track.toggled.connect(self.toggle_tracking_viz); t_lay.addWidget(self.chk_track)

        # SAVE BUTTON (Manual Trigger)
        btn_save = QPushButton("Save Settings for this Item")
        btn_save.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_current_settings_to_db)
        t_lay.addWidget(btn_save)

        chk_grid = QCheckBox("Show Grid"); 
        chk_grid.toggled.connect(lambda c: setattr(self.viewer, 'show_grid', c)); t_lay.addWidget(chk_grid)
        
        self.txt = QTextEdit(); self.txt.setMaximumHeight(80); t_lay.addWidget(self.txt)
        
        grid = QGridLayout()
        # Generic Jewelry Params (Applies to Necklace, Bracelet, Ring, etc.)
        # We use prefix "J_" (Jewelry) instead of "B_" to be generic
        # [Name, Min, Max, Default, Scale]
        params_jewelry = [
            ("Scale", 1, 500, 100, 0.01),
            ("Slide", -500, 500, 0, 0.5),
            ("Rot X", -180, 180, 0, 1.0),
            ("Rot Y", -180, 180, 0, 1.0),
            ("Rot Z", -180, 180, 0, 1.0)
        ]
        
        btn_reset = QPushButton("Reset Transform")
        btn_reset.setStyleSheet("background-color: #AA4444; color: white; font-weight: bold;")
        btn_reset.clicked.connect(lambda: self.reset_transform("J"))
        t_lay.addWidget(btn_reset)
        
        self.add_sliders("J", params_jewelry, grid) 
        grp = QGroupBox("Jewelry Adjustment"); grp.setLayout(grid); t_lay.addWidget(grp)
        t_lay.addStretch(); tabs.addTab(tab_trans, "Jewelry")
        
        # --- TAB 2: OCCLUDER ---
        tab_occ = QWidget(); o_lay = QVBoxLayout(tab_occ)
        # Removed Reload button (handled automatically)
        
        chk_vis = QCheckBox("Debug: Show Hand Mesh (Red)"); chk_vis.setChecked(False) #false default
        chk_vis.toggled.connect(lambda c: setattr(self.viewer, 'debug_occluder', c)); o_lay.addWidget(chk_vis)
        
        occ_grid = QGridLayout()
        params_occ = [
            ("Scale", 1, 1000, 100, 0.01),
            ("Slide", -500, 500, 0, 0.5),
            ("Rot X", -180, 180, 0, 1.0),
            ("Rot Y", -180, 180, 0, 1.0),
            ("Rot Z", -180, 180, 0, 1.0)
        ]

        # FIX: Using the exact same 'params_occ' structure as Bracelet
        # This replaces the old Pos X/Y/Z with Slide/Rot logic
        btn_reset_occ = QPushButton("Reset Hand Transform")
        btn_reset_occ.setStyleSheet("background-color: #AA4444; color: white; font-weight: bold;")
        btn_reset_occ.clicked.connect(lambda: self.reset_transform("H"))
        o_lay.addWidget(btn_reset_occ)
        
        self.add_sliders("H", params_occ, occ_grid) 
        grp_occ = QGroupBox("Occluder Adjustment"); grp_occ.setLayout(occ_grid); o_lay.addWidget(grp_occ)
        o_lay.addStretch(); tabs.addTab(tab_occ, "Occluder")

        p_layout.addWidget(tabs)
        
        # --- TAB 3: CAMERA ---
        tab_cam = QWidget(); c_lay = QVBoxLayout(tab_cam)
        cam_grid = QGridLayout()
        cam_params = [("FOV", 20, 120, 45, 1.0)]
        self.add_sliders("Cam", cam_params, cam_grid, self.update_camera_params)
        grp_cam = QGroupBox("Lens"); grp_cam.setLayout(cam_grid)
        c_lay.addWidget(grp_cam)
        c_lay.addStretch()
        tabs.addTab(tab_cam, "Camera")

    # --- CRITICAL FIX: The Safe Loading Logic ---
    def set_active_item(self, item):
        """
        1. Saves settings for the OLD item.
        2. Loads settings for the NEW item.
        3. Queues the 3D model for loading.
        """
        # A. Save previous item state
        if self.current_item is not None:
            self.save_current_settings_to_db()
        
        # B. Switch to new Item
        self.current_item = item
        self.pending_item = item
        
        # C. Load Settings from DB (or use defaults)
        if item.settings:
            print(f"Restoring settings for {item.name}")
            self.apply_slider_values(item.settings)
        else:
            print(f"No saved settings for {item.name}, using defaults.")
            self.reset_transform("J") # Reset Jewelry sliders
        
        # D. Trigger Load if visible
        if self.isVisible():
            self._process_pending_loads()

    def save_current_settings_to_db(self):
        """Gathers all slider values and updates the DB."""
        if not self.current_item: return
        
        settings = {}
        for key, info in self.sliders.items():
            # Save raw integer value from slider
            settings[key] = info['obj'].value()
            
        self.db.update_item_settings(self.current_item.id, settings)
        print(f"Saved configuration for {self.current_item.name}")

    def apply_slider_values(self, settings):
        """Restores slider positions from a dictionary."""
        self.viewer.blockSignals(True) # Prevent lag while setting multiple
        for key, value in settings.items():
            if key in self.sliders:
                self.sliders[key]['obj'].setValue(value)
        self.viewer.blockSignals(False)
        self.viewer.update()

    # NEW: Only load OpenGL content when the window is actually shown!
    def showEvent(self, event):
        """Called automatically when the widget becomes visible."""
        super().showEvent(event)
        # Now that we are visible, the OpenGL Context is real. Load everything.
        QTimer.singleShot(50, self._process_pending_loads)

    def _process_pending_loads(self):
        # 1. Load the Jewelry Item
        if self.pending_item:
            print(f"Loading 3D Model: {self.pending_item.model_path}")
            self.viewer.load_object(self.pending_item.model_path, is_occluder=False)
            self.pending_item = None # Clear the queue

        # 2. Load the Hand Occluder (Once)
        if not self.occluder_loaded:
            hand_path = os.path.abspath("data/3d_models/3d_hand.obj") 
            if os.path.exists(hand_path):
                print("Loading Occluder...")
                self.viewer.load_object(hand_path, is_occluder=True)
                self.occluder_loaded = True

    def add_sliders(self, prefix, params, layout, callback=None):
        for i, (n, min_v, max_v, def_v, scale) in enumerate(params):
            key = f"{prefix}_{n}"
            l = QLabel(n); s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v)
            if callback: s.valueChanged.connect(callback)
            self.sliders[key] = {'obj': s, 'scale': scale, 'default': def_v}
            layout.addWidget(l, i, 0); layout.addWidget(s, i, 1)

    def update_camera_params(self):
        # Callback when slider moves
        fov_val = self.sliders["Cam_FOV"]['obj'].value()
        self.viewer.fov = fov_val
        self.viewer.update() # Trigger repaint 


    def toggle_ai(self, c):
        self.use_ai = c
        self.btn_ai.setText("AI TRACKING: ON" if c else "Enable AI")
        self.btn_ai.setStyleSheet(f"background-color: {'#00AA00' if c else '#555555'}; color: white; padding: 10px;")

    def toggle_tracking_viz(self, c):
        self.show_landmarks = c

    def reset_transform(self, prefix):
        """Unified reset function for both Bracelet (B) and Hand (H)"""
        print(f"Resetting {prefix} Transform...")
        self.viewer.blockSignals(True)
        for key, info in self.sliders.items():
            if key.startswith(f"{prefix}_"):
                info['obj'].setValue(info['default'])
        self.viewer.blockSignals(False)
        self.viewer.update()

    def loop(self):
        if not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return
        frame_disp = cv2.flip(frame, 0)
        
        if self.use_ai:
            rvec, tvec, info = self.tracker.process(frame_disp)
            if self.show_landmarks:
                frame_disp = self.tracker.draw_debug(frame_disp)
            
            if info['found']:
                # Check if it is a Left Hand
                is_left = (info['hand'] == "Left")
                # Use "J" prefix for Jewelry, "H" for Hand
                # PASS 'is_left' TO THE FUNCTION
                self.viewer.model_bracelet = self.compute_matrix(
                    rvec, tvec, "J", ai=True, is_left=is_left
                )
                
                # Occluder also needs to know!
                self.viewer.model_occluder = self.compute_matrix(
                    rvec, tvec, "H", ai=True, is_left=is_left
                )
                
                self.txt.setText(f"Hand: {info['hand']} | Depth: {info['z_depth']:.1f}")
            else: 
                self.txt.setText("Searching...")
        else:
            self.viewer.model_bracelet = self.compute_matrix(None, None, "J", ai=False)
            self.viewer.model_occluder = self.compute_matrix(None, None, "H", ai=False)
            self.txt.setText("Manual Mode")
        
        self.viewer.update_bg(frame_disp)
        self.viewer.update()

    def compute_matrix(self, rvec, tvec, prefix, ai=False, is_left=False):
        def val(n): 
            key = f"{prefix}_{n}"
            if key in self.sliders:
                return self.sliders[key]['obj'].value() * self.sliders[key]['scale']
            return 0.0
            
        # Base Transformation
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

        # Scale
        scale_val = val("Scale"); 
        if scale_val == 0: scale_val = 1.0
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
        
        # Unified Rotation Order
        M_rot = np.eye(4, dtype=np.float32)
        M_rot[:3, :3] = Rx @ Ry @ Rz 
        
        # 5. TRANSLATE (The "Bead Slide") - "Outer Layer"
        # We ONLY translate along Y (The Bone Axis)
        # We force X and Z to be 0 so it can never leave the line
        slide_dist = val("Slide") 
        T_slide = np.eye(4, dtype=np.float32)
        T_slide[:3, 3] = [0, slide_dist, 0] 
        
        # 6. Final Multiplication Order
        # Scale -> Rotate -> Translate -> Base(AI) -> FixCoords
        return cv_to_gl @ T_base @ T_slide @ M_rot @ S
    
    def closeEvent(self, event):
        """Called when the window is being closed."""
        # 1. Save the state
        self.save_current_settings_to_db() # Save one last time
        
        # 2. Release the camera resource
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        # 3. Stop the timer to prevent background processing
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        event.accept()