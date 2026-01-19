# ui/tryon_view.py
import cv2
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QGroupBox, QGridLayout, 
                             QCheckBox, QTextEdit, QScrollArea, QTabWidget)
from PyQt5.QtCore import Qt, QTimer

from graphics.renderer import ARViewerWidget
from trackers.pose_engine import PoseEngine
from trackers.strategies.wrist import WristStrategy
from trackers.strategies.ring import RingStrategy

class TryOnWindow(QWidget): 
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.ai_engine = PoseEngine()
        self.strategy = None
        self.current_item = None
        self.sliders = {} # Keep track of active sliders
        # --- LOADING STATE ---
        self.pending_item = None

        self.setup_ui()
        
        self.cap = cv2.VideoCapture() 
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)

    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # 1. Left: Viewer
        self.viewer = ARViewerWidget()
        layout.addWidget(self.viewer, stretch=3)
        
        # 2. Right: Controls Panel
        self.panel = QWidget()
        self.p_layout = QVBoxLayout(self.panel)
        layout.addWidget(self.panel, stretch=1)
        tabs = QTabWidget()
        self.p_layout.addWidget(tabs)

        # --- TAB 1: ITEM SETTINGS (Dynamic) ---
        self.tab_item = QWidget()
        self.item_layout = QVBoxLayout(self.tab_item)
        
        # 1. Dynamic Slider Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.dynamic_container = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_container)
        scroll.setWidget(self.dynamic_container)
        self.item_layout.addWidget(scroll)

        # 2. Controls Group (AI, Debug, Save)
        ctrl_layout = QVBoxLayout()
        
        # A. AI Toggle
        self.btn_ai = QPushButton("Enable AI")
        self.btn_ai.setCheckable(True)
        self.btn_ai.setStyleSheet("background-color: #555; color: white; padding: 8px;")
        self.btn_ai.toggled.connect(self.toggle_ai)
        ctrl_layout.addWidget(self.btn_ai)

        # B. Mask Visibility Checkbox (NEW)
        self.chk_mask = QCheckBox("Show Occlusion Mask (Debug)")
        self.chk_mask.setChecked(True) # Default to Visible for tuning
        self.chk_mask.setStyleSheet("color: #DDD; font-weight: bold; margin: 5px;")
        self.chk_mask.toggled.connect(self.toggle_mask)
        ctrl_layout.addWidget(self.chk_mask)
        
        # C. Save Button
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold; padding: 8px;")
        self.btn_save.clicked.connect(self.save_settings)
        ctrl_layout.addWidget(self.btn_save)

        self.item_layout.addLayout(ctrl_layout)
        
        # Debug Text
        self.txt = QTextEdit(); self.txt.setMaximumHeight(50)
        self.item_layout.addWidget(self.txt)
        
        tabs.addTab(self.tab_item, "Item Adjustments")
        
        # --- TAB 2: SCENE (Global) ---
        self.tab_scene = QWidget()
        scene_layout = QVBoxLayout(self.tab_scene)
        
        # Camera
        cam_grid = QGridLayout()
        # Key becomes "Cam_FOV"
        self.add_sliders("Cam", [("FOV", 20, 120, 45, 1.0)], cam_grid, self.update_scene)
        grp_cam = QGroupBox("Camera"); grp_cam.setLayout(cam_grid)
        scene_layout.addWidget(grp_cam)
        
        # Lighting
        light_grid = QGridLayout()
        # Keys become "Light_Exp", "Light_Gam", etc.
        light_params = [
            ("Exp", 1, 50, 10, 0.1),   # Exposure 0.1 - 5.0
            ("Gam", 10, 30, 22, 0.1),  # Gamma 1.0 - 3.0
            ("Amb", 0, 100, 40, 0.01)  # Ambient 0.0 - 1.0
        ]
        self.add_sliders("Light", light_params, light_grid, self.update_scene)
        grp_light = QGroupBox("Lighting"); grp_light.setLayout(light_grid)
        scene_layout.addWidget(grp_light)
        
        scene_layout.addStretch()
        tabs.addTab(self.tab_scene, "Scene Settings")

    def toggle_mask(self, checked):
        """Toggles between Red Cylinder (Debug) and Invisible Occluder (Production)"""
        self.viewer.debug_occluder = checked
        self.viewer.update()

    def update_scene(self):
        # Callback for Tab 2 sliders
        if "Cam_FOV" in self.sliders:
            self.viewer.fov = self.sliders["Cam_FOV"]['obj'].value()
            
        if "Light_Exp" in self.sliders:
            # Check if renderer supports exposure (it should with updated code)
            if hasattr(self.viewer, 'exposure'):
                self.viewer.exposure = self.sliders["Light_Exp"]['obj'].value() * 0.1
                self.viewer.gamma = self.sliders["Light_Gam"]['obj'].value() * 0.1
                self.viewer.ambient_str = self.sliders["Light_Amb"]['obj'].value() * 0.01
                
        self.viewer.update()

    def set_active_item(self, item):
        # 1. Save old
        if self.current_item and self.strategy:
            self.save_settings()

        self.current_item = item
        
        # 2. Select Strategy
        if item.category == "Bracelet":
            self.strategy = WristStrategy()
        elif item.category == "Ring":
            self.strategy = RingStrategy()
        # elif item.category == "Necklace": self.strategy = NeckStrategy()
        
        # 3. Queue the Item for loading
        self.pending_item = item
        
        # 4. Load Saved Settings into Strategy
        if item.settings:
            self.strategy.update_settings(item.settings)

        # 5. Rebuild UI for this Strategy
        self.rebuild_controls()

        # 5. Trigger Load if visible
        if self.isVisible():
            self._process_pending_loads()

    def showEvent(self, event):
        """Called automatically when window opens."""
        super().showEvent(event)
        # Wait 50ms for OpenGL context to be ready
        QTimer.singleShot(50, self._process_pending_loads)

    def _process_pending_loads(self):
        """Actually loads the mesh into OpenGL."""
        if self.pending_item:
            print(f"Loading Model: {self.pending_item.model_path}")
            self.viewer.load_object(self.pending_item.model_path, is_occluder=False)
            self.pending_item = None

    def save_settings(self):
        """Gathers all slider values and updates the DB."""
        if self.current_item and self.strategy:
            # Strategy.settings already has the latest values
            self.db.update_item_settings(self.current_item.id, self.strategy.settings)
            print("Settings Saved!")

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


    def toggle_ai(self, checked):
        self.use_ai = checked
        self.btn_ai.setText("AI TRACKING: ON" if checked else "Enable AI")

    def rebuild_controls(self):
        """Rebuilds ONLY the strategy sliders, keeping Scene sliders intact."""
        # 1. Safe Layout Clear (Handles Widgets AND Spacers)
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0) # Take the first item
            widget = item.widget()
            if widget:
                widget.setParent(None) # Remove widget
            # If it's a spacer (None), takeAt already removed it from the layout
        
        # 2. Clean up self.sliders (Remove non-scene keys)
        # We assume Scene keys start with "Cam_" or "Light_"
        scene_prefixes = ("Cam_", "Light_")
        keys_to_remove = [k for k in self.sliders if not k.startswith(scene_prefixes)]
        for k in keys_to_remove:
            del self.sliders[k]

        if not self.strategy: return

        # 3. Add new sliders
        # Get definitions: [(Key, Label, Min, Max, Default, Scale), ...]
        defs = self.strategy.get_slider_definitions()
        
        grid = QGridLayout()
        row = 0
        for key, label, min_v, max_v, def_v, _ in defs:
            # Check if we have a saved value for this key
            current_val = self.strategy.settings.get(key, def_v)
            
            lbl = QLabel(label)
            sld = QSlider(Qt.Horizontal)
            sld.setRange(min_v, max_v)
            sld.setValue(int(current_val))
            
            # Connect Signal
            # We use a closure (lambda) to capture the specific 'key'
            sld.valueChanged.connect(lambda val, k=key: self.on_slider_change(k, val))
            
            # Store in dict
            self.sliders[key] = {'obj': sld, 'default': def_v}
            
            grid.addWidget(lbl, row, 0)
            grid.addWidget(sld, row, 1)
            row += 1

        grp = QGroupBox("Adjustments")
        grp.setLayout(grid)
        self.dynamic_layout.addWidget(grp)
        self.dynamic_layout.addStretch()

    def on_slider_change(self, key, value):
        # Update the Strategy's memory immediately
        if self.strategy:
            self.strategy.update_settings({key: value})


    def loop(self):
        if not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return
        
        # Reset Render Lists
        self.viewer.render_instances = []
        self.viewer.occluder_instances = []
        
        if self.use_ai and self.strategy:
            # 1. AI Engine
            results = self.ai_engine.process(frame)
            h, w, _ = frame.shape
            
            # 2. Strategy Calculation (Returns Render Commands)
            commands = self.strategy.process_frame(results, w, h)
            
            count = 0
            for cmd in commands:
                if cmd['type'] == 'mesh':
                    self.viewer.render_instances.append(cmd['matrix'])
                    count += 1
                elif cmd['type'] == 'occluder':
                    self.viewer.occluder_instances.append(cmd['matrix'])
            
            self.txt.setText(f"Active Objects: {count}")
            
        else:
            self.txt.setText("Manual / No AI")

        self.viewer.update_bg(frame)
        self.viewer.update()
        
    def closeEvent(self, event):
        """Called when the window is being closed."""
        self.save_settings()

        # 2. Release the camera resource
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        # 3. Stop the timer to prevent background processing
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        event.accept()