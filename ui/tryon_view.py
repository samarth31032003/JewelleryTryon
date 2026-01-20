# ui/tryon_view.py
import cv2
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QGroupBox, QGridLayout, 
                             QCheckBox, QTextEdit, QScrollArea, QTabWidget,
                             QComboBox)
from PyQt5.QtCore import Qt, QTimer

from graphics.renderer import ARViewerWidget
from trackers.pose_engine import PoseEngine

from trackers.strategies.wrist import WristStrategy
from trackers.strategies.ring import RingStrategy
from trackers.strategies.waist import WaistStrategy
from trackers.strategies.neck import NeckStrategy
from trackers.strategies.nose import NoseStrategy
from trackers.strategies.ear import EarringStrategy
from trackers.strategies.forehead import ForeheadStrategy
from trackers.strategies.collection import CollectionStrategy

class TryOnWindow(QWidget): 
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.ai_engine = PoseEngine()
        self.strategy = None
        self.current_item = None
        self.sliders = {} 
        self.pending_item = None
        
        # New State for Dropdown
        self.combo_component = None 

        self.setup_ui()
        
        self.cap = cv2.VideoCapture(0)
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

        # --- TAB 1: ITEM SETTINGS ---
        self.tab_item = QWidget()
        self.item_layout = QVBoxLayout(self.tab_item)
        
        # NEW: Component Dropdown (Hidden by default)
        self.combo_component = QComboBox()
        self.combo_component.setStyleSheet("padding: 5px; font-weight: bold;")
        self.combo_component.currentIndexChanged.connect(self.on_component_switch)
        self.combo_component.setVisible(False) 
        self.item_layout.addWidget(self.combo_component)

        # 1. Dynamic Slider Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.dynamic_container = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_container)
        scroll.setWidget(self.dynamic_container)
        self.item_layout.addWidget(scroll)

        # 2. Controls Group
        ctrl_layout = QVBoxLayout()
        self.btn_ai = QPushButton("Enable AI")
        self.btn_ai.setCheckable(True)
        self.btn_ai.setStyleSheet("background-color: #555; color: white; padding: 8px;")
        self.btn_ai.toggled.connect(self.toggle_ai)
        ctrl_layout.addWidget(self.btn_ai)

        self.chk_mask = QCheckBox("Show Occlusion Mask (Debug)")
        self.chk_mask.setChecked(False)
        self.chk_mask.setStyleSheet("color: #DDD; font-weight: bold; margin: 5px;")
        self.chk_mask.toggled.connect(self.toggle_mask)
        ctrl_layout.addWidget(self.chk_mask)
        
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold; padding: 8px;")
        self.btn_save.clicked.connect(self.save_settings)
        ctrl_layout.addWidget(self.btn_save)

        self.item_layout.addLayout(ctrl_layout)
        self.txt = QTextEdit(); self.txt.setMaximumHeight(50)
        self.item_layout.addWidget(self.txt)
        
        tabs.addTab(self.tab_item, "Item Adjustments")
        
        # --- TAB 2: SCENE ---
        self.tab_scene = QWidget()
        scene_layout = QVBoxLayout(self.tab_scene)
        
        # Camera
        cam_grid = QGridLayout()
        self.add_sliders("Cam", [("FOV", 20, 120, 45, 1.0)], cam_grid, self.update_scene)
        grp_cam = QGroupBox("Camera"); grp_cam.setLayout(cam_grid)
        scene_layout.addWidget(grp_cam)
        
        # Lighting
        light_grid = QGridLayout()
        light_params = [("Exp", 1, 50, 10, 0.1), ("Gam", 10, 30, 22, 0.1), ("Amb", 0, 100, 40, 0.01)]
        self.add_sliders("Light", light_params, light_grid, self.update_scene)
        grp_light = QGroupBox("Lighting"); grp_light.setLayout(light_grid)
        scene_layout.addWidget(grp_light)
        
        scene_layout.addStretch()
        tabs.addTab(self.tab_scene, "Scene Settings")

    # --- NEW: Dropdown Handler ---
    def on_component_switch(self, index):
        """Called when user selects a different part of the collection."""
        if not self.strategy or not isinstance(self.strategy, CollectionStrategy):
            return
            
        key = self.combo_component.currentText()
        if key:
            self.strategy.set_active_component(key)
            self.rebuild_controls() # Re-draw sliders for this component

    def toggle_mask(self, checked):
        self.viewer.debug_occluder = checked
        self.viewer.update()

    def update_scene(self):
        if "Cam_FOV" in self.sliders:
            self.viewer.fov = self.sliders["Cam_FOV"]['obj'].value()
        if "Light_Exp" in self.sliders:
            self.viewer.exposure = self.sliders["Light_Exp"]['obj'].value() * 0.1
            self.viewer.gamma = self.sliders["Light_Gam"]['obj'].value() * 0.1
            self.viewer.ambient_str = self.sliders["Light_Amb"]['obj'].value() * 0.01
        self.viewer.update()

    def set_active_item(self, item):
        # 1. Save Old Settings
        if self.current_item and self.strategy:
            self.save_settings()

        # 2. CLEAR GPU MEMORY (Crucial for Phase 1)
        self.viewer.clear_scene()

        self.current_item = item
        
        # 3. Strategy Selection
        if item.category == "Collection":
            self.strategy = CollectionStrategy()
        elif item.category == "Bracelet":
            self.strategy = WristStrategy()
        elif item.category == "Ring":
            self.strategy = RingStrategy()
        elif item.category == "Waist Band":
            self.strategy = WaistStrategy()
        elif item.category == "Necklace":
            self.strategy = NeckStrategy()
        elif item.category == "Nose Pin":
            self.strategy = NoseStrategy()
        elif item.category == "Earring":
            self.strategy = EarringStrategy()
        elif item.category == "Forehead Pendant": 
            self.strategy = ForeheadStrategy()
        
        # 4. Queue Loading (Directory vs File)
        self.pending_item = item
        
        # 5. Restore Settings
        if item.settings:
            self.strategy.update_settings(item.settings)

        # 6. UI Update
        self.rebuild_controls()
        if self.isVisible():
            self._process_pending_loads()

    def showEvent(self, event):
        """Called automatically when window opens."""
        super().showEvent(event)
        if not self.timer.isActive():
            self.timer.start(30)
        QTimer.singleShot(50, self._process_pending_loads)

    def _process_pending_loads(self):
        """Smart Loader: Handles Files AND Directories."""
        if not self.pending_item: return
        
        path = self.pending_item.model_path
        
        # Check if Directory (Collection)
        if os.path.isdir(path):
            print(f"Loading Collection from: {path}")
            
            # Scan Folder (needs to check atleast 2 models and max 4 one for each.)
            found_parts = {}
            for fname in os.listdir(path):
                if fname.lower().endswith('.obj'):
                    full_path = os.path.join(path, fname)
                    # Simple Keyword Matching
                    lower_name = fname.lower()
                    key = None
                    if "neck" in lower_name: key = "neck"
                    elif "ear" in lower_name: key = "ear"
                    elif "nose" in lower_name: key = "nose"
                    elif "tikka" in lower_name or "fore" in lower_name: key = "forehead"
                    
                    if key:
                        found_parts[key] = full_path
                        # Load into Renderer with specific Key
                        self.viewer.load_object(full_path, key=key)
            
            # Init Strategy with found parts
            if isinstance(self.strategy, CollectionStrategy):
                self.strategy.load_components(found_parts)
                # Update UI Dropdown
                self.rebuild_controls()

        else:
            # Single File Mode
            print(f"Loading Single Item: {path}")
            self.viewer.load_object(path, key='default')

        self.pending_item = None

    def save_settings(self):
        """Gathers all slider values and updates the DB."""
        if self.current_item and self.strategy:
            self.db.update_item_settings(self.current_item.id, self.strategy.settings)
            print("Settings Saved!")

    def add_sliders(self, prefix, params, layout, callback=None):
        for i, (n, min_v, max_v, def_v, scale) in enumerate(params):
            key = f"{prefix}_{n}"
            l = QLabel(n); s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v)
            if callback: s.valueChanged.connect(callback)
            self.sliders[key] = {'obj': s, 'scale': scale, 'default': def_v}
            layout.addWidget(l, i, 0); layout.addWidget(s, i, 1)

    def rebuild_controls(self):
        """Rebuilds sliders. Handles Dropdown logic for Collections."""
        # 1. Clear Old Sliders
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.setParent(None)
        
        # 2. Clean up self.sliders (Remove non-scene keys)
        # We assume Scene keys start with "Cam_" or "Light_"
        scene_prefixes = ("Cam_", "Light_")
        keys_to_remove = [k for k in self.sliders if not k.startswith(scene_prefixes)]
        for k in keys_to_remove: del self.sliders[k]

        if not self.strategy: return

        # 2. Handle Dropdown Visibility
        if isinstance(self.strategy, CollectionStrategy):
            self.combo_component.setVisible(True)
            
            # Populate if empty or changed
            current_list = self.strategy.get_component_list()
            # If combo items don't match strategy items, refresh
            combo_items = [self.combo_component.itemText(i) for i in range(self.combo_component.count())]
            if set(current_list) != set(combo_items):
                self.combo_component.blockSignals(True)
                self.combo_component.clear()
                self.combo_component.addItems(current_list)
                # Select active
                if self.strategy.active_component:
                    idx = self.combo_component.findText(self.strategy.active_component)
                    if idx >= 0: self.combo_component.setCurrentIndex(idx)
                self.combo_component.blockSignals(False)
        else:
            self.combo_component.setVisible(False)

        # 3. Add Sliders
        defs = self.strategy.get_slider_definitions()
        grid = QGridLayout()
        row = 0
        for key, label, min_v, max_v, def_v, _ in defs:
            # Handle nested settings for Collections? 
            # Strategy.settings might be {scale: 100} OR {neck: {scale: 100}}
            # But get_slider_definitions returns generic keys.
            # Strategy.settings[key] access handles this abstraction in base/collection class.
            
            # For CollectionStrategy, self.strategy.settings is the MASTER dict {neck: {...}}
            # But we want the value for the ACTIVE component.
            # We need a helper to get the value.
            
            if isinstance(self.strategy, CollectionStrategy):
                # Dig into sub-strategy settings
                comp = self.strategy.active_component
                val = def_v
                if comp and comp in self.strategy.settings:
                    val = self.strategy.settings[comp].get(key, def_v)
            else:
                val = self.strategy.settings.get(key, def_v)
            
            lbl = QLabel(label)
            sld = QSlider(Qt.Horizontal)
            sld.setRange(min_v, max_v)
            sld.setValue(int(val))
            sld.valueChanged.connect(lambda val, k=key: self.on_slider_change(k, val))
            
            self.sliders[key] = {'obj': sld, 'default': def_v}
            grid.addWidget(lbl, row, 0); grid.addWidget(sld, row, 1)
            row += 1

        grp = QGroupBox(f"Adjustments: {self.strategy.active_component if isinstance(self.strategy, CollectionStrategy) else ''}")
        grp.setLayout(grid)
        self.dynamic_layout.addWidget(grp)
        self.dynamic_layout.addStretch()

    def on_slider_change(self, key, value):
        # Update the Strategy's memory immediately
        if self.strategy:
            self.strategy.update_settings({key: value})

    def toggle_ai(self, checked):
        self.use_ai = checked
        self.btn_ai.setText("AI TRACKING: ON" if checked else "Enable AI")

    def loop(self):
        if not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return
        
        # Reset Render Lists
        self.viewer.render_instances = []
        self.viewer.occluder_instances = []
        
        if getattr(self, 'use_ai', False) and self.strategy:
            results = self.ai_engine.process(frame)
            h, w, _ = frame.shape
            
            commands = self.strategy.process_frame(results, w, h)
            
            count = 0
            for cmd in commands:
                if cmd['type'] == 'mesh':
                    # Phase 2: Renderer now accepts Dicts with 'model_key'
                    # CollectionStrategy adds 'model_key' to cmd
                    self.viewer.render_instances.append(cmd) 
                    count += 1
                elif cmd['type'] == 'occluder':
                    self.viewer.occluder_instances.append(cmd['matrix'])
            
            self.txt.setText(f"Active Objects: {count}")
        else:
            self.txt.setText("AI Paused")

        self.viewer.update_bg(frame)
        self.viewer.update()
        
    def closeEvent(self, event):
        """Called when the window is being closed."""
        self.save_settings()
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        if hasattr(self, 'timer'): self.timer.stop()
        event.accept()