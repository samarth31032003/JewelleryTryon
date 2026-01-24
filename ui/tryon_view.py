# ui/tryon_view.py
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QGroupBox, QGridLayout, 
                             QCheckBox, QTextEdit, QScrollArea, QTabWidget,
                             QComboBox)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

from graphics.renderer import ARViewerWidget
from workers.camera import CameraWorker
from workers.ai import AIWorker

# Strategies
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
        self.current_item = None
        self.sliders = {} 
        self.pending_item = None
        self.combo_component = None 
        self.use_ai = False

        self.setup_ui()
        
        # --- THREAD SETUP ---
        # 1. Create Workers
        self.cam_worker = CameraWorker(camera_index=0)
        self.ai_worker = AIWorker()

        # 2. Connect Camera -> UI (Background Update)
        self.cam_worker.frame_captured.connect(self.on_frame_received)
        
        # 3. Connect AI -> UI (Mesh Update)
        self.ai_worker.results_ready.connect(self.on_ai_results)

    def start_session(self):
        """Called by Main when switching TO this screen."""
        if not self.cam_worker.isRunning():
            self.cam_worker = CameraWorker(0) # Re-init to be safe
            self.cam_worker.frame_captured.connect(self.on_frame_received)
            self.cam_worker.start()
        
        if not self.ai_worker.isRunning():
            self.ai_worker.start()

    def stop_session(self):
        """Called by Main when switching AWAY from this screen."""
        self.cam_worker.stop()
        self.ai_worker.stop()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Viewer
        self.viewer = ARViewerWidget()
        layout.addWidget(self.viewer, stretch=3)
        
        # Right: Controls
        self.panel = QWidget()
        self.p_layout = QVBoxLayout(self.panel)
        layout.addWidget(self.panel, stretch=1)
        tabs = QTabWidget()
        self.p_layout.addWidget(tabs)

        # Tab 1: Adjustments
        self.tab_item = QWidget()
        self.item_layout = QVBoxLayout(self.tab_item)
        
        self.combo_component = QComboBox()
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
        self.chk_mask.toggled.connect(self.toggle_mask)
        ctrl_layout.addWidget(self.chk_mask)
        
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.clicked.connect(self.save_settings)
        ctrl_layout.addWidget(self.btn_save)

        self.item_layout.addLayout(ctrl_layout)
        self.txt = QTextEdit(); self.txt.setMaximumHeight(50)
        self.item_layout.addWidget(self.txt)
        
        tabs.addTab(self.tab_item, "Item Adjustments")
        
        # Tab 2: Scene
        self.tab_scene = QWidget()
        scene_layout = QVBoxLayout(self.tab_scene)
        cam_grid = QGridLayout()
        self.add_sliders("Cam", [("FOV", 20, 120, 45, 1.0)], cam_grid, self.update_scene)
        grp_cam = QGroupBox("Camera"); grp_cam.setLayout(cam_grid)
        scene_layout.addWidget(grp_cam)
        
        light_grid = QGridLayout()
        light_params = [("Exp", 1, 50, 10, 0.1), ("Gam", 10, 30, 22, 0.1), ("Amb", 0, 100, 40, 0.01)]
        self.add_sliders("Light", light_params, light_grid, self.update_scene)
        grp_light = QGroupBox("Lighting"); grp_light.setLayout(light_grid)
        scene_layout.addWidget(grp_light)
        scene_layout.addStretch()
        tabs.addTab(self.tab_scene, "Scene Settings")

    # --- SIGNAL SLOTS (The New "Loop") ---
    
    @pyqtSlot(np.ndarray)
    def on_frame_received(self, frame):
        """Called every time CameraWorker has a new frame."""
        # 1. Update Background (Instant)
        self.viewer.update_bg(frame)
        
        # 2. Send to AI (if enabled)
        if self.use_ai:
            self.ai_worker.process_frame(frame)
        else:
            self.txt.setText("AI Paused")

    @pyqtSlot(list, bool)
    def on_ai_results(self, commands, face_found):
        """Called when AIWorker finishes math."""
        # 1. Update Renderer Lists
        self.viewer.render_instances = []
        self.viewer.occluder_instances = []
        
        count = 0
        for cmd in commands:
            if cmd['type'] == 'mesh':
                self.viewer.render_instances.append(cmd)
                count += 1
            elif cmd['type'] == 'occluder':
                self.viewer.occluder_instances.append(cmd['matrix'])
        
        # 2. Trigger Repaint
        self.viewer.update()
        self.txt.setText(f"Active Objects: {count} | Face: {'Yes' if face_found else 'No'}")

    # --- LOGIC CONTROL ---

    def set_active_item(self, item):
        # 1. Save Old
        if self.current_item and self.ai_worker.strategy:
            self.save_settings()

        # 2. Clear Scene
        self.viewer.clear_scene()
        self.current_item = item
        
        # 3. Create Strategy
        strategy = None
        if item.category == "Collection": strategy = CollectionStrategy()
        elif item.category == "Bracelet": strategy = WristStrategy()
        elif item.category == "Ring": strategy = RingStrategy()
        elif item.category == "Waist Band": strategy = WaistStrategy()
        elif item.category == "Necklace": strategy = NeckStrategy()
        elif item.category == "Nose Pin": strategy = NoseStrategy()
        elif item.category == "Earring": strategy = EarringStrategy()
        elif item.category == "Forehead Pendant": strategy = ForeheadStrategy()
        
        # 4. Pass Strategy to AI Worker
        if strategy:
            if item.settings:
                if hasattr(strategy, 'update_settings'):
                    strategy.update_settings(item.settings)
                else:
                    strategy.settings.update(item.settings)
            
            self.ai_worker.set_strategy(strategy)
        
        # 5. Queue Loading
        self.pending_item = item
        self.rebuild_controls()
        
        if self.isVisible():
            QTimer.singleShot(50, self._process_pending_loads)

    def _process_pending_loads(self):
        """Smart Loader: Handles Files AND Directories."""
        if not self.pending_item: return
        path = self.pending_item.model_path
        
        if os.path.isdir(path):
            print(f"Loading Collection: {path}")
            # Scan Folder (needs to check atleast 2 models and max 4 one for each.)
            found_parts = {}
            for fname in os.listdir(path):
                if fname.lower().endswith('.obj'):
                    full_path = os.path.join(path, fname)
                    lower = fname.lower()
                    key = None
                    if "neck" in lower: key = "neck"
                    elif "ear" in lower: key = "ear"
                    elif "nose" in lower: key = "nose"
                    elif "tikka" in lower or "fore" in lower: key = "forehead"
                    
                    if key:
                        found_parts[key] = full_path
                        self.viewer.load_object(full_path, key=key)
            
            # Update Strategy with found parts
            if isinstance(self.ai_worker.strategy, CollectionStrategy):
                self.ai_worker.strategy.load_components(found_parts)
                self.rebuild_controls()
        else:
            print(f"Loading Single: {path}")
            self.viewer.load_object(path, key='default')

        self.pending_item = None

    def on_slider_change(self, key, value):
        # Update AI Worker safely
        self.ai_worker.update_settings({key: value})

    def save_settings(self):
        """Gathers all slider values and updates the DB."""
        if self.current_item and self.ai_worker.strategy:
            self.db.update_item_settings(self.current_item.id, self.ai_worker.strategy.settings)
            print("Settings Saved!")

    def toggle_ai(self, checked):
        self.use_ai = checked
        self.btn_ai.setText("AI TRACKING: ON" if checked else "Enable AI")

    def rebuild_controls(self):
        """Rebuilds sliders. Handles Dropdown logic for Collections."""
        # Clear Old
        while self.dynamic_layout.count():
            w = self.dynamic_layout.takeAt(0).widget()
            if w: w.setParent(None)
        
        # Cleanup Sliders dict
        scene_prefixes = ("Cam_", "Light_")
        self.sliders = {k:v for k,v in self.sliders.items() if k.startswith(scene_prefixes)}

        if not self.ai_worker.strategy: return
        strat = self.ai_worker.strategy

        # Handle Dropdown
        if isinstance(strat, CollectionStrategy):
            self.combo_component.setVisible(True)
            current_list = strat.get_component_list()
            self.combo_component.blockSignals(True)
            self.combo_component.clear()
            self.combo_component.addItems(current_list)
            if strat.active_component:
                idx = self.combo_component.findText(strat.active_component)
                if idx >= 0: self.combo_component.setCurrentIndex(idx)
            self.combo_component.blockSignals(False)
        else:
            self.combo_component.setVisible(False)

        # Build Sliders
        defs = strat.get_slider_definitions()
        grid = QGridLayout()
        row = 0
        for key, label, min_v, max_v, def_v, _ in defs:
            # Get Value
            if isinstance(strat, CollectionStrategy):
                comp = strat.active_component
                val = def_v
                if comp and comp in strat.settings:
                    val = strat.settings[comp].get(key, def_v)
            else:
                val = strat.settings.get(key, def_v)

            lbl = QLabel(label)
            sld = QSlider(Qt.Horizontal)
            sld.setRange(min_v, max_v)
            sld.setValue(int(val))
            sld.valueChanged.connect(lambda v, k=key: self.on_slider_change(k, v))
            
            self.sliders[key] = {'obj': sld, 'default': def_v}
            grid.addWidget(lbl, row, 0); grid.addWidget(sld, row, 1)
            row += 1

        grp = QGroupBox(f"Adjustments")
        grp.setLayout(grid)
        self.dynamic_layout.addWidget(grp)
        self.dynamic_layout.addStretch()

    def on_component_switch(self, index):
        if isinstance(self.ai_worker.strategy, CollectionStrategy):
            key = self.combo_component.currentText()
            if key:
                self.ai_worker.strategy.set_active_component(key)
                self.rebuild_controls()

    # --- HELPERS ---
    def toggle_mask(self, checked):
        self.viewer.debug_occluder = checked
        self.viewer.update()

    def add_sliders(self, prefix, params, layout, callback=None):
        for i, (n, min_v, max_v, def_v, scale) in enumerate(params):
            key = f"{prefix}_{n}"
            l = QLabel(n); s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v)
            if callback: s.valueChanged.connect(callback)
            self.sliders[key] = {'obj': s, 'scale': scale, 'default': def_v}
            layout.addWidget(l, i, 0); layout.addWidget(s, i, 1)

    def update_scene(self):
        if "Cam_FOV" in self.sliders:
            self.viewer.fov = self.sliders["Cam_FOV"]['obj'].value()
        if "Light_Exp" in self.sliders:
            self.viewer.exposure = self.sliders["Light_Exp"]['obj'].value() * 0.1
            self.viewer.gamma = self.sliders["Light_Gam"]['obj'].value() * 0.1
            self.viewer.ambient_str = self.sliders["Light_Amb"]['obj'].value() * 0.01
        self.viewer.update()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self._process_pending_loads)

    def closeEvent(self, event):
        """Clean shutdown of threads."""
        self.save_settings()
        self.cam_worker.stop()
        self.ai_worker.stop()
        event.accept()