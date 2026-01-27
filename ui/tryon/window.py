# ui/tryon/window.py
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QShortcut, QPushButton
from PyQt5.QtCore import pyqtSlot, QTimer, Qt, QEvent, pyqtSignal
from PyQt5.QtGui import QKeySequence

from graphics.renderer import ARViewerWidget
from workers.camera import CameraWorker
from workers.ai import AIWorker
from .controls import TryOnControls

from trackers.strategies.wrist import WristStrategy
from trackers.strategies.ring import RingStrategy
from trackers.strategies.waist import WaistStrategy
from trackers.strategies.neck import NeckStrategy
from trackers.strategies.nose import NoseStrategy
from trackers.strategies.ear import EarringStrategy
from trackers.strategies.forehead import ForeheadStrategy
from trackers.strategies.collection import CollectionStrategy

class TryOnWindow(QWidget): 
    # Signal to tell Main to switch screens
    back_clicked = pyqtSignal()

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_item = None
        self.pending_item = None
        self.use_ai = False

        # --- LAYOUT & UI ---
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0) 

        # 1. Viewer
        self.viewer = ARViewerWidget()
        self.viewer.installEventFilter(self)
        self.main_layout.addWidget(self.viewer, stretch=1) 
        
        # --- FLOATING BUTTONS ---
        # A. Fullscreen Button (Top-Right)
        self.btn_fullscreen = QPushButton("⛶", self.viewer)
        self.btn_fullscreen.setCursor(Qt.PointingHandCursor)
        self.btn_fullscreen.clicked.connect(self.toggle_cinema_mode)
        self.btn_fullscreen.setFixedSize(40, 40)
        self.btn_fullscreen.show()

        # B. Back Button (Top-Left) - NEW
        self.btn_back = QPushButton("← Collection", self.viewer)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self.back_clicked.emit) # Emits signal
        self.btn_back.setFixedSize(140, 40)
        self.btn_back.show()

        # Shared Style for Floating Buttons
        float_style = """
            QPushButton {
                background-color: rgba(0, 0, 0, 0.4);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #D4AF37; 
                color: black;
                border: 1px solid #D4AF37;
            }
        """
        self.btn_fullscreen.setStyleSheet(float_style)
        self.btn_back.setStyleSheet(float_style)
        
        # 2. Controls
        self.controls = TryOnControls()
        self.main_layout.addWidget(self.controls, stretch=0)

        # Threads
        self.cam_worker = CameraWorker(camera_index=0)
        self.ai_worker = AIWorker()

        self.setup_connections()
        
        # Shortcuts
        self.shortcut_f11 = QShortcut(QKeySequence("F11"), self, context=Qt.ApplicationShortcut)
        self.shortcut_f11.activated.connect(self.toggle_cinema_mode)
        self.shortcut_f = QShortcut(QKeySequence("F"), self, context=Qt.ApplicationShortcut)
        self.shortcut_f.activated.connect(self.toggle_cinema_mode)

    def resizeEvent(self, event):
        """Update floating button positions on resize."""
        super().resizeEvent(event)
        padding = 20
        
        # Back Button: Top-Left
        self.btn_back.move(padding, padding)
        
        # Fullscreen Button: Top-Right
        x = self.viewer.width() - self.btn_fullscreen.width() - padding
        self.btn_fullscreen.move(x, padding)

    def eventFilter(self, source, event):
        if source == self.viewer and event.type() == QEvent.MouseButtonDblClick:
            self.toggle_cinema_mode()
            return True 
        return super().eventFilter(source, event)

    def setup_connections(self):
        self.cam_worker.frame_captured.connect(self.on_frame_received)
        self.ai_worker.results_ready.connect(self.on_ai_results)
        self.controls.setting_changed.connect(self.on_setting_changed)
        self.controls.component_changed.connect(self.on_component_changed)
        self.controls.ai_toggled.connect(self.on_ai_toggled)
        self.controls.mask_toggled.connect(self.on_mask_toggled)
        self.controls.save_requested.connect(self.save_settings)
        self.controls.setting_changed.connect(self.update_scene_params)

    # --- SESSION ---
    def start_session(self):
        if not self.cam_worker.isRunning():
            self.cam_worker = CameraWorker(0) 
            self.cam_worker.frame_captured.connect(self.on_frame_received)
            self.cam_worker.start()
        if not self.ai_worker.isRunning():
            self.ai_worker.start()

    def stop_session(self):
        self.save_settings()
        self.cam_worker.stop()
        self.ai_worker.stop()

    # --- WORKER SLOTS ---
    @pyqtSlot(np.ndarray)
    def on_frame_received(self, frame):
        self.viewer.update_bg(frame)
        if self.use_ai: self.ai_worker.process_frame(frame)

    @pyqtSlot(list, bool)
    def on_ai_results(self, commands, face_found):
        self.viewer.render_instances = []
        self.viewer.occluder_instances = []
        for cmd in commands:
            if cmd['type'] == 'mesh': self.viewer.render_instances.append(cmd)
            elif cmd['type'] == 'occluder': self.viewer.occluder_instances.append(cmd['matrix'])
        self.viewer.update()

    # --- LOGIC ---
    def set_active_item(self, item):
        self.save_settings()
        self.current_item = item
        self.viewer.clear_scene()
        strategy = self._get_strategy_for_category(item.category)
        if strategy and item.settings:
            if hasattr(strategy, 'update_settings'): strategy.update_settings(item.settings)
            else: strategy.settings.update(item.settings)

        self.ai_worker.set_strategy(strategy)
        self.controls.rebuild_sliders(strategy)
        self.pending_item = item
        QTimer.singleShot(50, self._process_pending_loads)

    def _get_strategy_for_category(self, cat):
        if cat == "Collection": return CollectionStrategy()
        if cat == "Bracelet": return WristStrategy()
        if cat == "Ring": return RingStrategy()
        if cat == "Waist Band": return WaistStrategy()
        if cat == "Necklace": return NeckStrategy()
        if cat == "Nose Pin": return NoseStrategy()
        if cat == "Earring": return EarringStrategy()
        if cat == "Forehead Pendant": return ForeheadStrategy()
        return None

    def _process_pending_loads(self):
        """Smart Loader: Handles Collections with Sub-Folders."""
        if not self.pending_item: return
        path = self.pending_item.model_path
        
        if os.path.isdir(path):
            print(f"Loading Collection from Directory: {path}")
            found_parts = {}
            
            # Helper to guess body part from text
            def classify_part(name):
                lower = name.lower()
                if "neck" in lower: return "neck"
                if "ear" in lower: return "ear"
                if "nose" in lower: return "nose"
                if "tikka" in lower or "fore" in lower: return "forehead"
                if "bindi" in lower: return "forehead"
                return None

            # 1. SCAN SUB-DIRECTORIES (Preferred Method)
            # Looks for folders like "Necklace", "Earrings" inside the collection folder
            for entry in os.scandir(path):
                if entry.is_dir():
                    key = classify_part(entry.name)
                    if key:
                        # Find the first .obj inside this subfolder
                        for sub_f in os.listdir(entry.path):
                            if sub_f.lower().endswith('.obj'):
                                full_path = os.path.join(entry.path, sub_f)
                                found_parts[key] = full_path
                                # Key allows Renderer to identify which mesh is which
                                self.viewer.load_object(full_path, key=key)
                                print(f"  -> Found {key}: {sub_f}")
                                break # Found one obj for this part, move to next folder

            # 2. SCAN ROOT FILES (Fallback Method)
            # If you still have loose files in the root, we catch them here
            for fname in os.listdir(path):
                if fname.lower().endswith('.obj'):
                    key = classify_part(fname)
                    # Only add if we didn't already find this part in a folder
                    if key and key not in found_parts:
                        full_path = os.path.join(path, fname)
                        found_parts[key] = full_path
                        self.viewer.load_object(full_path, key=key)
                        print(f"  -> Found {key} (Root): {fname}")

            # 3. Pass to Strategy
            if isinstance(self.ai_worker.strategy, CollectionStrategy):
                self.ai_worker.strategy.load_components(found_parts)
                
                if self.pending_item.settings:
                    print(f"Restoring Collection Settings: {self.pending_item.settings}")
                    self.ai_worker.strategy.update_settings(self.pending_item.settings)
                self.controls.rebuild_sliders(self.ai_worker.strategy)
        
        else:
            # Single Item File
            print(f"Loading Single: {path}")
            self.viewer.load_object(path, key='default')

        self.pending_item = None

    def on_setting_changed(self, key, val): self.ai_worker.update_settings({key: val})
    def on_component_changed(self, comp_name):
        if hasattr(self.ai_worker.strategy, 'set_active_component'):
            self.ai_worker.strategy.set_active_component(comp_name)
            self.controls.rebuild_sliders(self.ai_worker.strategy)
    def on_ai_toggled(self, enabled): self.use_ai = enabled
    def on_mask_toggled(self, enabled): self.viewer.debug_occluder = enabled
    def update_scene_params(self, key, val):
        if key == "Cam_FOV": self.viewer.fov = val
        elif key == "Light_Exp": self.viewer.exposure = val * 0.1
        elif key == "Light_Gam": self.viewer.gamma = val * 0.1
        elif key == "Light_Amb": self.viewer.ambient_str = val * 0.01
        self.viewer.update()
    def save_settings(self):
        if self.current_item and self.ai_worker.strategy:
            self.db.update_item_settings(self.current_item.id, self.ai_worker.strategy.settings)

    def toggle_cinema_mode(self):
        """Toggles Fullscreen and visibility of UI elements."""
        if self.isFullScreen():
            self.showNormal()
            self.controls.show()
            self.btn_back.show() # Show Back Button
            self.btn_fullscreen.setText("⛶")
        else:
            self.showFullScreen()
            self.controls.hide()
            self.btn_back.hide() # Hide Back Button
            self.btn_fullscreen.setText("✖")
        
        # Force resize to update button positions immediately
        self.resizeEvent(None)