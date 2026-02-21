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

# --- CONFIG ---
CAMERA_INDEX = 0

class TryOnWindow(QWidget): 
    # Signal to tell Main to switch screens
    back_clicked = pyqtSignal()

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_item = None
        self.pending_item = None
        self.use_ai = False
        self.scene_settings = {"Cam_FOV": 45}

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
        self.btn_fullscreen.setFixedSize(48, 48)
        self.btn_fullscreen.show()

        # B. Back Button (Top-Left) - NEW
        self.btn_back = QPushButton("← Collection", self.viewer)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self.back_clicked.emit) # Emits signal
        self.btn_back.setFixedSize(150, 44)
        self.btn_back.show()

        # Shared Style for Floating Buttons
        float_style = """
            QPushButton {
                background-color: rgba(15, 26, 43, 0.8);
                color: #e8f0ff;
                border: 1px solid #1f2c42;
                border-radius: 14px;
                font-weight: 700;
                font-size: 14px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #2f7ae5;
                color: white;
                border: 1px solid #2f7ae5;
            }
        """
        self.btn_fullscreen.setStyleSheet(float_style)
        self.btn_back.setStyleSheet(float_style)
        
        # 2. Controls
        self.controls = TryOnControls()
        self.main_layout.addWidget(self.controls, stretch=0)

        # Threads
        # creating CameraWorker here is wasteful.
        self.cam_worker = None 
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
        
        # 2. AI -> UI
        self.ai_worker.results_ready.connect(self.on_ai_results)
        
        # 3. Controls -> Logic
        self.controls.setting_changed.connect(self.on_setting_changed)
        self.controls.component_changed.connect(self.on_component_changed)
        self.controls.ai_toggled.connect(self.on_ai_toggled)
        self.controls.mask_toggled.connect(self.on_mask_toggled)
        self.controls.save_requested.connect(self.save_settings)
        
        # 4. Scene Settings
        self.controls.setting_changed.connect(self.update_scene_params)

    # --- SESSION MANAGEMENT ---
    
    def start_session(self):
        """Called when entering the screen."""
        # Check if None or not running
        if self.cam_worker is None or not self.cam_worker.isRunning():
            self.cam_worker = CameraWorker(CAMERA_INDEX) 
            self.cam_worker.frame_captured.connect(self.on_frame_received)
            self.cam_worker.start()
        
        if not self.ai_worker.isRunning():
            self.ai_worker.start()

    def stop_session(self):
        """Called when leaving (Back to Catalogue)."""
        self.save_settings() 
        if self.cam_worker:
            self.cam_worker.stop()
        self.ai_worker.stop()

    # --- WORKER SLOTS ---

    @pyqtSlot(np.ndarray)
    def on_frame_received(self, frame):
        self.viewer.update_bg(frame)
        if self.use_ai:
            self.ai_worker.process_frame(frame)

    @pyqtSlot(list, bool)
    def on_ai_results(self, commands, face_found):
        self.viewer.render_instances = []
        self.viewer.occluder_instances = []
        
        for cmd in commands:
            if cmd['type'] == 'mesh':
                self.viewer.render_instances.append(cmd)
            elif cmd['type'] == 'occluder':
                self.viewer.occluder_instances.append(cmd)
        
        self.viewer.update()

    # --- LOGIC CONTROL ---

    def set_active_item(self, item, mode="3d"):
        self.save_settings()
        self.current_item = item
        self.current_mode = mode # 2d/3d mode
        self.viewer.clear_scene()
        
        strategy = self._get_strategy_for_category(item.category)
        if strategy :
            strategy.mode = mode

            if item.settings:
                if hasattr(strategy, 'update_settings'):
                    strategy.update_settings(item.settings)
                else:
                    strategy.settings.update(item.settings)

        # Restore scene settings
        if item.settings and isinstance(item.settings, dict):
            scene_cfg = item.settings.get("Scene", {})
            if isinstance(scene_cfg, dict):
                self.scene_settings.update(scene_cfg)
                if "Cam_FOV" in scene_cfg:
                    self.viewer.fov = scene_cfg.get("Cam_FOV", self.viewer.fov)

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
        """Smart Loader: Handles Collections with Sub-Folders, Handles 2D Sprites, 3D Collections, and 3D Singles.."""
        if not self.pending_item: return

        path = self.pending_item.model_path
        # SCENARIO 1: IT IS A COLLECTION (FOLDER)
        if os.path.isdir(path):
            print(f"Loading Collection ({self.current_mode} mode): {path}")
            found_parts = {}
            
            # Decide what file extension we are hunting for
            target_ext = '.png' if self.current_mode == "2d" else '.obj'
            
            def classify_part(name):
                lower = name.lower()
                if "neck" in lower: return "neck"
                if "ear" in lower: return "ear"
                if "nose" in lower: return "nose"
                if "tikka" in lower or "fore" in lower: return "forehead"
                if "bindi" in lower: return "forehead"
                return None

            # Scan the folder
            for root, dirs, files in os.walk(path):
                for fname in files:
                    if fname.lower().endswith(target_ext):
                        # Use folder name OR file name to classify
                        # e.g., "ear/gold.obj" -> uses "ear" from folder name
                        # e.g., "gold_earring.png" -> uses "ear" from file name
                        folder_name = os.path.basename(root)
                        key = classify_part(folder_name) or classify_part(fname)
                        
                        if key and key not in found_parts:
                            full_path = os.path.join(root, fname)
                            found_parts[key] = full_path
                            
                            # Load it into the viewer using the correct renderer method
                            if self.current_mode == "2d":
                                self.viewer.load_quad(full_path, key=key)
                            else:
                                self.viewer.load_object(full_path, key=key)

            # 3. Pass to Strategy
            if isinstance(self.ai_worker.strategy, CollectionStrategy):
                self.ai_worker.strategy.load_components(found_parts)
                if self.pending_item.settings:
                    self.ai_worker.strategy.update_settings(self.pending_item.settings)
                self.controls.rebuild_sliders(self.ai_worker.strategy)
        
        # SCENARIO 2: SINGLE ITEM (NOT A FOLDER)
        else:
            if self.current_mode == "2d" and self.pending_item.image_2d_path:
                print(f"Loading Single 2D Sprite: {self.pending_item.image_2d_path}")
                self.viewer.load_quad(self.pending_item.image_2d_path, key='default')
            else:
                print(f"Loading Single 3d: {path}")
                # tex_path = getattr(self.pending_item, 'texture_path', None)
                # self.viewer.load_object(path, texture_path=tex_path, key='default')
                self.viewer.load_object(path, key='default') # no texture path yet.

        self.pending_item = None

    def on_setting_changed(self, key, val): self.ai_worker.update_settings({key: val})
    def on_component_changed(self, comp_name):
        if hasattr(self.ai_worker.strategy, 'set_active_component'):
            self.ai_worker.strategy.set_active_component(comp_name)
            self.controls.rebuild_sliders(self.ai_worker.strategy)
    def on_ai_toggled(self, enabled): self.use_ai = enabled
    def on_mask_toggled(self, enabled): self.viewer.debug_occluder = enabled
    def update_scene_params(self, key, val):
        if key == "Cam_FOV":
            self.viewer.fov = val
            self.scene_settings["Cam_FOV"] = val
        elif key == "Light_Exp": self.viewer.exposure = val * 0.1
        elif key == "Light_Gam": self.viewer.gamma = val * 0.1
        elif key == "Light_Amb": self.viewer.ambient_str = val * 0.01
        self.viewer.update()

    def save_settings(self):
        if self.current_item:
            merged = {}
            if self.ai_worker.strategy: merged.update(self.ai_worker.strategy.settings)
            merged["Scene"] = dict(self.scene_settings)
            self.db.update_item_settings(self.current_item.id, merged)

    def toggle_cinema_mode(self):
        """Toggles Fullscreen and visibility of UI elements."""
        if self.isFullScreen():
            self.showNormal()
            self.controls.show()
            self.btn_back.show()
            self.btn_fullscreen.setText("⛶")
        else:
            self.showFullScreen()
            self.controls.hide()
            self.btn_back.hide()
            self.btn_fullscreen.setText("✖")
        self.resizeEvent(None)
