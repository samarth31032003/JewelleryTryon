from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from model.database import JewelryDB
from workers.camera import get_available_cameras
from utils.logger import logger

class SettingsDialog(QDialog):
    def __init__(self, db: JewelryDB, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(300, 100)
        self.db = db
        
        layout = QVBoxLayout(self)
        
        # Camera Choice
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Select Camera:"))
        
        self.combo_cam = QComboBox()
        avail_cams = get_available_cameras()
        if not avail_cams:
            self.combo_cam.addItem("No Cameras Found", -1)
        else:
            for c in avail_cams:
                self.combo_cam.addItem(f"Camera {c}", c)
                
        current_cam = int(self.db.get_setting("camera_index", 0))
        idx = self.combo_cam.findData(current_cam)
        if idx >= 0:
            self.combo_cam.setCurrentIndex(idx)
            
        cam_layout.addWidget(self.combo_cam)
        layout.addLayout(cam_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def save(self):
        sel_idx = self.combo_cam.currentData()
        if sel_idx != -1:
            self.db.save_setting("camera_index", str(sel_idx))
            logger.bind(component="ui").info(f"Saved camera_index '{sel_idx}' to settings.")
        self.accept()
