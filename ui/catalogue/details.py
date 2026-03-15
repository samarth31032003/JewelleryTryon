# ui/catalogue/details.py
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextBrowser, QFrame, QWidget, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from ui.styles import get_stylesheet
from ui.catalogue.model_viewer import ModernMeshWidget

class ItemDetailsDialog(QDialog):
    """
    Popup that shows full item info with optional 3D Preview.
    """
    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.item = item
        self.setWindowTitle(item.name)
        # Increased size for better 3D viewing
        self.resize(800, 500) 
        self.setStyleSheet(get_stylesheet())
        
        # Main Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # --- LEFT: MEDIA CONTAINER (Image / 3D) ---
        media_container = QFrame()
        media_container.setStyleSheet("background-color: #121212; border-right: 1px solid #444;")
        # Increased width for the visual area
        media_container.setFixedWidth(400) 
        
        v_media = QVBoxLayout(media_container)
        v_media.setContentsMargins(0,0,0,0)
        
        # 1. Stack for toggling
        self.media_stack = QStackedWidget()
        
        # Page 0: Image
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        if item.thumbnail_path and os.path.exists(item.thumbnail_path):
            pix = QPixmap(item.thumbnail_path)
            self.lbl_img.setPixmap(pix.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_img.setText(item.category[:2].upper())
            self.lbl_img.setStyleSheet("font-size: 80px; color: #333; font-weight: bold;")
        self.media_stack.addWidget(self.lbl_img)
        
        # Page 1: 3D Viewer
        self.gl_widget = ModernMeshWidget()
        self.media_stack.addWidget(self.gl_widget)
        
        v_media.addWidget(self.media_stack)
        
        # 2. Toggle Button Overlay (Bottom Left corner of media area)
        # We put it in a small layout at the bottom
        h_toggle = QHBoxLayout()
        h_toggle.setContentsMargins(10, 10, 10, 10)
        h_toggle.addStretch()
        
        # Only show 3D button if it's a valid single file (not a directory/collection)
        if not os.path.isdir(item.model_path) and item.model_path.lower().endswith(('.obj', '.glb')):
            self.btn_3d = QPushButton("View 3D 🧊")
            self.btn_3d.setCheckable(True)
            # self.btn_3d.setFixedSize(160, 35)
            # --- Use setMinimumWidth instead ---
            self.btn_3d.setMinimumWidth(140) 
            self.btn_3d.setFixedHeight(35) # Keep height fixed for consistency

            self.btn_3d.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 0.6);
                    border: 1px solid #555;
                    border-radius: 17px;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background-color: #D4AF37;
                    color: black;
                    border: 1px solid #D4AF37;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
            """)
            self.btn_3d.toggled.connect(self.toggle_3d_view)
            h_toggle.addWidget(self.btn_3d)
        
        h_toggle.addStretch()
        v_media.addLayout(h_toggle)
        
        layout.addWidget(media_container)
        
        # --- RIGHT: INFO ---
        info_container = QWidget()
        v_info = QVBoxLayout(info_container)
        v_info.setContentsMargins(30, 30, 30, 30)
        v_info.setSpacing(15)
        
        # Category Badge
        lbl_cat = QLabel(item.category.upper())
        lbl_cat.setStyleSheet("color: #448AFF; font-size: 12px; letter-spacing: 1px; font-weight: bold;")
        v_info.addWidget(lbl_cat)
        
        # Title
        lbl_name = QLabel(item.name)
        lbl_name.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        lbl_name.setWordWrap(True)
        v_info.addWidget(lbl_name)
        
        # Details Text
        txt_desc = QTextBrowser()
        detail_text = item.details if item.details else "No additional details provided."
        txt_desc.setHtml(f"<div style='color: #ccc; font-size: 14px; line-height: 1.4;'>{detail_text}</div>")
        txt_desc.setFrameShape(QFrame.NoFrame)
        txt_desc.setStyleSheet("background: transparent;")
        v_info.addWidget(txt_desc)
        
        # Action Buttons
        h_btns = QHBoxLayout()
        h_btns.setSpacing(15)
        
        btn_close = QPushButton("Close")
        btn_close.setMinimumHeight(45)
        btn_close.clicked.connect(self.reject)
        
        btn_try = QPushButton("TRY ON NOW")
        btn_try.setObjectName("PrimaryButton")
        btn_try.setMinimumHeight(45)
        btn_try.setCursor(Qt.PointingHandCursor)
        btn_try.clicked.connect(self.accept)
        
        h_btns.addWidget(btn_close)
        h_btns.addWidget(btn_try)
        
        v_info.addStretch()
        v_info.addLayout(h_btns)
        
        layout.addWidget(info_container)

    def toggle_3d_view(self, checked):
        if checked:
            self.media_stack.setCurrentIndex(1)
            self.btn_3d.setText("View Image")
            # Lazy Load the mesh only when requested
            if not self.gl_widget.mesh_ready:
                self.gl_widget.load_mesh(self.item.model_path)
        else:
            self.media_stack.setCurrentIndex(0)
            self.btn_3d.setText("View 3D 🧊")