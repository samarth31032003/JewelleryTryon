# ui/catalogue/details.py
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextBrowser, QFrame, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from ui.styles import get_stylesheet

class ItemDetailsDialog(QDialog):
    """
    Popup that shows full item info.
    Has a 'TRY ON' button that confirms the action.
    """
    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.item = item
        self.setWindowTitle(item.name)
        self.resize(600, 450)
        self.setStyleSheet(get_stylesheet())
        
        # Main Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # --- LEFT: IMAGE ---
        img_container = QFrame()
        img_container.setStyleSheet("background-color: #1a1a1a; border-right: 1px solid #444;")
        img_container.setFixedWidth(250)
        v_img = QVBoxLayout(img_container)
        v_img.setAlignment(Qt.AlignCenter)
        
        lbl_img = QLabel()
        lbl_img.setAlignment(Qt.AlignCenter)
        if item.thumbnail_path and os.path.exists(item.thumbnail_path):
            pix = QPixmap(item.thumbnail_path)
            lbl_img.setPixmap(pix.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl_img.setText(item.category[:2].upper())
            lbl_img.setStyleSheet("font-size: 60px; color: #555; font-weight: bold;")
        
        v_img.addWidget(lbl_img)
        layout.addWidget(img_container)
        
        # --- RIGHT: INFO ---
        info_container = QWidget()
        v_info = QVBoxLayout(info_container)
        v_info.setContentsMargins(30, 30, 30, 30)
        v_info.setSpacing(15)
        
        # Category Badge
        lbl_cat = QLabel(item.category.upper())
        lbl_cat.setStyleSheet("color: #D4AF37; font-size: 12px; letter-spacing: 1px; font-weight: bold;")
        v_info.addWidget(lbl_cat)
        
        # Title
        lbl_name = QLabel(item.name)
        lbl_name.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        lbl_name.setWordWrap(True)
        v_info.addWidget(lbl_name)
        
        # Details Text (Scrollable if long)
        txt_desc = QTextBrowser()
        txt_desc.setHtml(f"<div style='color: #ccc; font-size: 14px; line-height: 1.4;'>{item.details or 'No additional details provided.'}</div>")
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
        btn_try.clicked.connect(self.accept) # 'accept' means user wants to try on
        
        h_btns.addWidget(btn_close)
        h_btns.addWidget(btn_try)
        
        v_info.addStretch()
        v_info.addLayout(h_btns)
        
        layout.addWidget(info_container)