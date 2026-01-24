# ui/catalogue/card.py
import os
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QLabel, QWidget, 
                             QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from ui.styles import COLOR_PRIMARY

class JewelryCard(QFrame):
    clicked = pyqtSignal(object)
    delete_clicked = pyqtSignal(int) # Emits ID

    def __init__(self, item):
        super().__init__()
        self.item = item
        self.setObjectName("JewelryCard")
        self.setFixedSize(220, 280)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        
        # 1. Image Area
        self.lbl_img = QLabel()
        self.lbl_img.setFixedHeight(180)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setStyleSheet("background-color: rgba(0,0,0,0.2); border-top-left-radius: 16px; border-top-right-radius: 16px;")
        
        if item.thumbnail_path and os.path.exists(item.thumbnail_path):
            pix = QPixmap(item.thumbnail_path)
            self.lbl_img.setPixmap(pix.scaled(200, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Show "SET" for collections, "RG" for Ring, etc.
            if item.category == "Collection":
                self.lbl_img.setText("SET")
            else:
                self.lbl_img.setText(item.category[:2].upper())
            self.lbl_img.setStyleSheet("font-size: 40px; color: #555; background-color: rgba(0,0,0,0.2); border-radius: 16px 16px 0 0;")
        
        layout.addWidget(self.lbl_img)

        # 2. Info Area
        info = QWidget()
        info_lay = QVBoxLayout(info)
        info_lay.setContentsMargins(15, 10, 15, 15)
        
        name = QLabel(item.name)
        name.setStyleSheet("font-weight: bold; font-size: 16px; color: white;")
        cat = QLabel(item.category)
        cat.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: 12px;")
        
        # Delete Button (Small Trash Icon)
        h_row = QHBoxLayout()
        h_row.addWidget(cat)
        h_row.addStretch()
        
        btn_del = QPushButton("×")
        btn_del.setObjectName("DestructiveButton")
        btn_del.setFixedSize(24, 24)
        btn_del.setToolTip("Delete Item")
        btn_del.clicked.connect(self.on_delete)
        h_row.addWidget(btn_del)

        info_lay.addWidget(name)
        info_lay.addLayout(h_row)
        layout.addWidget(info)

    def mousePressEvent(self, event):
        self.clicked.emit(self.item)

    def on_delete(self):
        self.delete_clicked.emit(self.item.id)