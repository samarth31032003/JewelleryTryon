# ui/catalogue/card.py
import os
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QLabel, QWidget, 
                             QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from ui.styles import COLOR_PRIMARY, COLOR_MUTED, COLOR_GOLD
class JewelryCard(QFrame):
    clicked = pyqtSignal(object)
    delete_clicked = pyqtSignal(int) # Emits ID

    def __init__(self, item):
        super().__init__()
        self.item = item
        self.setObjectName("JewelryCard")
        self.setFixedSize(248, 330)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        
        # 1. Image Area
        self.lbl_img = QLabel()
        self.lbl_img.setFixedHeight(210)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0d1729, stop:1 #0b1320);"
            "border-top-left-radius: 18px; border-top-right-radius: 18px;"
        )
        self.lbl_img.setMargin(6)
        
        if item.thumbnail_path and os.path.exists(item.thumbnail_path):
            pix = QPixmap(item.thumbnail_path)
            self.lbl_img.setPixmap(pix.scaled(200, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Show "SET" for collections, "RG" for Ring, etc.
            if item.category == "Collection":
                self.lbl_img.setText("SET")
            else:
                self.lbl_img.setText(item.category[:2].upper())
            self.lbl_img.setStyleSheet(
                "font-size: 42px; color: #4a6ea3; font-weight: 700;"
                "background: qradialgradient(cx:0.5, cy:0.5, radius:0.9, stop:0 #102037, stop:1 #0b1320);"
                "border-top-left-radius: 18px; border-top-right-radius: 18px;"
            )
        
        layout.addWidget(self.lbl_img)

        # 2. Info Area
        info = QWidget()
        info_lay = QVBoxLayout(info)
        info_lay.setContentsMargins(16, 12, 16, 16)
        
        name = QLabel(item.name)
        name.setStyleSheet("font-weight: 800; font-size: 16px; color: #e8f0ff;")
        cat = QLabel(item.category)
        cat.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: 12px; font-weight: 700;")
        cat.setContentsMargins(0, 4, 0, 8)
        
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

        # Details row
        details_text = (item.details or "Virtual try-on ready").strip()
        if len(details_text) > 70:
            details_text = details_text[:67] + "..."
        details_lbl = QLabel(details_text)
        details_lbl.setWordWrap(True)
        details_lbl.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 12px;")

        # Meta row
        meta_row = QHBoxLayout()
        # sku_lbl = QLabel(f"#R-{item.id:03d}")
        # sku_lbl.setStyleSheet("color: #7fa2ff; font-weight: 700;")
        badge = QLabel("Try On")
        badge.setStyleSheet(
            f"color: #0c1220; background-color: {COLOR_GOLD};"
            " padding: 4px 8px; border-radius: 10px; font-weight: 700;"
        )
        # meta_row.addWidget(sku_lbl)
        meta_row.addStretch()
        meta_row.addWidget(badge)

        info_lay.addWidget(name)
        info_lay.addLayout(h_row)
        info_lay.addWidget(details_lbl)
        info_lay.addLayout(meta_row)
        layout.addWidget(info)

    def mousePressEvent(self, event):
        self.clicked.emit(self.item)

    def on_delete(self):
        self.delete_clicked.emit(self.item.id)