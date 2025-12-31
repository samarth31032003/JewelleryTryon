# ui/catalogue.py
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QScrollArea, QFrame, QDialog, QLineEdit,
                             QComboBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
# Make sure you have ui/styles.py, otherwise this import will fail
from ui.styles import get_stylesheet, COLOR_PRIMARY

# Fixed categories to ensure tracking compatibility
SUPPORTED_CATEGORIES = ["Bracelet", "Necklace", "Ring", "Earring"]

class AddItemDialog(QDialog):
    """Popup to add new jewelry. Forces category selection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Jewelry")
        self.resize(400, 400)
        self.setStyleSheet(get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Name
        layout.addWidget(QLabel("Item Name:"))
        self.txt_name = QLineEdit()
        layout.addWidget(self.txt_name)
        
        # Category (RESTRICTED)
        layout.addWidget(QLabel("Category (Tracking Type):"))
        self.cmb_cat = QComboBox()
        self.cmb_cat.addItems(SUPPORTED_CATEGORIES)
        layout.addWidget(self.cmb_cat)
        
        # Model Path
        layout.addWidget(QLabel("3D Model (.obj, .glb):"))
        h_model = QHBoxLayout()
        self.txt_model = QLineEdit(); self.txt_model.setReadOnly(True)
        btn_model = QPushButton("Browse"); btn_model.clicked.connect(self.browse_model)
        h_model.addWidget(self.txt_model); h_model.addWidget(btn_model)
        layout.addLayout(h_model)
        
        # Thumbnail Path
        layout.addWidget(QLabel("Thumbnail Image (Optional):"))
        h_thumb = QHBoxLayout()
        self.txt_thumb = QLineEdit(); self.txt_thumb.setReadOnly(True)
        btn_thumb = QPushButton("Browse"); btn_thumb.clicked.connect(self.browse_thumb)
        h_thumb.addWidget(self.txt_thumb); h_thumb.addWidget(btn_thumb)
        layout.addLayout(h_thumb)
        
        # Buttons
        h_btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save to Catalogue"); btn_save.setObjectName("PrimaryButton")
        btn_save.clicked.connect(self.validate_and_accept)
        h_btns.addWidget(btn_cancel); h_btns.addWidget(btn_save)
        layout.addLayout(h_btns)

    def browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select 3D Model", "", "3D Files (*.obj *.glb *.gltf)")
        if f: self.txt_model.setText(f)

    def browse_thumb(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if f: self.txt_thumb.setText(f)
        
    def validate_and_accept(self):
        if not self.txt_name.text() or not self.txt_model.text():
            QMessageBox.warning(self, "Missing Data", "Name and 3D Model are required.")
            return
        self.accept()
        
    def get_data(self):
        return {
            "name": self.txt_name.text(),
            "category": self.cmb_cat.currentText(),
            "model_path": self.txt_model.text(),
            "thumbnail_path": self.txt_thumb.text() if self.txt_thumb.text() else None
        }

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

class CatalogueWidget(QWidget):
    item_selected = pyqtSignal(object) # To Main
    
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.setStyleSheet(get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        h_header = QHBoxLayout()
        
        title_box = QVBoxLayout()
        lbl_title = QLabel("JEWELRY COLLECTION")
        lbl_title.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Select an item to try on")
        lbl_sub.setStyleSheet("color: #888;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        
        btn_add = QPushButton("+ Add New Item")
        btn_add.setObjectName("PrimaryButton")
        btn_add.clicked.connect(self.open_add_dialog)
        
        h_header.addLayout(title_box)
        h_header.addStretch()
        h_header.addWidget(btn_add)
        
        layout.addLayout(h_header)
        layout.addSpacing(20)

        # Scrollable Grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(25)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        self.refresh_grid()

    def refresh_grid(self):
        # Clear existing
        for i in reversed(range(self.grid.count())): 
            self.grid.itemAt(i).widget().setParent(None)
            
        items = self.db.get_all_items()
        
        cols = 4
        for i, item in enumerate(items):
            card = JewelryCard(item)
            card.clicked.connect(self.item_selected.emit)
            card.delete_clicked.connect(self.delete_item)
            self.grid.addWidget(card, i // cols, i % cols)

    def open_add_dialog(self):
        dlg = AddItemDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            data = dlg.get_data()
            self.db.add_item(
                name=data['name'],
                category=data['category'],
                model_path=data['model_path'],
                thumbnail_path=data['thumbnail_path']
            )
            self.refresh_grid()

    def delete_item(self, item_id):
        confirm = QMessageBox.question(
            self, "Confirm Delete", 
            "Are you sure you want to remove this item?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            if hasattr(self.db, 'delete_item'):
                self.db.delete_item(item_id)
                self.refresh_grid()
            else:
                print("Database missing delete_item method")