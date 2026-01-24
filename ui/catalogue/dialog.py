# ui/catalogue/dialog.py
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QPushButton, QFileDialog, 
                             QMessageBox)
from ui.styles import get_stylesheet

# Centralized List
SUPPORTED_CATEGORIES = [
    "Bracelet", "Necklace", "Ring", "Earring", 
    "Waist Band", "Nose Pin", "Forehead Pendant", "Collection"
]

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
        self.txt_name.setPlaceholderText("e.g. Gold Bridal Set")
        layout.addWidget(self.txt_name)
        
        # Category
        layout.addWidget(QLabel("Category (Tracking Type):"))
        self.cmb_cat = QComboBox()
        self.cmb_cat.addItems(SUPPORTED_CATEGORIES)
        self.cmb_cat.currentTextChanged.connect(self.on_category_change) 
        layout.addWidget(self.cmb_cat)
        
        # Model Path
        self.lbl_model = QLabel("3D Model (.obj, .glb):") 
        layout.addWidget(self.lbl_model)
        
        h_model = QHBoxLayout()
        self.txt_model = QLineEdit()
        self.txt_model.setReadOnly(True)
        self.btn_browse_model = QPushButton("Browse") 
        self.btn_browse_model.clicked.connect(self.browse_model)
        h_model.addWidget(self.txt_model)
        h_model.addWidget(self.btn_browse_model)
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

        # Trigger initial state
        self.on_category_change(self.cmb_cat.currentText())

    def on_category_change(self, text):
        if text == "Collection":
            self.lbl_model.setText("Collection Folder (Must contain .obj files):")
            self.btn_browse_model.setText("Select Folder")
            self.txt_model.setPlaceholderText("Select directory containing set items...")
        else:
            self.lbl_model.setText("3D Model (.obj, .glb):")
            self.btn_browse_model.setText("Browse File")
            self.txt_model.setPlaceholderText("Select a single 3D model file...")

    def browse_model(self):
        cat = self.cmb_cat.currentText()
        if cat == "Collection":
            f = QFileDialog.getExistingDirectory(self, "Select Collection Folder")
        else:
            f, _ = QFileDialog.getOpenFileName(self, "Select 3D Model", "", "3D Files (*.obj *.glb *.gltf)")
        if f: self.txt_model.setText(f)

    def browse_thumb(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if f: self.txt_thumb.setText(f)
        
    def validate_and_accept(self):
        name = self.txt_name.text()
        path = self.txt_model.text()
        cat = self.cmb_cat.currentText()

        if not name or not path:
            QMessageBox.warning(self, "Missing Data", "Name and Path are required.")
            return

        if cat == "Collection":
            if not os.path.isdir(path):
                QMessageBox.warning(self, "Error", "For Collections, you must select a Directory.")
                return
            has_obj = any(f.lower().endswith('.obj') for f in os.listdir(path))
            if not has_obj:
                QMessageBox.warning(self, "Error", "Selected folder contains no .obj files!")
                return
        else:
            if not os.path.isfile(path):
                QMessageBox.warning(self, "Error", "File not found.")
                return

        self.accept()
        
    def get_data(self):
        return {
            "name": self.txt_name.text(),
            "category": self.cmb_cat.currentText(),
            "model_path": self.txt_model.text(),
            "thumbnail_path": self.txt_thumb.text() if self.txt_thumb.text() else None
        }