# ui/catalogue/dialog.py
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QPushButton, QFileDialog, 
                             QMessageBox, QTextEdit)
from ui.styles import get_stylesheet

SUPPORTED_CATEGORIES = [
    "Bracelet", "Necklace", "Ring", "Earring", 
    "Waist Band", "Nose Pin", "Forehead Pendant", "Collection"
]

class AddItemDialog(QDialog):
    """Popup to add new jewelry with Details."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Jewelry")
        self.resize(450, 650)
        self.setStyleSheet(get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Name
        layout.addWidget(QLabel("Item Name:"))
        self.txt_name = QLineEdit()
        layout.addWidget(self.txt_name)
        
        # Category
        layout.addWidget(QLabel("Category:"))
        self.cmb_cat = QComboBox()
        self.cmb_cat.addItems(SUPPORTED_CATEGORIES)
        layout.addWidget(self.cmb_cat)
        
        # Path Selection (ALWAYS FOLDER)
        self.lbl_model = QLabel("Select Folder (Contains .obj + textures):") 
        layout.addWidget(self.lbl_model)
        
        h_model = QHBoxLayout()
        self.txt_model = QLineEdit(); self.txt_model.setReadOnly(True)
        self.btn_browse_model = QPushButton("Select Folder") 
        self.btn_browse_model.clicked.connect(self.browse_folder) # Changed handler
        h_model.addWidget(self.txt_model); h_model.addWidget(self.btn_browse_model)
        layout.addLayout(h_model)
        
        # Thumbnail
        layout.addWidget(QLabel("Thumbnail Image (Optional):"))
        h_thumb = QHBoxLayout()
        self.txt_thumb = QLineEdit(); self.txt_thumb.setReadOnly(True)
        btn_thumb = QPushButton("Browse Image"); btn_thumb.clicked.connect(self.browse_thumb)
        h_thumb.addWidget(self.txt_thumb); h_thumb.addWidget(btn_thumb)
        layout.addLayout(h_thumb)
        
        # --- NEW 2D TRACKING IMAGE ROW ---
        layout.addWidget(QLabel("2D Tracking Image (.png) [Optional]:"))
        h_2d = QHBoxLayout()
        self.txt_2d = QLineEdit()
        self.txt_2d.setReadOnly(True)
        self.txt_2d.setPlaceholderText("Leave blank if 3D only")
        btn_browse_2d = QPushButton("Browse Image")
        btn_browse_2d.clicked.connect(self.browse_2d)
        h_2d.addWidget(self.txt_2d)
        h_2d.addWidget(btn_browse_2d)
        layout.addLayout(h_2d)
        
        # Details
        layout.addWidget(QLabel("Details:"))
        self.txt_details = QTextEdit()
        self.txt_details.setPlaceholderText("Enter details like gold purity, weight, price, or collection info...")
        self.txt_details.setMaximumHeight(80)
        layout.addWidget(self.txt_details)
        
        # Buttons
        h_btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save"); btn_save.setObjectName("PrimaryButton")
        btn_save.clicked.connect(self.validate_and_accept)
        h_btns.addWidget(btn_cancel); h_btns.addWidget(btn_save)
        layout.addLayout(h_btns)

    def browse_folder(self):
        # ALWAYS ask for a directory
        f = QFileDialog.getExistingDirectory(self, "Select Folder containing OBJ and Textures")
        if f: self.txt_model.setText(f)

    def browse_thumb(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if f: self.txt_thumb.setText(f)
        
    def browse_2d(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select 2D Image", "", "PNG Images (*.png)")
        if f: 
            self.txt_2d.setText(f)

    def validate_and_accept(self):
        name = self.txt_name.text()
        path = self.txt_model.text()

        if not name or not path:
            QMessageBox.warning(self, "Error", "Name and Path required.")
            return
            
        if not os.path.isdir(path):
            QMessageBox.warning(self, "Error", "Please select a Folder.")
            return

        # RECURSIVE CHECK
        has_obj = False
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith('.obj'):
                    has_obj = True
                    break
            if has_obj: break
            
        if not has_obj:
            QMessageBox.warning(self, "Error", "No .obj files found in this folder (or subfolders)!")
            return

        self.accept()
        
    def get_data(self):
        return {
            "name": self.txt_name.text(),
            "category": self.cmb_cat.currentText(),
            "model_path": self.txt_model.text(), # the SOURCE FOLDER path
            "thumbnail_path": self.txt_thumb.text() if self.txt_thumb.text() else None,
            "image_2d_path": self.txt_2d.text() if self.txt_2d.text() else None,
            "details": self.txt_details.toPlainText()
        }