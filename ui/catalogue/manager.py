# ui/catalogue/manager.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTabWidget, QMessageBox)
from PyQt5.QtCore import pyqtSignal
from ui.styles import get_stylesheet

from .grid import JewelryGrid
from .dialog import AddItemDialog, SUPPORTED_CATEGORIES

class CatalogueWidget(QWidget):
    item_selected = pyqtSignal(object) # To Main
    
    def __init__(self, db):
        super().__init__()
        self.db = db
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # 1. Header
        h_header = QHBoxLayout()
        title_box = QVBoxLayout()
        lbl_title = QLabel("JEWELRY COLLECTION")
        lbl_title.setObjectName("HeaderTitle") # Uses Global Style
        lbl_sub = QLabel("Select an item to try on")
        lbl_sub.setStyleSheet("color: #888;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        
        btn_add = QPushButton("+ Add New Item")
        btn_add.setObjectName("PrimaryButton") # Uses Global Style
        btn_add.clicked.connect(self.open_add_dialog)
        
        h_header.addLayout(title_box)
        h_header.addStretch()
        h_header.addWidget(btn_add)
        layout.addLayout(h_header)
        layout.addSpacing(20)

        # 2. Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 3. Create Grids (One for ALL, One for each Category)
        self.grids = {} # {'All': grid_obj, 'Ring': grid_obj...}
        
        # 'All' Tab
        self.add_tab("All Items")
        
        # Category Tabs
        for cat in SUPPORTED_CATEGORIES:
            # Display logic: "Collection" -> "Sets"
            display_name = "Sets" if cat == "Collection" else cat + "s"
            self.add_tab(display_name, category_filter=cat)

        self.refresh_all_grids()

    def add_tab(self, name, category_filter=None):
        grid = JewelryGrid()
        # Connect signals
        grid.item_clicked.connect(self.item_selected.emit)
        grid.delete_clicked.connect(self.delete_item)
        
        self.tabs.addTab(grid, name)
        # Store reference
        self.grids[name] = {'widget': grid, 'filter': category_filter}

    def refresh_all_grids(self):
        # 1. Fetch ALL data once
        all_items = self.db.get_all_items()
        
        # 2. Distribute to grids
        for name, data in self.grids.items():
            grid = data['widget']
            cat_filter = data['filter']
            
            if cat_filter is None:
                # 'All' tab gets everything
                grid.refresh(all_items)
            else:
                # Filter specific items
                filtered = [i for i in all_items if i.category == cat_filter]
                grid.refresh(filtered)

    def open_add_dialog(self):
        dlg = AddItemDialog(self)
        if dlg.exec_():
            data = dlg.get_data()
            self.db.add_item(
                name=data['name'],
                category=data['category'],
                model_path=data['model_path'],
                thumbnail_path=data['thumbnail_path']
            )
            self.refresh_all_grids()

    def delete_item(self, item_id):
        confirm = QMessageBox.question(
            self, "Confirm Delete", 
            "Are you sure you want to remove this item?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.db.delete_item(item_id)
            self.refresh_all_grids()