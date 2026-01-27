# ui/catalogue/manager.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTabWidget, QMessageBox, QApplication,
                             QScrollArea, QFrame, QLineEdit)
from PyQt5.QtCore import pyqtSignal, Qt
from ui.styles import get_stylesheet

from .grid import JewelryGrid
from .dialog import AddItemDialog, SUPPORTED_CATEGORIES
from .details import ItemDetailsDialog 

class CatalogueWidget(QWidget):
    item_selected = pyqtSignal(object) 
    
    def __init__(self, db):
        super().__init__()
        self.db = db
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # 1. Header
        h_header = QHBoxLayout()
        title_box = QVBoxLayout()
        lbl_title = QLabel("JEWELRY COLLECTION")
        lbl_title.setObjectName("HeaderTitle") 
        lbl_sub = QLabel("Select an item to view details")
        lbl_sub.setStyleSheet("color: #888;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        
        # --- SEARCH BAR ---
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search by name...")
        self.txt_search.setFixedWidth(300)
        # Custom styling for a rounded "Capsule" look
        self.txt_search.setStyleSheet("""
            QLineEdit {
                border-radius: 18px;
                padding: 8px 15px;
                background-color: #2d2d2d;
                border: 1px solid #444;
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus { 
                border: 1px solid #448AFF; 
            }
        """)
        # Live filtering as you type
        self.txt_search.textChanged.connect(self.refresh_all_grids)
        
        # Add Item Button
        btn_add = QPushButton("+ Add New Item")
        btn_add.setObjectName("PrimaryButton") 
        btn_add.clicked.connect(self.open_add_dialog)
        
        h_header.addLayout(title_box)
        h_header.addStretch()
        h_header.addWidget(self.txt_search) # Search bar added here
        h_header.addSpacing(15)
        h_header.addWidget(btn_add)
        layout.addLayout(h_header)
        layout.addSpacing(20)

        # 2. Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 3. Create Views
        self.grids = {} 

        # --- A. "All Items" Tab (Grouped) ---
        self.all_items_scroll = QScrollArea()
        self.all_items_scroll.setWidgetResizable(True)
        self.all_items_scroll.setStyleSheet("background: transparent; border: none;")
        
        self.all_items_content = QWidget()
        self.all_items_layout = QVBoxLayout(self.all_items_content)
        self.all_items_layout.setSpacing(30)
        self.all_items_layout.setAlignment(Qt.AlignTop)
        
        self.all_items_scroll.setWidget(self.all_items_content)
        self.tabs.addTab(self.all_items_scroll, "All Items")

        # --- B. Category Tabs ---
        for cat in SUPPORTED_CATEGORIES:
            display_name = "Sets" if cat == "Collection" else cat + "s"
            self.add_category_tab(display_name, category_filter=cat)

        self.refresh_all_grids()

    def add_category_tab(self, name, category_filter):
        """Creates a standard Grid tab for a single category."""
        grid = JewelryGrid()
        # Connect to INTERNAL handler, not directly to main
        grid.item_clicked.connect(self.on_item_clicked) 
        grid.delete_clicked.connect(self.delete_item)
        self.tabs.addTab(grid, name)
        # Store reference
        self.grids[name] = {'widget': grid, 'filter': category_filter}

    def on_item_clicked(self, item):
        """Opens the Details Popup instead of going straight to camera."""
        dlg = ItemDetailsDialog(item, self)
        if dlg.exec_(): 
            # If user clicked "TRY ON NOW" (which calls accept())
            self.item_selected.emit(item)

    def refresh_all_grids(self):
        # 1. Fetch from DB
        all_items = self.db.get_all_items()
        
        # 2. NEW: Apply Search Filter
        search_term = self.txt_search.text().lower().strip()
        if search_term:
            all_items = [i for i in all_items if search_term in i.name.lower()]
        
        # 3. Refresh "All Items"
        self._rebuild_grouped_view(all_items)
        
        # 4. Refresh Individual Tabs
        for name, data in self.grids.items():
            grid = data['widget']
            cat_filter = data['filter']
            filtered = [i for i in all_items if i.category == cat_filter]
            grid.refresh(filtered)

    def _rebuild_grouped_view(self, all_items):
        """Builds the 'Netflix-style' rows."""
        # Clear existing rows
        while self.all_items_layout.count():
            item = self.all_items_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        # Group items by category
        grouped = {}
        for item in all_items:
            if item.category not in grouped: grouped[item.category] = []
            grouped[item.category].append(item)
            
        # Create a Section for each Category that has items
        # We iterate SUPPORTED_CATEGORIES to keep a nice order
        for cat in SUPPORTED_CATEGORIES:
            if cat in grouped and len(grouped[cat]) > 0:
                # A. Section Header
                display_name = "Sets" if cat == "Collection" else cat + "s"
                lbl = QLabel(display_name)
                lbl.setStyleSheet("color: #448AFF; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
                self.all_items_layout.addWidget(lbl)
                
                # B. Grid for this section
                # We reuse your JewelryGrid, but restrict its height slightly so it acts like a row
                # OR we just let it expand naturally.
                section_grid = JewelryGrid()
                section_grid.refresh(grouped[cat])
                # Connect to INTERNAL handler
                section_grid.item_clicked.connect(self.on_item_clicked)
                section_grid.delete_clicked.connect(self.delete_item)
                
                # Remove internal scroll from the sub-grid so the main page scrolls
                section_grid.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                # The grid needs a minimum height to show cards
                # (Approx Card Height is 280, + padding)
                # We calculate height based on rows: (count / 4 cols) * row_height
                import math
                rows = math.ceil(len(grouped[cat]) / 4) 
                height = rows * 310 
                section_grid.setMinimumHeight(height)
                
                self.all_items_layout.addWidget(section_grid)
                
                # Separator Line
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setStyleSheet("color: #444;")
                self.all_items_layout.addWidget(line)

    def open_add_dialog(self):
        dlg = AddItemDialog(self)
        if dlg.exec_():
            data = dlg.get_data()
            self.db.add_item(
                name=data['name'],
                category=data['category'],
                model_path=data['model_path'],
                thumbnail_path=data['thumbnail_path'],
                details=data['details']
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