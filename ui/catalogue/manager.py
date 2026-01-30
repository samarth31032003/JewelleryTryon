# ui/catalogue/manager.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QMessageBox, QApplication,
                             QScrollArea, QFrame, QLineEdit, QWidget)
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
        layout.setContentsMargins(32, 32, 32, 32)

        # 1. Header bar
        h_header = QHBoxLayout()
        h_header.setSpacing(18)

        brand_box = QVBoxLayout()
        brand_title = QLabel("Sringar Jewellers")
        brand_title.setObjectName("HeaderTitle")
        brand_title.setStyleSheet("font-size: 22px;")
        brand_sub = QLabel("v2.0 Offline Client")
        brand_sub.setStyleSheet("color: #7fa2ff; font-weight: 600;")
        brand_box.addWidget(brand_title)
        brand_box.addWidget(brand_sub)

        # Search bar (pill)
        search_wrap = QFrame()
        search_wrap.setObjectName("SearchWrap")
        search_wrap.setStyleSheet(
            "QFrame#SearchWrap {"
            " background-color: #0f1a2b;"
            " border: 1px solid #1f2c42;"
            " border-radius: 18px;"
            " padding: 0 12px;"
            "}"
        )
        search_layout = QHBoxLayout(search_wrap)
        search_layout.setContentsMargins(10, 6, 10, 6)
        search_layout.setSpacing(8)
        lbl_icon = QLabel("🔍")
        lbl_icon.setStyleSheet("font-size: 16px;")
        self.txt_search = QLineEdit()
        self.txt_search.setProperty("search", True)
        self.txt_search.setPlaceholderText("Search by SKU, stone, weight, or style...")
        self.txt_search.setMinimumWidth(360)
        self.txt_search.textChanged.connect(self.refresh_all_grids)
        search_layout.addWidget(lbl_icon)
        search_layout.addWidget(self.txt_search)

        # Call-to-action
        btn_add = QPushButton("Load Jewelry")
        btn_add.setObjectName("PrimaryButton")
        btn_add.clicked.connect(self.open_add_dialog)

        h_header.addLayout(brand_box)
        h_header.addStretch()
        h_header.addWidget(search_wrap)
        h_header.addSpacing(12)
        h_header.addWidget(btn_add)
        layout.addLayout(h_header)
        layout.addSpacing(12)

        # 2. Filter chips row
        self.filter_row = QHBoxLayout()
        self.filter_row.setSpacing(8)
        self.filter_row.setContentsMargins(4, 0, 4, 0)
        layout.addLayout(self.filter_row)
        layout.addSpacing(18)

        # 3. Single scroll view (no redundant tabs)
        self.all_items_scroll = QScrollArea()
        self.all_items_scroll.setWidgetResizable(True)
        self.all_items_scroll.setStyleSheet("background: transparent; border: none;")
        
        self.all_items_content = QWidget()
        self.all_items_layout = QVBoxLayout(self.all_items_content)
        self.all_items_layout.setSpacing(30)
        self.all_items_layout.setAlignment(Qt.AlignTop)
        
        self.all_items_scroll.setWidget(self.all_items_content)
        layout.addWidget(self.all_items_scroll)

        # Filters
        self.selected_filters = {"__all"}
        self._build_filter_chips()

        self.refresh_all_grids()

    def _build_filter_chips(self):
        # Clear existing chips
        while self.filter_row.count():
            item = self.filter_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        def make_chip(label, key):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key in self.selected_filters)
            btn.setStyleSheet(
                "QPushButton { background-color: #0f1a2b; border: 1px solid #1f2c42;"
                " border-radius: 16px; padding: 8px 14px; color: #e8f0ff; }"
                "QPushButton:checked { background-color: #2f7ae5; color: white; border-color: #2f7ae5; }"
                "QPushButton:hover { border-color: #2f7ae5; color: #2f7ae5; }"
            )
            btn.clicked.connect(lambda _=None, k=key: self._on_filter_clicked(k))
            return btn

        chips = [("All Items", "__all")] + [(f"{c if c != 'Collection' else 'Collections'}", c) for c in SUPPORTED_CATEGORIES]
        for label, key in chips:
            self.filter_row.addWidget(make_chip(label, key))
        self.filter_row.addStretch()

    def _on_filter_clicked(self, key):
        if key == "__all":
            self.selected_filters = {"__all"}
        else:
            if "__all" in self.selected_filters:
                self.selected_filters.remove("__all")
            if key in self.selected_filters:
                self.selected_filters.remove(key)
            else:
                self.selected_filters.add(key)
            if not self.selected_filters:
                self.selected_filters = {"__all"}
        self._build_filter_chips()
        self.refresh_all_grids()

    def on_item_clicked(self, item):
        """Opens the Details Popup instead of going straight to camera."""
        dlg = ItemDetailsDialog(item, self)
        if dlg.exec_(): 
            # If user clicked "TRY ON NOW" (which calls accept())
            self.item_selected.emit(item)

    def refresh_all_grids(self):
        # 1. Fetch from DB
        all_items = self.db.get_all_items()

        # 2. Apply Search Filter
        search_term = self.txt_search.text().lower().strip()
        if search_term:
            all_items = [i for i in all_items if search_term in i.name.lower()]

        # 3. Apply category chips filter
        if "__all" not in self.selected_filters:
            all_items = [i for i in all_items if i.category in self.selected_filters]

        # 4. Refresh "All Items"
        self._rebuild_grouped_view(all_items)

        # Only single consolidated view now

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
                lbl.setStyleSheet("color: #2f7ae5; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
                self.all_items_layout.addWidget(lbl)
                
                # B. Grid for this section
                # We reuse your JewelryGrid, but restrict its height slightly so it acts like a row
                # OR we just let it expand naturally.
                section_grid = JewelryGrid()
                section_grid.refresh(grouped[cat])
                # Connect to INTERNAL handler
                section_grid.item_clicked.connect(self.on_item_clicked)
                section_grid.delete_clicked.connect(self.delete_item)
                
                # The grid needs a minimum height to show cards
                import math
                rows = math.ceil(len(grouped[cat]) / 4)
                height = rows * 340
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