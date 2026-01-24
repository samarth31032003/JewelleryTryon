# ui/catalogue/grid.py
from PyQt5.QtWidgets import (QWidget, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from .card import JewelryCard

class JewelryGrid(QWidget):
    """Displays a grid of JewelryCards. Dumb widget (just displays what it's given)."""
    item_clicked = pyqtSignal(object)
    delete_clicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Tight fit inside the scroll area
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(25)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid.setContentsMargins(20, 20, 20, 20)
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

    def refresh(self, items):
        """Rebuilds the grid with the provided list of items."""
        # Clear existing
        for i in reversed(range(self.grid.count())): 
            w = self.grid.itemAt(i).widget()
            if w: w.setParent(None)
            
        cols = 4
        for i, item in enumerate(items):
            card = JewelryCard(item)
            card.clicked.connect(self.item_clicked.emit)
            card.delete_clicked.connect(self.delete_clicked.emit)
            self.grid.addWidget(card, i // cols, i % cols)