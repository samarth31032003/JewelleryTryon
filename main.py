# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from model.database import JewelryDB
from ui.catalogue import CatalogueWidget
from ui.login import LoginWindow
from ui.styles import get_stylesheet
from ui.tryon_view import TryOnWindow 

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sringar Jewellers - Virtual Try-On")
        self.resize(1280, 800)
        self.setStyleSheet(get_stylesheet())
        
        # Database
        self.db = JewelryDB()
        
        # Central Stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # --- 1. Login Screen ---
        self.login_screen = LoginWindow()
        # Hack: LoginWindow in your previous code was a window. 
        # Ideally, refactor LoginWindow to inherit QWidget and emit signal.
        # For now, we connect its signal to the stack switcher.
        self.login_screen.login_successful.connect(self.go_to_catalogue)
        self.stack.addWidget(self.login_screen)
        
        # --- 2. Catalogue Screen ---
        self.catalogue_screen = CatalogueWidget(self.db)
        self.catalogue_screen.item_selected.connect(self.go_to_tryon)
        self.stack.addWidget(self.catalogue_screen)
        
        # --- 3. Try-On Screen ---
        self.tryon_screen = TryOnWindow(self.db) 
        self.inject_back_button(self.tryon_screen)
        self.stack.addWidget(self.tryon_screen)

    def inject_back_button(self, tryon_widget):
        """Adds a Back button to the existing layout of the TryOn screen."""
        # This assumes TryOnWindow has a layout. 
        # If it's a QMainWindow, we access the central widget's layout.
        
        central = tryon_widget.centralWidget() if isinstance(tryon_widget, QMainWindow) else tryon_widget
        layout = central.layout()
        
        # Create Back Button
        from PyQt5.QtWidgets import QPushButton
        btn_back = QPushButton("← Back to Collection")
        btn_back.setStyleSheet("""
            background-color: rgba(0,0,0,0.5); 
            color: white; 
            border: 1px solid #D4AF37;
            padding: 8px 15px;
            font-weight: bold;
        """)
        btn_back.setFixedSize(180, 40)
        btn_back.clicked.connect(self.go_to_catalogue)
        
        # Add to the layout
        layout = tryon_widget.layout()
        if layout:
            layout.insertWidget(0, btn_back)

    def go_to_catalogue(self):
        # 1. SAVE: Explicitly save settings before leaving TryOn
        print("Navigating back: Saving settings...")
        self.tryon_screen.save_current_settings_to_db()
        
        # 2. STOP: Stop camera resources
        if hasattr(self.tryon_screen, 'timer'):
            self.tryon_screen.timer.stop()
        if hasattr(self.tryon_screen, 'cap') and self.tryon_screen.cap.isOpened():
             self.tryon_screen.cap.release()
        
        # 3. REFRESH: Reload Catalogue so it has the latest data from DB
        self.catalogue_screen.refresh_grid()
             
        self.stack.setCurrentWidget(self.catalogue_screen)

    def go_to_tryon(self, item):
        print(f"Loading {item.name} ({item.category})...")
        
        # 1. Determine Tracking Mode based on Category
        # If Necklace -> Standard Mode (or Neck Mode if you implemented it)
        # If Bracelet -> AI Hand Mode
        
        use_ai_hand = (item.category == "Bracelet")
        self.tryon_screen.use_ai = use_ai_hand
        
        # Pass the ITEM OBJECT, not just the path
        self.tryon_screen.set_active_item(item)
        
        # 3. Start Camera
        self.tryon_screen.cap.open(0)
        self.tryon_screen.timer.start(30)
        
        self.stack.setCurrentWidget(self.tryon_screen)

    def closeEvent(self, event):
        """Handle app closure to ensure data is saved."""
        print("App closing: Saving state...")
        # Force save on the tryon screen if it's active or initialized
        if hasattr(self, 'tryon_screen'):
            self.tryon_screen.save_current_settings_to_db()
            # Release camera
            if self.tryon_screen.cap.isOpened():
                self.tryon_screen.cap.release()
                
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())