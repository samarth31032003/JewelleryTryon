# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from model.database import JewelryDB
# Ensure you have the folder structure ui/catalogue/ or the file ui/catalogue.py
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
        from PyQt5.QtWidgets import QPushButton
        
        # Access the layout safely
        central = tryon_widget.centralWidget() if isinstance(tryon_widget, QMainWindow) else tryon_widget
        layout = tryon_widget.layout() if tryon_widget.layout() else central.layout()
        
        if layout:
            # Create Back Button
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
            
            # Insert at top (index 0)
            layout.insertWidget(0, btn_back)

    def go_to_catalogue(self):
        """Stops camera/AI and switches back to grid."""
        # 1. Stop the threads to save resources/battery
        self.tryon_screen.stop_session()
        
        # 2. Refresh Grid
        if hasattr(self.catalogue_screen, 'refresh_all_grids'):
            self.catalogue_screen.refresh_all_grids()
        elif hasattr(self.catalogue_screen, 'refresh_grid'):
            self.catalogue_screen.refresh_grid()
        
        # 3. Switch Screen
        self.stack.setCurrentWidget(self.catalogue_screen)

    def go_to_tryon(self, item=None):
        """Starts camera/AI and loads the selected jewelry."""
        # 1. Start the Camera/AI threads
        self.tryon_screen.start_session() 
        
        # 2. Load the item (Sets the Strategy)
        if item:
            self.tryon_screen.set_active_item(item)
            
        # 3. Switch Screen
        self.stack.setCurrentWidget(self.tryon_screen)

    def closeEvent(self, event):
        """Handle app closure cleanly."""
        print("App closing: Saving state...")
        
        if hasattr(self, 'tryon_screen'):
            # Save settings
            self.tryon_screen.save_settings()
            
            # Stop Camera and AI Threads cleanly
            self.tryon_screen.stop_session()
                
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())