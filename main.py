# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from model.database import JewelryDB
from ui.catalogue import CatalogueWidget 
from ui.login import LoginWindow
from ui.styles import get_stylesheet
from ui.tryon.window import TryOnWindow

from utils.security import LicenseGuard

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
        self.login_screen = LoginWindow(self.db) 
        self.login_screen.login_successful.connect(self.go_to_catalogue)
        self.stack.addWidget(self.login_screen)
        
        # --- 2. Catalogue Screen ---
        self.catalogue_screen = CatalogueWidget(self.db)
        self.catalogue_screen.item_selected.connect(self.go_to_tryon)
        self.stack.addWidget(self.catalogue_screen)
        
        # --- 3. Try-On Screen ---
        self.tryon_screen = TryOnWindow(self.db)
        self.tryon_screen.back_clicked.connect(self.go_to_catalogue)
        self.stack.addWidget(self.tryon_screen)

    def go_to_catalogue(self):
        """Stops camera/AI and switches back to grid."""
        self.tryon_screen.stop_session()
        if hasattr(self.catalogue_screen, 'refresh_all_grids'):
            self.catalogue_screen.refresh_all_grids()
        self.stack.setCurrentWidget(self.catalogue_screen)

    def go_to_tryon(self, item=None):
        """Starts camera/AI and loads the selected jewelry."""
        if not item:
            return

        selected_mode = "3d" # Default

        # --- Checking for 2D capability and ask user ---
        if item.image_2d_path:
            msg = QMessageBox(self)
            msg.setWindowTitle("Select Try-On Mode")
            msg.setText(f"How would you like to try on '{item.name}'?")
            
            btn_3d = msg.addButton("Try 3D Model", QMessageBox.ActionRole)
            btn_2d = msg.addButton("Try 2D Image", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            
            msg.exec_()
            
            if msg.clickedButton() == btn_cancel:
                return # Abort screen switch if cancel
            elif msg.clickedButton() == btn_2d:
                selected_mode = "2d"
        # 1. Start the Camera/AI threads
        self.tryon_screen.start_session() 
        
        # 2. Load the item (Sets the Strategy)
        self.tryon_screen.set_active_item(item, mode=selected_mode)
            
        # 3. Switch Screen
        self.stack.setCurrentWidget(self.tryon_screen)

    def closeEvent(self, event):
        """Handle app closure cleanly."""
        print("App closing: Saving state...")
        if hasattr(self, 'tryon_screen'):
            self.tryon_screen.save_settings()
            self.tryon_screen.stop_session()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # This ensures unauthorized copies cannot even see the login screen
    if not LicenseGuard.validate_license():
        err = QMessageBox()
        err.setIcon(QMessageBox.Critical)
        err.setWindowTitle("License Error")
        err.setText("Unauthorized Device.")
        err.setInformativeText(f"Your Hardware ID:\n{LicenseGuard.get_hwid()}\n\nContact support to activate.")
        err.exec_()
        sys.exit(1)

    window = MainApp()
    window.show()
    sys.exit(app.exec_())