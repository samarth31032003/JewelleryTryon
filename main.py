# main.py
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from model.database import JewelryDB
from ui.catalogue import CatalogueWidget 
from ui.login import LoginWindow
from ui.styles import get_stylesheet
from ui.tryon.window import TryOnWindow

from utils.security import LicenseGuard
from utils.paths import USER_DATA_ROOT

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
        if not item: return

        has_3d = False
        has_2d = False

        if item.category == "Collection" and item.model_path:
            full_path = USER_DATA_ROOT / item.model_path
            if full_path.exists() and full_path.is_dir():
                for root, dirs, files in os.walk(full_path):
                    for f in files:
                        if f.lower().endswith('.obj'): has_3d = True
                        if f.lower().endswith('.png'): has_2d = True
        else:
            if item.model_path: has_3d = True
            if item.image_2d_path: has_2d = True

        # --- DETERMINE MODE ---
        selected_mode = "3d"
        
        if has_3d and has_2d:
            # Item has BOTH -> Ask the user
            msg = QMessageBox(self)
            msg.setWindowTitle("Select Try-On Mode")
            msg.setText(f"How would you like to try on '{item.name}'?")
            btn_3d = msg.addButton("Try 3D Model", QMessageBox.ActionRole)
            btn_2d = msg.addButton("Try 2D Image", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.exec_()
            
            if msg.clickedButton() == btn_cancel: return
            elif msg.clickedButton() == btn_2d: selected_mode = "2d"
            else: selected_mode = "3d"
            
        elif has_2d and not has_3d:
            selected_mode = "2d" # Force 2D
        elif has_3d and not has_2d:
            selected_mode = "3d" # Force 3D
        else:
            QMessageBox.warning(self, "Error", "This item has no 2D or 3D files associated with it.")
            return

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