# main.py
import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QMessageBox, 
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
)
from PyQt5.QtCore import Qt

from model.database import JewelryDB
from ui.catalogue import CatalogueWidget 
from ui.login import LoginWindow
from ui.styles import get_stylesheet
from ui.tryon.window import TryOnWindow

from utils.security import LicenseGuard
from utils.paths import USER_DATA_ROOT
from utils.logger import logger
from PyQt5.QtCore import qInstallMessageHandler, QtDebugMsg, QtInfoMsg, QtWarningMsg, QtCriticalMsg, QtFatalMsg

log = logger.bind(component="ui")

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR Jewelry TryOn")
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
        log.info("App closing: Saving state...")
        if hasattr(self, 'tryon_screen'):
            self.tryon_screen.save_settings()
            self.tryon_screen.stop_session()
        event.accept()

# --- CUSTOM LICENSE POPUP ---
class LicenseErrorDialog(QDialog):
    def __init__(self, hwid, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Software Activation Required")
        self.setFixedSize(450, 180)
        
        layout = QVBoxLayout()
        
        lbl_msg = QLabel(
            "Unregistered Device Detected.\n\n"
            "Please copy the Hardware ID below and send it to Aspiron IT solution to activate your software license."
        )
        lbl_msg.setWordWrap(True)
        lbl_msg.setStyleSheet("font-size: 13px;")
        layout.addWidget(lbl_msg)
        
        self.hwid_input = QLineEdit(hwid)
        self.hwid_input.setReadOnly(True) 
        self.hwid_input.setStyleSheet("background-color: #f8f9fa; padding: 8px; font-family: monospace; font-size: 14px; border: 1px solid #ccc;")
        layout.addWidget(self.hwid_input)
        
        self.btn_copy = QPushButton("📋 Copy to Clipboard")
        self.btn_copy.setStyleSheet("padding: 10px; font-weight: bold; font-size: 13px;")
        self.btn_copy.clicked.connect(self.copy_hwid)
        layout.addWidget(self.btn_copy)
        
        self.setLayout(layout)

    def copy_hwid(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.hwid_input.text())
        self.btn_copy.setText("✅ Copied to Clipboard!")
        self.btn_copy.setStyleSheet("padding: 10px; font-weight: bold; font-size: 13px; background-color: #dcfce7; color: #166534; border: 1px solid #166534;")

if __name__ == "__main__":
    
    # GLOBAL CRASH HANDLERS
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log.critical("Unhandled python exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    def qt_message_handler(mode, context, message):
        if mode == QtDebugMsg:
            log.debug(f"[Qt] {message}")
        elif mode == QtInfoMsg:
            log.info(f"[Qt] {message}")
        elif mode == QtWarningMsg:
            log.warning(f"[Qt] {message}")
        elif mode in (QtCriticalMsg, QtFatalMsg):
            log.critical(f"[Qt] {message}")

    qInstallMessageHandler(qt_message_handler)

    app = QApplication(sys.argv)

    # This ensures unauthorized copies cannot even see the login screen
    if not LicenseGuard.validate_license():
        dialog = LicenseErrorDialog(LicenseGuard.get_hwid())
        dialog.exec_()
        sys.exit(1)

    window = MainApp()
    window.show()
    sys.exit(app.exec_())