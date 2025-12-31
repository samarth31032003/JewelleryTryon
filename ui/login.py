# ui/login.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class LoginWindow(QWidget):
    # Signal emitted when password is correct
    login_successful = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Check")
        self.resize(400, 500)
        # Dark Theme Styling
        self.setStyleSheet("""
            QWidget { background-color: #222; color: white; }
            QLineEdit { padding: 10px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; }
            QPushButton { padding: 10px; border-radius: 5px; background-color: #0078D7; font-weight: bold; }
            QPushButton:hover { background-color: #0099FF; }
        """)

        # Main Layout (Centered)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)

        # 1. Title / Logo area
        lbl_title = QLabel("GUJARAT AI\nCHRONICLE") # Or your App Name
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setFont(QFont("Arial", 18, QFont.Bold))
        lbl_title.setStyleSheet("margin-bottom: 20px; color: #FFD700;") # Gold color
        layout.addWidget(lbl_title)

        lbl_subtitle = QLabel("Restricted Access")
        lbl_subtitle.setAlignment(Qt.AlignCenter)
        lbl_subtitle.setStyleSheet("color: #888; margin-bottom: 30px;")
        layout.addWidget(lbl_subtitle)

        # 2. Password Field
        self.txt_pass = QLineEdit()
        self.txt_pass.setPlaceholderText("Enter Access Key")
        self.txt_pass.setEchoMode(QLineEdit.Password) # Hides text with dots
        self.txt_pass.returnPressed.connect(self.check_password) # Enter key triggers login
        layout.addWidget(self.txt_pass)

        # 3. Login Button
        btn_login = QPushButton("Unlock System")
        btn_login.clicked.connect(self.check_password)
        layout.addWidget(btn_login)
        
        # Spacer
        layout.addStretch()
        
        # Footer
        lbl_footer = QLabel("Authorized Personnel Only")
        lbl_footer.setAlignment(Qt.AlignCenter)
        lbl_footer.setStyleSheet("color: #444; font-size: 10px;")
        layout.addWidget(lbl_footer)

    def check_password(self):
        password = self.txt_pass.text()
        
        # --- PASSWORD LOGIC ---
        # You can change this to match a database or config file later
        CORRECT_PASSWORD = "123" 
        
        if password == CORRECT_PASSWORD:
            print("Access Granted")
            self.login_successful.emit() # Tell Main.py to switch screens
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Password.\nPlease try again.")
            self.txt_pass.clear()
            self.txt_pass.setFocus()