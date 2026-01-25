# ui/login.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from ui.styles import get_stylesheet

class LoginWindow(QWidget):
    """
    The Entry Screen. Password Only.
    Connects to the DB to verify the hash.
    """
    login_successful = pyqtSignal()

    def __init__(self, db):
        super().__init__()
        self.db = db # Store reference to Database
        self.setStyleSheet(get_stylesheet())
        
        # Main Layout (Centered)
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)

        # --- THE LOGIN CARD ---
        self.card = QFrame()
        self.card.setObjectName("JewelryCard")
        self.card.setFixedSize(400, 400) # Compact size since username is gone
        
        # Specific styling for the card container
        self.card.setStyleSheet("""
            QFrame#JewelryCard {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 20px;
            }
        """)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(20)

        # 1. Logo / Title
        lbl_title = QLabel("SRINGAR")
        lbl_title.setObjectName("HeaderTitle") # Uses Gold Font
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 32px; letter-spacing: 5px;")
        
        lbl_sub = QLabel("Virtual Try-On Suite")
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setStyleSheet("color: #888; font-size: 14px; margin-bottom: 20px;")

        card_layout.addWidget(lbl_title)
        card_layout.addWidget(lbl_sub)

        # 2. Password Input Only
        self.inp_pass = QLineEdit()
        self.inp_pass.setPlaceholderText("Enter Access Key")
        self.inp_pass.setEchoMode(QLineEdit.Password)
        self.inp_pass.setMinimumHeight(45)
        # Hitting Enter triggers login
        self.inp_pass.returnPressed.connect(self.validate_login) 

        card_layout.addWidget(self.inp_pass)

        # 3. Login Button
        self.btn_login = QPushButton("UNLOCK VAULT")
        self.btn_login.setObjectName("PrimaryButton") # Uses Gold Background
        self.btn_login.setMinimumHeight(50)
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.clicked.connect(self.validate_login)
        
        card_layout.addWidget(self.btn_login)
        card_layout.addStretch()

        main_layout.addWidget(self.card)

    def validate_login(self):
        pwd = self.inp_pass.text().strip()

        # SECURE CHECK using the Database
        if self.db.verify_password(pwd):
            print("Access Granted")
            self.login_successful.emit()
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Password.")
            self.inp_pass.clear()
            self.inp_pass.setFocus()