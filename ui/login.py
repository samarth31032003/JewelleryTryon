# ui/login.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QFrame, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from ui.styles import get_stylesheet
from utils.logger import logger

log = logger.bind(component="ui")

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
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addStretch()

        # --- THE LOGIN CARD ---
        self.card = QFrame()
        self.card.setObjectName("JewelryCard")
        self.card.setFixedSize(460, 520)
        self.card.setStyleSheet(
            "QFrame#JewelryCard {"
            " background-color: rgba(15, 26, 43, 0.25);"
            " border: 1px solid #1f2c42;"
            " border-radius: 22px;"
            "}"
        )

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(36, 40, 36, 36)
        card_layout.setSpacing(20)

        # Top brand line
        brand = QLabel("Aspiron IT solution")
        brand.setAlignment(Qt.AlignCenter)
        brand.setStyleSheet("color: #d4af37; letter-spacing: 2px; font-weight: 700;")
        card_layout.addWidget(brand)

        # Avatar / emblem
        avatar = QFrame()
        avatar.setFixedSize(96, 96)
        avatar.setStyleSheet(
            "QFrame {"
            " border-radius: 48px;"
            " background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, stop:0 #1a2a42, stop:1 #0c1523);"
            " border: 2px solid #2f7ae5;"
            "}"
        )
        avatar_layout = QVBoxLayout(avatar)
        avatar_layout.setAlignment(Qt.AlignCenter)
        avatar_lbl = QLabel("🛡")
        avatar_lbl.setAlignment(Qt.AlignCenter)
        avatar_lbl.setStyleSheet("font-size: 28px;")
        avatar_layout.addWidget(avatar_lbl)
        avatar_row = QHBoxLayout()
        avatar_row.addStretch()
        avatar_row.addWidget(avatar)
        avatar_row.addStretch()
        card_layout.addLayout(avatar_row)

        # Titles
        lbl_welcome = QLabel("Welcome, Admin")
        lbl_welcome.setAlignment(Qt.AlignCenter)
        lbl_welcome.setStyleSheet("font-size: 28px; font-weight: 800; letter-spacing: 0.8px;")
        lbl_sub = QLabel("Please verify your identity to access the virtual try-on dashboard.")
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet("color: #9ab0d0; font-size: 14px;")
        card_layout.addWidget(lbl_welcome)
        card_layout.addWidget(lbl_sub)

        # Spacer to keep layout consistent without loading bar
        spacer = QWidget()
        spacer.setFixedHeight(4)
        card_layout.addWidget(spacer)

        # Password Input Only
        self.inp_pass = QLineEdit()
        self.inp_pass.setPlaceholderText("Enter your secure password")
        self.inp_pass.setEchoMode(QLineEdit.Password)
        self.inp_pass.setMinimumHeight(48)
        self.inp_pass.returnPressed.connect(self.validate_login)
        card_layout.addWidget(self.inp_pass)

        # Login Button
        self.btn_login = QPushButton("Login")
        self.btn_login.setObjectName("PrimaryButton")
        self.btn_login.setMinimumHeight(50)
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.clicked.connect(self.validate_login)
        card_layout.addWidget(self.btn_login)

        # Footer link
        footer_link = QLabel("Forgot Password?")
        footer_link.setAlignment(Qt.AlignCenter)
        footer_link.setStyleSheet("color: #7fa2ff; font-weight: 600;")
        card_layout.addWidget(footer_link)
        card_layout.addStretch()

        # Offline / version footer below card
        footer = QLabel("Offline Mode")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #5f7396; letter-spacing: 0.3px;")

        main_layout.addWidget(self.card, alignment=Qt.AlignHCenter)
        main_layout.addSpacing(12)
        main_layout.addWidget(footer, alignment=Qt.AlignHCenter)
        main_layout.addStretch()

    def validate_login(self):
        pwd = self.inp_pass.text().strip()

        # SECURE CHECK using the Database
        if self.db.verify_password(pwd):
            log.info("Access Granted")
            self.login_successful.emit()
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Password.")
            self.inp_pass.clear()
            self.inp_pass.setFocus()