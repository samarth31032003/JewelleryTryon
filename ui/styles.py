# ui/styles.py

# --- LUXE VIRTUAL THEME (deep navy + royal blue + gold) ---
COLOR_BG = "#0c1624"          # Canvas base
COLOR_BG_ALT = "#0f1b2e"      # Card surface
COLOR_FG = "#e8f0ff"          # Primary text
COLOR_MUTED = "#9ab0d0"       # Secondary text
COLOR_BORDER = "#1f2c42"      # Subtle outline

# Accent palette
COLOR_PRIMARY = "#2f7ae5"     # Luxe blue
COLOR_PRIMARY_HOVER = "#4c8ef2"
COLOR_PRIMARY_DARK = "#1f5bb8"
COLOR_GOLD = "#d4af37"
COLOR_SUCCESS = "#4ade80"
COLOR_PLACEHOLDER = "#6f86a8"

def get_stylesheet():
    return f"""
    /* --- GLOBAL DEFAULTS --- */
    QWidget {{
        background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 {COLOR_BG}, stop:1 #0a1020);
        color: {COLOR_FG};
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        letter-spacing: 0.1px;
    }}

    /* --- CARDS --- */
    QFrame#JewelryCard {{
        background-color: {COLOR_BG_ALT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 18px;
    }}
    QFrame#JewelryCard:hover {{
        border-color: {COLOR_PRIMARY};
    }}

    /* --- INPUT FIELDS --- */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: #0f1a2b;
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 10px 14px;
        color: {COLOR_FG};
        selection-background-color: {COLOR_PRIMARY};
        selection-color: black;
    }}
    QLineEdit[search="true"] {{
        padding-left: 38px;
    }}
    QLineEdit::placeholder, QTextEdit::placeholder, QPlainTextEdit::placeholder {{
        color: {COLOR_PLACEHOLDER};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {COLOR_PRIMARY};
        box-shadow: 0 0 0 2px rgba(47,122,229,0.25);
    }}
    QLineEdit:read-only {{
        background-color: #0d1422;
        color: {COLOR_MUTED};
    }}

    /* --- BUTTONS --- */
    QPushButton {{
        background-color: #0f1a2b;
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 10px 18px;
        color: {COLOR_FG};
        font-weight: 600;
    }}
    QPushButton:hover {{
        border-color: {COLOR_PRIMARY};
        color: {COLOR_PRIMARY};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_PRIMARY_DARK};
        color: white;
    }}

    /* Primary CTA */
    QPushButton#PrimaryButton {{
        background-color: {COLOR_PRIMARY};
        color: white;
        border: none;
        border-radius: 14px;
        padding: 12px 20px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {COLOR_PRIMARY_HOVER};
    }}
    QPushButton#PrimaryButton:pressed {{
        background-color: {COLOR_PRIMARY_DARK};
    }}
    QPushButton#PrimaryButton:checked {{
        background-color: {COLOR_SUCCESS};
        color: #0c1220;
        border: 1px solid {COLOR_SUCCESS};
    }}

    /* Accent Destructive */
    QPushButton#DestructiveButton {{
        background-color: rgba(212, 79, 79, 0.1);
        color: #ff7b7b;
        font-weight: bold;
        font-size: 18px;
        border: 1px solid rgba(255, 123, 123, 0.35);
        border-radius: 12px;
    }}
    QPushButton#DestructiveButton:hover {{
        background-color: rgba(255, 123, 123, 0.2);
    }}

    /* --- COMBO BOX --- */
    QComboBox {{
        background-color: #0f1a2b;
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 10px 12px;
        color: {COLOR_FG};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 28px;
        border-left: 0;
    }}
    QComboBox::down-arrow {{
        border-left: 2px solid {COLOR_PRIMARY};
        border-bottom: 2px solid {COLOR_PRIMARY};
        width: 8px;
        height: 8px;
        transform: rotate(-45deg);
        margin-top: -2px;
        margin-right: 10px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {COLOR_BG_ALT};
        color: {COLOR_FG};
        border: 1px solid {COLOR_PRIMARY};
        outline: none;
        selection-background-color: {COLOR_PRIMARY};
        selection-color: white;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 6px 10px;
    }}

    /* --- CHECKBOX --- */
    QCheckBox {{
        spacing: 10px;
        color: {COLOR_FG};
    }}
    QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        background-color: #0f1a2b;
    }}
    QCheckBox::indicator:unchecked:hover {{
        border: 1px solid {COLOR_PRIMARY};
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLOR_PRIMARY};
        border: 1px solid {COLOR_PRIMARY};
        image: none;
    }}
    QCheckBox::indicator:checked:hover {{
        background-color: {COLOR_PRIMARY_HOVER};
    }}

    /* --- SCROLL BARS --- */
    QScrollBar:vertical {{
        border: none;
        background: transparent;
        width: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLOR_BORDER};
        min-height: 30px;
        border-radius: 6px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {COLOR_PRIMARY};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}

    /* --- TAB WIDGET --- */
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        background-color: rgba(15, 26, 43, 0.75);
        padding: 4px;
    }}
    QTabBar::tab {{
        background-color: transparent;
        color: {COLOR_MUTED};
        padding: 10px 22px;
        border: 0;
        margin: 2px 4px;
    }}
    QTabBar::tab:selected {{
        color: white;
        border-bottom: 3px solid {COLOR_PRIMARY};
        font-weight: 700;
    }}
    QTabBar::tab:hover {{ color: {COLOR_FG}; }}

    /* --- LABELS --- */
    QLabel#HeaderTitle {{
        color: {COLOR_FG};
        font-weight: 800;
        font-size: 24px;
        letter-spacing: 1px;
    }}

    /* --- FILE DIALOG FIX --- */
    QAbstractItemView {{
        background-color: {COLOR_BG_ALT};
        color: {COLOR_FG};
        selection-background-color: {COLOR_PRIMARY};
        selection-color: white;
    }}
    QHeaderView::section {{
        background-color: {COLOR_BG_ALT};
        color: {COLOR_FG};
        border: 1px solid {COLOR_BORDER};
        padding: 6px;
    }}
    """