# ui/styles.py

# --- 1. THEME VARIABLES (The DNA) ---
COLORS = {
    # Brand Colors
    "primary": "#D4AF37",       # Gold (Luxury)
    "primary_hover": "#B5952F", # Darker Gold
    "text_on_primary": "#000000",
    
    # Backgrounds
    "bg_dark": "#1e1e1e",       # Main Window Background
    "bg_panel": "#2d2d2d",      # Side Panels / Cards
    "bg_input": "#121212",      # Text Inputs
    
    # Text
    "text_main": "#ffffff",
    "text_dim": "#aaaaaa",
    
    # Functional
    "danger": "#e74c3c",        # Red (Delete/Cancel)
    "danger_hover": "#c0392b",
    "success": "#2ecc71",       # Green
    "border": "#444444"
}

FONTS = {
    "header": "22px 'Segoe UI', sans-serif",
    "subheader": "16px 'Segoe UI', sans-serif",
    "body": "14px 'Segoe UI', sans-serif",
    "small": "12px 'Segoe UI', sans-serif"
}

# --- 2. THE STYLESHEET GENERATOR ---
def get_stylesheet():
    return f"""
    /* --- GLOBAL DEFAULTS --- */
    QMainWindow, QDialog {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_main']};
    }}
    
    QWidget {{
        font: {FONTS['body']};
        color: {COLORS['text_main']};
    }}

    /* --- INPUTS (Text Fields, Dropdowns) --- */
    QLineEdit, QComboBox {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 8px;
        color: {COLORS['text_main']};
        font-size: 14px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {COLORS['primary']};
    }}

    /* --- SCROLL BARS (Modern Look) --- */
    QScrollBar:vertical {{
        background: {COLORS['bg_dark']};
        width: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['border']};
        min-height: 20px;
        border-radius: 5px;
    }}
    
    /* --- TABS --- */
    QTabWidget::pane {{ border: 0; }}
    QTabBar::tab {{
        background: transparent;
        color: {COLORS['text_dim']};
        padding: 10px 20px;
        font-size: 14px;
    }}
    QTabBar::tab:selected {{
        color: {COLORS['text_main']};
        border-bottom: 2px solid {COLORS['primary']};
        font-weight: bold;
    }}

    /* ==============================================
       COMPONENT CLASSES (Use setObjectName to apply)
       ============================================== */

    /* [PrimaryButton] - The Gold Action Button */
    QPushButton#PrimaryButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['text_on_primary']};
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 14px;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {COLORS['primary_hover']};
    }}

    /* [DestructiveButton] - The Red Danger Button */
    QPushButton#DestructiveButton {{
        background-color: transparent;
        color: {COLORS['danger']};
        border: 1px solid {COLORS['danger']};
        border-radius: 4px;
        padding: 5px 10px;
    }}
    QPushButton#DestructiveButton:hover {{
        background-color: {COLORS['danger']};
        color: white;
    }}

    /* [JewelryCard] - The Item Box in Catalogue */
    QFrame#JewelryCard {{
        background-color: {COLORS['bg_panel']};
        border-radius: 16px;
        border: 1px solid transparent;
    }}
    QFrame#JewelryCard:hover {{
        border: 1px solid {COLORS['primary']};
    }}

    /* [HeaderTitle] - Large Page Titles */
    QLabel#HeaderTitle {{
        font: {FONTS['header']};
        font-weight: bold;
        color: {COLORS['primary']};
    }}
    """

# Helper to expose the primary color for Python-side logic (like plotting)
COLOR_PRIMARY = COLORS['primary']