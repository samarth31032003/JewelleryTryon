# ui/styles.py

# Color Palette
COLOR_BACKGROUND = "#0F1A2B"
COLOR_SURFACE = "#141E2F"
COLOR_SURFACE_ALT = "#1B2C44"
COLOR_PRIMARY = "#D4AF37"  # Gold
COLOR_ACCENT = "#B76E79"   # Rose Gold
COLOR_HIGHLIGHT = "#F5F5F5"
COLOR_TEXT_MUTED = "rgba(245, 245, 245, 0.7)"
COLOR_CTA = "#6A0DAD"      # Royal Purple

def get_stylesheet():
    return f"""
    QWidget {{
        background-color: {COLOR_BACKGROUND};
        color: {COLOR_HIGHLIGHT};
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 14px;
    }}
    
    /* --- HEADERS --- */
    QLabel#HeaderTitle {{
        font-size: 32px;
        font-weight: 700;
        color: {COLOR_HIGHLIGHT};
    }}
    QLabel#SectionTitle {{
        font-size: 20px;
        font-weight: 600;
        color: {COLOR_PRIMARY};
        border-bottom: 2px solid {COLOR_PRIMARY};
        padding-bottom: 5px;
    }}
    
    /* --- BUTTONS --- */
    QPushButton {{
        background-color: {COLOR_SURFACE_ALT};
        border: 1px solid rgba(212, 175, 55, 0.5); /* Gold border */
        border-radius: 12px;
        padding: 10px 20px;
        color: {COLOR_HIGHLIGHT};
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_PRIMARY};
        color: #000000;
    }}
    QPushButton#PrimaryButton {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {COLOR_CTA}, stop:1 {COLOR_PRIMARY});
        border: none;
    }}
    QPushButton#DestructiveButton {{
        border: 1px solid #ff4444;
        color: #ff8888;
        background-color: transparent;
    }}
    QPushButton#DestructiveButton:hover {{
        background-color: #ff4444;
        color: white;
    }}

    /* --- CARDS (Catalogue) --- */
    QFrame#JewelryCard {{
        background-color: {COLOR_SURFACE};
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    QFrame#JewelryCard:hover {{
        border: 1px solid {COLOR_PRIMARY};
        background-color: {COLOR_SURFACE_ALT};
    }}
    
    /* --- INPUTS --- */
    QLineEdit, QComboBox {{
        background-color: {COLOR_SURFACE_ALT};
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px;
        color: white;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {COLOR_PRIMARY};
    }}
    """