# ui/styles.py

# --- SAPPHIRE BLUE THEME ---
COLOR_BG = "#1e1e1e"       # Dark Grey Background
COLOR_FG = "#ffffff"       # White Text
COLOR_ACCENT = "#2d2d2d"   # Lighter Grey (Cards/Input)
COLOR_BORDER = "#444444"   # Subtle Borders

# Blue Palette
COLOR_PRIMARY = "#448AFF"  # Sapphire/Electric Blue
COLOR_HOVER = "#82B1FF"    # Lighter Blue (for hover states)
COLOR_SUCCESS = "#00E676"  # Bright Teal-Green (for Active/Checked states)

def get_stylesheet():
    return f"""
    /* --- GLOBAL DEFAULTS --- */
    QWidget {{
        background-color: {COLOR_BG};
        color: {COLOR_FG};
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }}
    
    /* --- INPUT FIELDS --- */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        padding: 5px;
        color: white;
        selection-background-color: {COLOR_PRIMARY};
        selection-color: black;
    }}
    QLineEdit:focus {{
        border: 1px solid {COLOR_PRIMARY};
    }}
    QLineEdit:read-only {{
        background-color: #1a1a1a;
        color: #888;
    }}

    /* --- BUTTONS --- */
    QPushButton {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 6px 15px;
        color: white;
    }}
    QPushButton:hover {{
        background-color: #3d3d3d;
        border: 1px solid {COLOR_FG};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_PRIMARY};
        color: black;
    }}

    /* Primary Blue Button */
    QPushButton#PrimaryButton {{
        background-color: {COLOR_PRIMARY};
        color: black;
        font-weight: bold;
        border: none;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {COLOR_HOVER}; 
    }}
    
    /* Active State (Green) */
    QPushButton#PrimaryButton:checked {{
        background-color: {COLOR_SUCCESS}; 
        color: black;
        border: 1px solid {COLOR_SUCCESS};
    }}
    QPushButton#PrimaryButton:checked:hover {{
        background-color: #66BB6A;
    }}

    /* Destructive Button */
    QPushButton#DestructiveButton {{
        background-color: transparent;
        color: #ff5555;
        font-weight: bold;
        font-size: 18px;
        border: none;
    }}
    QPushButton#DestructiveButton:hover {{
        background-color: rgba(255, 85, 85, 0.2);
        border-radius: 12px;
    }}

    /* --- COMBO BOX --- */
    QComboBox {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        padding: 5px;
        color: white;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 0px;
    }}
    QComboBox::down-arrow {{
        border-left: 2px solid {COLOR_PRIMARY};
        border-bottom: 2px solid {COLOR_PRIMARY};
        width: 8px; 
        height: 8px;
        transform: rotate(-45deg);
        margin-top: -3px;
        margin-right: 6px;
    }}
    
    /* --- FIXED: DROPDOWN LIST --- */
    QComboBox QAbstractItemView {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: 1px solid {COLOR_PRIMARY};
        outline: none;
        
        /* KEY FIX: Set selection color here to force Gold Hover */
        selection-background-color: {COLOR_PRIMARY};
        selection-color: black;
    }}
    
    /* Force specific item states */
    QComboBox QAbstractItemView::item {{
        min-height: 25px;
        padding: 4px;
    }}
    QComboBox QAbstractItemView::item:selected, 
    /* Fallback for standard hover */
    QComboBox QAbstractItemView::item:hover {{
        background-color: {COLOR_PRIMARY};
        color: black;
    }}

    /* --- CHECKBOX --- */
    QCheckBox {{
        spacing: 8px;
        color: white;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {COLOR_BORDER};
        border-radius: 3px;
        background-color: {COLOR_ACCENT};
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
        background-color: {COLOR_HOVER};
    }}

    /* --- SCROLL BARS --- */
    QScrollBar:vertical {{
        border: none;
        background: {COLOR_BG};
        width: 8px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLOR_BORDER};
        min-height: 20px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {COLOR_PRIMARY};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* --- TAB WIDGET --- */
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER};
        background-color: {COLOR_BG};
    }}
    QTabBar::tab {{
        background-color: {COLOR_BG};
        color: #888;
        padding: 8px 20px;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{
        color: {COLOR_PRIMARY};
        border-bottom: 2px solid {COLOR_PRIMARY};
        font-weight: bold;
    }}
    QTabBar::tab:hover {{
        color: white;
    }}

    /* --- HEADERS --- */
    QLabel#HeaderTitle {{
        color: {COLOR_PRIMARY};
        font-weight: bold;
        font-size: 24px;
    }}

    /* --- FILE DIALOG FIX --- */
    /* This targets the file list inside Open/Save dialogs */
    QAbstractItemView {{
        background-color: {COLOR_ACCENT};
        color: white;
        selection-background-color: {COLOR_PRIMARY};
        selection-color: black;
    }}
    /* The header in file details view */
    QHeaderView::section {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: 1px solid {COLOR_BORDER};
        padding: 4px;
    }}
    """