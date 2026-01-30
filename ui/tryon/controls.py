# ui/tryon/controls.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QSlider, QGroupBox, QGridLayout, QCheckBox, 
                             QScrollArea, QTabWidget, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from ui.styles import get_stylesheet

class TryOnControls(QWidget):
    """
    The Right-Side Control Panel.
    """
    setting_changed = pyqtSignal(str, float)
    component_changed = pyqtSignal(str)
    ai_toggled = pyqtSignal(bool)
    mask_toggled = pyqtSignal(bool)
    save_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(get_stylesheet())
        self.setFixedWidth(320)
        
        # UI Setup
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.setup_tabs()

    def setup_tabs(self):
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # --- TAB 1: Item Adjustments ---
        self.tab_item = QWidget()
        item_layout = QVBoxLayout(self.tab_item)
        item_layout.setContentsMargins(15, 15, 15, 15)
        item_layout.setSpacing(15)

        # 1. Collection Dropdown
        self.combo_component = QComboBox()
        self.combo_component.currentIndexChanged.connect(
            lambda: self.component_changed.emit(self.combo_component.currentText())
        )
        self.combo_component.setVisible(False)
        item_layout.addWidget(self.combo_component)

        # 2. Sliders
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.slider_container = QWidget()
        self.slider_layout = QVBoxLayout(self.slider_container)
        self.slider_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.slider_container)
        item_layout.addWidget(self.scroll)

        # 3. Buttons
        btn_box = QWidget()
        btn_layout = QVBoxLayout(btn_box)
        btn_layout.setContentsMargins(0,0,0,0)
        
        # --- AI TOGGLE BUTTON ---
        self.btn_ai = QPushButton("Enable AI")
        self.btn_ai.setCheckable(True)
        self.btn_ai.setObjectName("PrimaryButton")
        # Connect to a local slot that updates Text AND emits signal
        self.btn_ai.toggled.connect(self.on_ai_btn_toggled) 
        
        btn_layout.addWidget(self.btn_ai)
        
        self.chk_mask = QCheckBox("Show Occlusion (Debug)")
        self.chk_mask.toggled.connect(self.mask_toggled.emit)
        btn_layout.addWidget(self.chk_mask)

        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setObjectName("PrimaryButton")
        self.btn_save.clicked.connect(self.on_save_clicked)
        btn_layout.addWidget(self.btn_save)
        
        item_layout.addWidget(btn_box)
        self.tabs.addTab(self.tab_item, "Adjustments")

        # --- TAB 2: Scene ---
        self.tab_scene = QWidget()
        scene_layout = QVBoxLayout(self.tab_scene)
        scene_layout.setContentsMargins(15, 15, 15, 15)
        
        grp_cam = QGroupBox("Camera")
        cam_grid = QGridLayout(grp_cam)
        self.add_slider_row(cam_grid, 0, "Cam_FOV", "FOV", 20, 120, 45)
        scene_layout.addWidget(grp_cam)
        
        grp_light = QGroupBox("Lighting")
        l_grid = QGridLayout(grp_light)
        self.add_slider_row(l_grid, 0, "Light_Exp", "Exposure", 1, 50, 10)
        self.add_slider_row(l_grid, 1, "Light_Gam", "Gamma", 10, 30, 22)
        self.add_slider_row(l_grid, 2, "Light_Amb", "Ambient", 0, 100, 40)
        scene_layout.addWidget(grp_light)
        
        scene_layout.addStretch()
        self.tabs.addTab(self.tab_scene, "Scene")

    def on_ai_btn_toggled(self, checked):
        """Updates UI Text and emits signal."""
        if checked:
            self.btn_ai.setText("AI ACTIVE")
        else:
            self.btn_ai.setText("Enable AI")
        self.ai_toggled.emit(checked)

    def rebuild_sliders(self, strategy):
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if not strategy: return

        if hasattr(strategy, 'get_component_list'):
            comps = strategy.get_component_list()
            self.combo_component.blockSignals(True)
            self.combo_component.clear()
            self.combo_component.addItems(comps)
            if strategy.active_component:
                self.combo_component.setCurrentText(strategy.active_component)
            self.combo_component.setVisible(True)
            self.combo_component.blockSignals(False)
        else:
            self.combo_component.setVisible(False)

        definitions = strategy.get_slider_definitions()
        for key, label_text, min_v, max_v, def_v, _ in definitions:
            if hasattr(strategy, 'active_component') and strategy.active_component:
                val = strategy.settings.get(strategy.active_component, {}).get(key, def_v)
            else:
                val = strategy.settings.get(key, def_v)

            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0,0,0,0)
            
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(80)
            sld = self._build_slider(min_v, max_v, int(val), key)

            btn_minus = self._round_button("–", lambda _=False, s=sld: s.setValue(s.value() - 1))
            btn_plus = self._round_button("+", lambda _=False, s=sld: s.setValue(s.value() + 1))

            row.addWidget(lbl)
            row.addWidget(btn_minus)
            row.addWidget(sld)
            row.addWidget(btn_plus)
            self.slider_layout.addWidget(container)

    def add_slider_row(self, layout, row, key, label, min_v, max_v, def_v):
        """Helper for static scene sliders."""
        l = QLabel(label)
        s = self._build_slider(min_v, max_v, def_v, key)

        btn_minus = self._round_button("–", lambda _=False, sl=s: sl.setValue(sl.value() - 1))
        btn_plus = self._round_button("+", lambda _=False, sl=s: sl.setValue(sl.value() + 1))

        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(l)
        h.addWidget(btn_minus)
        h.addWidget(s)
        h.addWidget(btn_plus)
        layout.addLayout(h, row, 0, 1, 2)

    def _build_slider(self, min_v, max_v, value, key):
        sld = QSlider(Qt.Horizontal)
        sld.setRange(min_v, max_v)
        sld.setValue(value)
        sld.valueChanged.connect(lambda v, k=key: self.setting_changed.emit(k, v))
        sld.setStyleSheet(
            "QSlider::groove:horizontal {"
            " height: 10px; background: #1f2c42; border-radius: 5px;"
            "}"
            "QSlider::handle:horizontal {"
            " background: #2f7ae5; width: 18px; height: 18px;"
            " margin: -5px 0; border-radius: 9px; border: 2px solid #0c1624;"
            "}"
            "QSlider::sub-page:horizontal { background: #2f7ae5; border-radius: 5px; }"
        )
        return sld

    def _round_button(self, text, on_click):
        btn = QPushButton(text)
        btn.setFixedSize(32, 32)
        btn.setStyleSheet(
            "QPushButton { background-color: #0f1a2b; color: #e8f0ff;"
            " border: 1px solid #1f2c42; border-radius: 16px; font-weight: 700; }"
            "QPushButton:hover { background-color: #2f7ae5; color: white; border-color: #2f7ae5; }"
        )
        btn.clicked.connect(lambda checked=False: on_click())
        return btn

    def on_save_clicked(self):
        self.save_requested.emit()