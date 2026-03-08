# trackers/strategies/strategy_2d.py
from trackers.strategies.base import TrackingStrategy
from utils.logger import logger

log = logger.bind(component="ai")

class Strategy2D(TrackingStrategy):
    """
    Adapter strategy used to bridge UI sliders to the 2D CollectionManager.
    When in 2D mode, AIWorker holds this strategy just so TryOnControls 
    can pull the slider definitions and push updates.
    """
    def __init__(self, manager_2d):
        super().__init__()
        self.manager_2d = manager_2d
        self.mode = "2d"
        # Initial slider defaults
        self.settings = {
            "Scale_2D": 100,
            "OffsetX_2D": 0,
            "OffsetY_2D": 0,
            "Bright_2D": 100
        }

    def get_slider_definitions(self):
        """Expose 2D adjustments to the UI."""
        return [
            ("Scale_2D", "Scale (%)", 10, 300, 100, 100.0),
            ("OffsetX_2D", "Offset X", -300, 300, 0, 1.0),
            ("OffsetY_2D", "Offset Y", -300, 300, 0, 1.0),
            ("Bright_2D", "Brightness (%)", 10, 300, 100, 100.0)
        ]

    def update_settings(self, new_settings):
        """Intercept UI changes, rename them for the DB, and push to overlays."""
        log.warning(f"🚨 [TRIPWIRE 2] Strategy2D processing: {new_settings}")
        
        # 1. Translate 3D UI keys into safe 2D Database keys
        for key, value in new_settings.items():
            if key == "Scale": self.settings["Scale_2D"] = value
            elif key == "Left_Right": self.settings["OffsetX_2D"] = value
            elif key == "Up_Down": self.settings["OffsetY_2D"] = value
            elif key == "Exposure": self.settings["Bright_2D"] = value
            elif key == "Fwd_Back": pass # 2D doesn't have Z-depth, ignore it
            else: self.settings[key] = value

        # 2. Extract the values using ONLY our safe 2D keys
        scale = self.settings.get("Scale_2D", 100) / 100.0
        offset_x = int(self.settings.get("OffsetX_2D", 0))
        offset_y = int(self.settings.get("OffsetY_2D", 0))
        brightness = self.settings.get("Bright_2D", 100) / 100.0
        
        # 3. Inject into the PNGs
        for key, overlay in self.manager_2d.overlays.items():
            overlay.scale = scale
            overlay.offset_x = offset_x
            overlay.offset_y = offset_y
            overlay.brightness = brightness
            log.warning(f"🚨 [TRIPWIRE 3] Injected into {key} overlay. New Scale: {overlay.scale}")
    def process_frame(self, results, width, height):
        """Bypass all 3D matrix math"""
        return []
