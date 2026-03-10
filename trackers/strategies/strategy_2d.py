# trackers/strategies/strategy_2d.py
from trackers.strategies.base import TrackingStrategy
from utils.logger import logger

log = logger.bind(component="ai")

class Strategy2D(TrackingStrategy):
    """
    Dynamic strategy used to bridge UI sliders to the 2D CollectionManager.
    Generates sliders dynamically based on currently loaded 2D items.
    """
    def __init__(self, manager_2d, active_type="necklace"):
        super().__init__()
        self.manager_2d = manager_2d
        self.mode = "2d"
        self.settings = {}
        # Store what we are currently trying to draw sliders for
        self.active_type = active_type 
    
    @property
    def active_component(self):
        """Tells the UI which dropdown item is currently selected."""
        return getattr(self, 'active_type', 'necklace')

    def get_component_list(self):
        """Triggers the UI to draw a Dropdown menu containing the loaded 2D items!"""
        return list(self.manager_2d.overlays.keys())

    def set_active_component(self, comp_name):
        """Called when the user selects a different item from the UI Dropdown."""
        self.active_type = comp_name
        
    def get_slider_definitions(self):
        """Expose dynamic 2D adjustments to the UI based on the active type."""
        sliders = []
        
        if self.active_type == "necklace":
            sliders.extend([
                ("Neck2D_Scale", "Necklace Scale (%)", 10, 300, 100, 100.0),
                ("Neck2D_OffsetX", "Necklace Offset X", -300, 300, 0, 1.0),
                ("Neck2D_OffsetY", "Necklace Offset Y", -300, 300, 0, 1.0),
                ("Neck2D_Bright", "Necklace Brightness (%)", 10, 300, 100, 100.0),
                ("Neck2D_Curve", "Necklace Curvature", 0, 100, 30, 100.0)
            ])
            
        elif self.active_type == "ear":
            sliders.extend([
                ("Ear2D_Scale", "Earring Scale (%)", 10, 300, 100, 100.0),
                ("Ear2D_OffsetX", "Earring Offset X", -300, 300, 0, 1.0),
                ("Ear2D_OffsetY", "Earring Offset Y", -300, 300, 0, 1.0),
                ("Ear2D_Bright", "Earring Brightness (%)", 10, 300, 100, 100.0)
            ])
            
        elif self.active_type == "nosepin":
            sliders.extend([
                ("Nose2D_Scale", "NosePin Scale (%)", 10, 300, 100, 100.0),
                ("Nose2D_OffsetX", "NosePin Offset X", -300, 300, 0, 1.0),
                ("Nose2D_OffsetY", "NosePin Offset Y", -300, 300, 0, 1.0),
                ("Nose2D_Bright", "NosePin Brightness (%)", 10, 300, 100, 100.0)
            ])
            
        elif self.active_type == "forehead":
            sliders.extend([
                ("Head2D_Scale", "Forehead Scale (%)", 10, 300, 100, 100.0),
                ("Head2D_OffsetX", "Forehead Offset X", -300, 300, 0, 1.0),
                ("Head2D_OffsetY", "Forehead Offset Y", -300, 300, 0, 1.0),
                ("Head2D_Bright", "Forehead Brightness (%)", 10, 300, 100, 100.0)
            ])
            
        return sliders

    def update_settings(self, new_settings):
        """Route specific UI changes directly to their respective active overlays."""
        super().update_settings(new_settings)
        log.warning(f"🚨 [TRIPWIRE 2] Strategy2D processing: {new_settings}")
        
        # Inject into the overlays dynamically
        for key, overlay in self.manager_2d.overlays.items():
            if key == "necklace":
                if "Neck2D_Scale" in self.settings:
                    overlay.scale = self.settings["Neck2D_Scale"] / 100.0
                if "Neck2D_OffsetX" in self.settings:
                    overlay.offset_x = int(self.settings["Neck2D_OffsetX"])
                if "Neck2D_OffsetY" in self.settings:
                    overlay.offset_y = int(self.settings["Neck2D_OffsetY"])
                if "Neck2D_Bright" in self.settings:
                    overlay.brightness = self.settings["Neck2D_Bright"] / 100.0
                if "Neck2D_Curve" in self.settings:
                    overlay.curvature_strength = self.settings["Neck2D_Curve"] / 100.0
                    
            elif key == "ear":
                if "Ear2D_Scale" in self.settings:
                    overlay.scale = self.settings["Ear2D_Scale"] / 100.0
                if "Ear2D_OffsetX" in self.settings:
                    overlay.offset_x = int(self.settings["Ear2D_OffsetX"])
                if "Ear2D_OffsetY" in self.settings:
                    overlay.offset_y = int(self.settings["Ear2D_OffsetY"])
                if "Ear2D_Bright" in self.settings:
                    overlay.brightness = self.settings["Ear2D_Bright"] / 100.0
                    
            elif key == "nosepin":
                if "Nose2D_Scale" in self.settings:
                    overlay.scale = self.settings["Nose2D_Scale"] / 100.0
                if "Nose2D_OffsetX" in self.settings:
                    overlay.offset_x = int(self.settings["Nose2D_OffsetX"])
                if "Nose2D_OffsetY" in self.settings:
                    overlay.offset_y = int(self.settings["Nose2D_OffsetY"])
                if "Nose2D_Bright" in self.settings:
                    overlay.brightness = self.settings["Nose2D_Bright"] / 100.0
                    
            elif key == "forehead":
                if "Head2D_Scale" in self.settings:
                    overlay.scale = self.settings["Head2D_Scale"] / 100.0
                if "Head2D_OffsetX" in self.settings:
                    overlay.offset_x = int(self.settings["Head2D_OffsetX"])
                if "Head2D_OffsetY" in self.settings:
                    overlay.offset_y = int(self.settings["Head2D_OffsetY"])
                if "Head2D_Bright" in self.settings:
                    overlay.brightness = self.settings["Head2D_Bright"] / 100.0
            log.warning(f"🚨 [TRIPWIRE 3] Injected into {key} overlay. New Scale: {overlay.scale}")

    def process_frame(self, results, width, height):
        """Bypass all 3D matrix math"""
        return []
