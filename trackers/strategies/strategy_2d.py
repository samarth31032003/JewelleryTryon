# trackers/strategies/strategy_2d.py
from trackers.strategies.base import TrackingStrategy
from utils.logger import logger
from utils.paths import get_tracking_config

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
        """Bypass all 3D matrix math, but apply 2D culling for ears and nose pins."""
        
        # 1. Check if we need to do math at all
        needs_culling = "ear" in self.manager_2d.overlays or "nosepin" in self.manager_2d.overlays
        
        if needs_culling and results and hasattr(results, 'face_landmarks') and results.face_landmarks:
            lms = results.face_landmarks.landmark
            
            # 2. Grab normalized 3D positions of Left Temple (234) and Right Temple (454)
            p_left = lms[234]
            p_right = lms[454]
            
            # 3. Calculate Yaw (Left/Right turn) using the Z-depth difference
            dx = p_right.x - p_left.x
            dz = p_right.z - p_left.z
            
            import math
            yaw_deg = math.degrees(math.atan2(dz, dx))
            
            config = get_tracking_config()
            TURN_LIMIT = config["EAR_2D_TURN_LIMIT"]
            NOSE_TURN_LIMIT = config["NOSE_2D_TURN_LIMIT"]

            # ==========================================
            # 4A. Apply Visibility Logic for Earrings
            # ==========================================
            if "ear" in self.manager_2d.overlays:
                overlay = self.manager_2d.overlays["ear"]
                
                # Ensure the flags exist on the object
                if not hasattr(overlay, 'hide_left'):
                    overlay.hide_left = False
                    overlay.hide_right = False

                if yaw_deg > TURN_LIMIT:
                    # Turning Right (Mirror) -> Hide Right Ear
                    overlay.hide_left = False
                    overlay.hide_right = True
                elif yaw_deg < -TURN_LIMIT:
                    # Turning Left (Mirror) -> Hide Left Ear
                    overlay.hide_left = True
                    overlay.hide_right = False
                else:
                    overlay.hide_left = False
                    overlay.hide_right = False

            # ==========================================
            # 4B. Apply Visibility Logic for Nose Pin (DYNAMIC)
            # ==========================================
            if "nosepin" in self.manager_2d.overlays:
                np_overlay = self.manager_2d.overlays["nosepin"]
                if not hasattr(np_overlay, 'hide'):
                    np_overlay.hide = False
                
                # 1. Get the exact center of the nose (Nose Tip)
                nose_tip_x = lms[1].x
                
                # 2. Get the anchor point the overlay is currently using
                if getattr(np_overlay, 'side', 'left') == "left":
                    anchor_x = (lms[129].x + lms[219].x) / 2
                else:
                    anchor_x = (lms[358].x + lms[439].x) / 2

                # 3. Calculate exactly where the user dragged the sticker
                # offset_x is in pixels, so we divide by width to match the 0.0-1.0 landmark scale
                normalized_offset_x = np_overlay.offset_x / width
                final_pin_x = anchor_x + normalized_offset_x

                # 4. Determine which side of the face the pin is ACTUALLY on
                # If final_pin_x > nose_tip_x, it is on the Screen-Right (User's Left Nostril)
                # If final_pin_x < nose_tip_x, it is on the Screen-Left (User's Right Nostril)
                is_screen_right = final_pin_x > nose_tip_x

                # 5. Apply the correct hiding logic based on its true physical location
                if is_screen_right and yaw_deg > NOSE_TURN_LIMIT:
                    # Pin is on Screen-Right, user turned Right -> Hide
                    np_overlay.hide = True
                elif not is_screen_right and yaw_deg < -NOSE_TURN_LIMIT:
                    # Pin is on Screen-Left, user turned Left -> Hide
                    np_overlay.hide = True
                else:
                    np_overlay.hide = False

        return []
