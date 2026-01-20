# trackers/strategies/collection.py
from .base import TrackingStrategy
from .neck import NeckStrategy
from .ear import EarringStrategy
from .nose import NoseStrategy
from .forehead import ForeheadStrategy

class CollectionStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.sub_strategies = {} # {'neck': NeckStrategy(), ...}
        self.settings = {}       # {'neck': {Scale:100}, ...}
        self.active_component = None # 'neck' (Key for UI slider Focus)

    def load_components(self, parts_dict):
        """
        Initializes sub-strategies based on found files.
        parts_dict = {'neck': 'path/to/obj', 'ear': 'path/to/obj'}
        """
        self.sub_strategies = {}
        self.settings = {}
        
        # Factory Logic
        for key, path in parts_dict.items():
            strat = None
            if "neck" in key: strat = NeckStrategy()
            elif "ear" in key: strat = EarringStrategy()
            elif "nose" in key: strat = NoseStrategy()
            elif "forehead" in key or "tikka" in key: strat = ForeheadStrategy()
            
            if strat:
                self.sub_strategies[key] = strat
                # Initialize settings from default
                self.settings[key] = strat.settings.copy()

        # Default Focus
        if self.sub_strategies:
            self.active_component = list(self.sub_strategies.keys())[0]

    def update_settings(self, new_settings):
        """Updates settings. Handles both full-restore and single-slider updates."""
        # 1. Check if this is a Full Restore (e.g. from DB load)
        # It's a full restore if the keys match our component names (neck, ear)
        is_restore = any(k in self.sub_strategies for k in new_settings.keys())
        
        if is_restore:
            for key, val in new_settings.items():
                if key in self.sub_strategies:
                    self.settings[key] = val
                    self.sub_strategies[key].settings = val
        
        # 2. Otherwise, it's a Slider Update for the Active Component
        elif self.active_component:
            # Update Master Record
            self.settings[self.active_component].update(new_settings)
            # Update Child Strategy
            self.sub_strategies[self.active_component].update_settings(new_settings)

    def get_slider_definitions(self):
        """Delegates slider definition to the Active Component."""
        if self.active_component and self.active_component in self.sub_strategies:
            return self.sub_strategies[self.active_component].get_slider_definitions()
        return []

    def set_active_component(self, key):
        """Called by UI Dropdown to switch slider context."""
        if key in self.sub_strategies:
            self.active_component = key

    def get_component_list(self):
        """Returns list of loaded parts for UI Dropdown."""
        return list(self.sub_strategies.keys())

    def process_frame(self, results, w, h):
        self.update_camera(w, h)
        all_commands = []
        
        for key, strategy in self.sub_strategies.items():
            # Pass camera info to children
            strategy.camera_matrix = self.camera_matrix
            strategy.dist_coeffs = self.dist_coeffs
            
            # Get commands
            cmds = strategy.process_frame(results, w, h)
            
            # Tag commands with the model_key so Renderer uses correct mesh
            for cmd in cmds:
                if cmd['type'] == 'mesh':
                    cmd['model_key'] = key 
            
            all_commands.extend(cmds)
            
        return all_commands