# trackers/strategies/collection.py
from .base import TrackingStrategy
from .neck import NeckStrategy
from .ear import EarringStrategy
from .nose import NoseStrategy
from .forehead import ForeheadStrategy
from trackers.occluder_shared import OccluderManager

class CollectionStrategy(TrackingStrategy):
    def __init__(self):
        super().__init__()
        self.sub_strategies = {} 
        self.active_component = None 

        # Define which keys are shared across all "Head" strategies
        self.SHARED_HEAD_KEYS = {"Occ_Head_Scale", "Occ_Head_X", "Occ_Head_Y", "Occ_Head_Z"}
        # Define which strategies share the head
        self.HEAD_STRAT_NAMES = ["ear", "nose", "forehead"]
        
        self._mode = "3d" # Internal mode tracker

    # --- MODE BROADCASTER ---
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        self._mode = val
        # When mode changes, broadcast it to all active child strategies!
        for strat in self.sub_strategies.values():
            strat.mode = val

    def load_components(self, parts_dict):
        """
        Initializes sub-strategies based on found files.
        parts_dict = {'neck': 'path/to/obj', 'ear': 'path/to/obj'}
        """
        self.sub_strategies = {}
        
        # 1. Initialize Sub-Strategies based on files found
        for key, path in parts_dict.items():
            strat = None
            # Determine type based on folder/filename
            if "neck" in key: strat = NeckStrategy()
            elif "ear" in key: strat = EarringStrategy()
            elif "nose" in key: strat = NoseStrategy()
            elif "forehead" in key or "tikka" in key: strat = ForeheadStrategy()
            
            if strat:
                strat.mode = self.mode # child inherit the mode.
                self.sub_strategies[key] = strat

        if self.sub_strategies:
            self.active_component = list(self.sub_strategies.keys())[0]


    def update_settings(self, new_settings):
        """
        Smart Update: If a shared Head setting is changed, broadcast it to ALL head strategies.
        """
        # Case A: Full Restore (Loading from Database)
        is_restore = any(k in self.sub_strategies for k in new_settings.keys())
        if is_restore:
            for key, val in new_settings.items():
                if key in self.sub_strategies:
                    self.sub_strategies[key].settings = val
        
        # Case B: Slider Update (Live UI)
        elif self.active_component:
            # 1. Update the Active Component normally
            self.sub_strategies[self.active_component].settings.update(new_settings)

            # 2. CHECK FOR SHARED HEAD KEYS
            # If the user moved a slider like "Occ_Head_Scale"...
            if any(k in self.SHARED_HEAD_KEYS for k in new_settings.keys()):
                
                # Extract only the shared settings
                shared_update = {k: v for k, v in new_settings.items() if k in self.SHARED_HEAD_KEYS}
                
                # Broadcast to other head strategies (Ear, Nose, Forehead)
                for name in self.HEAD_STRAT_NAMES:
                    # ensure strategy exists
                    if name != self.active_component and name in self.sub_strategies:
                        self.sub_strategies[name].settings.update(shared_update)
                        # print(f"[Collection] Synced {shared_update} to {name}")

    def get_slider_definitions(self):
        """
        Only show sliders for the CURRENTLY SELECTED component.
        This fixes the 'Dropdown doesn't work' issue.
        """
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
        
        raw_commands = []
        
        # 1. Run All Active Sub-Strategies
        for key, strategy in self.sub_strategies.items():
            strategy.camera_matrix = self.camera_matrix
            strategy.dist_coeffs = self.dist_coeffs
            
            cmds = strategy.process_frame(results, w, h)
            
            for cmd in cmds:
                if cmd['type'] == 'mesh':
                    cmd['model_key'] = key 
            
            raw_commands.extend(cmds)

        # 2. DEDUPLICATE OCCLUDERS
        final_cmds = []
        seen_occluders = set()
        
        for cmd in raw_commands:
            if cmd['type'] == 'occluder':
                occ_key = cmd.get('mesh_key') 
                
                if occ_key and occ_key not in seen_occluders:
                    seen_occluders.add(occ_key)
                    final_cmds.append(cmd)
            else:
                final_cmds.append(cmd)
            
        return final_cmds