# utils/paths.py
import sys
import os
import json
from pathlib import Path

# --- 1. DETERMINE ROOTS ---

# A. APP_ROOT: Internal Read-Only Assets (Baked into EXE)
if getattr(sys, 'frozen', False):
    # Running as compiled EXE
    APP_ROOT = Path(sys._MEIPASS)
else:
    # Running as script (Dev mode)
    APP_ROOT = Path(__file__).resolve().parent.parent

# B. USER_DATA_ROOT: External Read-Write Data (Next to EXE)
if getattr(sys, 'frozen', False):
    # Folder next to the EXE file
    USER_DATA_ROOT = Path(sys.executable).parent / "library"
else:
    # 'data' folder in project directory for Dev
    USER_DATA_ROOT = APP_ROOT / "data"

# --- 2. DEFINE EXTERNAL PATHS (Read-Write) ---
# We use a 'library' folder to keep things clean next to the EXE
USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)

DB_PATH = USER_DATA_ROOT / "jewelry.db"
CONFIG_PATH = USER_DATA_ROOT / "tracking_settings.json"

# Sub-folders for assets
MODELS_DIR = USER_DATA_ROOT / "models"
THUMBNAILS_DIR = USER_DATA_ROOT / "thumbnails"

MODELS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# --- 3. DEFINE INTERNAL PATHS (Read-Only) ---
# These must match your 'data/assets' structure in dev
ASSETS_DIR = APP_ROOT / "data" / "assets"

OCCLUDER_HEAD_PATH = ASSETS_DIR / "head" / "occluder_head.obj"
OCCLUDER_BODY_PATH = ASSETS_DIR / "body" / "occluder_body.obj"
ICON_PATH = ASSETS_DIR / "icon.png"

# --- 4. CONFIGURATION MANAGER ---
def get_tracking_config():
    """
    Safely finds or creates a JSON config file in the USER_DATA_ROOT
    so the client can edit tracking thresholds without recompiling.
    """
    default_config = {
        "EAR_3D_TURN_LIMIT": 15.0,
        "EAR_2D_TURN_LIMIT": 15.0,
        "NOSE_2D_TURN_LIMIT": 10.0,
        "SHOULDER_WIDTH_M": 0.40  # temp for future improvement.
    }
    
    if not CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        except Exception as e:
            print(f"[Config] Could not create config file: {e}")
            return default_config
            
    try:
        with open(CONFIG_PATH, 'r') as f:
            user_config = json.load(f)
            
            # Safety check: Merge defaults if the user accidentally deleted a line
            needs_save = False
            for key, val in default_config.items():
                if key not in user_config:
                    user_config[key] = val
                    needs_save = True
            
            # If we had to fix the file, save it back
            if needs_save:
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(user_config, f, indent=4)
                    
            return user_config
    except Exception as e:
        print(f"[Config] Error reading config file (broken JSON?): {e}")
        return default_config

# --- 5. HELPER ---
def resolve_path(relative_path: str) -> str:
    """
    Resolves a relative DB path (e.g., 'models/ring.obj') to absolute system path.
    """
    if not relative_path: return ""
    return str(USER_DATA_ROOT / relative_path)