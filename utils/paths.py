# utils/paths.py
import sys
import os
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
# CONFIG_PATH = USER_DATA_ROOT / "config.json" # not using currently

# Sub-folders for assets
MODELS_DIR = USER_DATA_ROOT / "models"
THUMBNAILS_DIR = USER_DATA_ROOT / "thumbnails"

MODELS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# --- 3. DEFINE INTERNAL PATHS (Read-Only) ---
# These must match your 'data/assets' structure in dev
ASSETS_DIR = APP_ROOT / "data" / "assets"

OCCLUDER_HEAD_PATH = ASSETS_DIR / "occluder_head.obj"
OCCLUDER_BODY_PATH = ASSETS_DIR / "occluder_body.obj"
ICON_PATH = ASSETS_DIR / "icon.png"

# --- 4. HELPER ---
def resolve_path(relative_path: str) -> str:
    """
    Resolves a relative DB path (e.g., 'models/ring.obj') to absolute system path.
    """
    if not relative_path: return ""
    return str(USER_DATA_ROOT / relative_path)