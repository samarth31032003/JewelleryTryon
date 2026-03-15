import sys
from loguru import logger
from utils.paths import USER_DATA_ROOT 

# Define log directory
LOG_DIR = USER_DATA_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default console logger
logger.remove()

# Configure master log file
# Captures EVERYTHING at DEBUG level
logger.add(
    LOG_DIR / "app_main.log",
    level="DEBUG",
    rotation="5 MB",
    retention="7 days",
    compression="zip",
    enqueue=True, # Thread-safe
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
)

# --- Component Specific Logs ---
# Helper filter functions based on bound 'component'

def filter_component(component_name):
    def filter_func(record):
        return record["extra"].get("component") == component_name
    return filter_func

# Camera Worker
logger.add(
    LOG_DIR / "camera.log",
    filter=filter_component("camera"),
    level="DEBUG",
    rotation="1 MB",
    retention="3 days",
    enqueue=True
)

# AI Engine
logger.add(
    LOG_DIR / "ai.log",
    filter=filter_component("ai"),
    level="DEBUG",
    rotation="2 MB",
    retention="3 days",
    enqueue=True
)

# Database
logger.add(
    LOG_DIR / "database.log",
    filter=filter_component("database"),
    level="DEBUG",
    rotation="1 MB",
    retention="3 days",
    enqueue=True
)

# UI Elements
logger.add(
    LOG_DIR / "ui.log",
    filter=filter_component("ui"),
    level="DEBUG",
    rotation="2 MB",
    retention="3 days",
    enqueue=True
)

# Tracking Strategies
logger.add(
    LOG_DIR / "trackers.log",
    filter=filter_component("trackers"),
    level="DEBUG",
    rotation="2 MB",
    retention="3 days",
    enqueue=True
)

# General Utilities
logger.add(
    LOG_DIR / "utils.log",
    filter=filter_component("utils"),
    level="DEBUG",
    rotation="1 MB",
    retention="3 days",
    enqueue=True
)
