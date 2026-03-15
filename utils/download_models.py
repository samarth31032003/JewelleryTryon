"""
Downloads MediaPipe Task model files required by PoseEngine.

Run once during development setup or during CI/CD builds.
Usage: python utils/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

# Resolve paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "data" / "assets" / "mediapipe"

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def download_models(target_dir=None):
    """Downloads all required .task model files. Skips existing files."""
    dest_dir = Path(target_dir) if target_dir else MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS.items():
        dest = dest_dir / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            print(f"  [SKIP] {filename} already exists ({size_mb:.1f} MB)")
            continue

        print(f"  [DOWNLOAD] {filename} ...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / 1e6
            print(f"  [OK] {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")
            # Clean up partial file
            if dest.exists():
                dest.unlink()
            sys.exit(1)

    print("\nAll MediaPipe Task models are ready.")


if __name__ == "__main__":
    print("MediaPipe Task Model Downloader")
    print(f"Target directory: {MODELS_DIR}\n")
    download_models()
