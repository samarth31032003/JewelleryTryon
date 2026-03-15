# MediaPipe Tasks API Migration — Change Document

## Overview

Migrated the AR Jewelry TryOn application from the **deprecated** `mp.solutions.holistic` (MediaPipe Holistic)
to the new **`mediapipe.tasks`** (MediaPipe Tasks API), which uses 3 separate landmarker models
instead of a single unified pipeline.

**Date:** 2026-03-15  
**Scope:** Backend tracking engine only — zero changes to UI, rendering, or strategy logic.

---

## Why This Migration Was Needed

| Aspect               | Old (Holistic)                          | New (Tasks API)                              |
|----------------------|-----------------------------------------|----------------------------------------------|
| API                  | `mp.solutions.holistic.Holistic()`      | `mediapipe.tasks.python.vision.*Landmarker`  |
| Status               | ⚠️ Deprecated, frozen at v0.10.14       | ✅ Actively maintained by Google              |
| Face Landmarks       | 468 points                              | 478 points (468 face + 10 iris)              |
| Pose Landmarks       | 33 points                               | 33 points (identical)                        |
| Hand Landmarks       | 21 × 2 hands                            | 21 × 2 hands (identical)                     |
| Running Mode         | Per-frame (IMAGE-like)                  | VIDEO mode (built-in temporal tracking)      |
| GPU Acceleration     | Limited                                 | DirectML support on Windows                  |
| Model Files          | Bundled inside mediapipe package        | Separate `.task` files (~21MB total)         |

---

## Files Changed

### 1. `trackers/pose_engine.py` — **Complete Rewrite**

**Before:** Single `Holistic()` pipeline returning a unified results object.

```python
# OLD CODE (removed)
self.mp_holistic = mp.solutions.holistic
self.holistic = self.mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    refine_face_landmarks=False
)
# ...
self.last_results = self.holistic.process(img_rgb)
```

**After:** 3 separate landmarkers + adapter classes for backward compatibility.

**New Classes Added:**

- **`LandmarkListAdapter`** — Wraps the new API's landmark list to expose `.landmark[i]` 
  interface matching the old Holistic format. All existing strategy code accesses landmarks via
  `results.face_landmarks.landmark[i].x/y/z/visibility` — this adapter preserves that contract.

- **`CombinedResults`** — Aggregates results from all 3 landmarkers into a single object with
  `.face_landmarks`, `.pose_landmarks`, `.left_hand_landmarks`, `.right_hand_landmarks` attributes,
  exactly matching the old Holistic results interface.

- **`PoseEngine` (rewritten)** — Now initializes 3 separate landmarkers:
  - `vision.FaceLandmarker` — 478 face landmarks, VIDEO mode
  - `vision.PoseLandmarker` — 33 body landmarks, VIDEO mode
  - `vision.HandLandmarker` — 21 landmarks × 2 hands, VIDEO mode

  The `process()` method runs all 3, wraps results in `CombinedResults`, and returns it.
  Uses monotonically increasing timestamps (`frame_count * 33ms`) for VIDEO mode temporal tracking.

**Key Design Decision:** Adapter pattern was chosen specifically so that **all 8 strategy files,
overlay_2d.py, strategy_2d.py, and ai.py required ZERO changes.** The interface contract is
preserved exactly.

---

### 2. `utils/paths.py` — Added 4 Constants

```python
# NEW LINES ADDED:
MEDIAPIPE_MODELS_DIR = ASSETS_DIR / "mediapipe"
FACE_LANDMARKER_PATH = MEDIAPIPE_MODELS_DIR / "face_landmarker.task"
POSE_LANDMARKER_PATH = MEDIAPIPE_MODELS_DIR / "pose_landmarker_full.task"
HAND_LANDMARKER_PATH = MEDIAPIPE_MODELS_DIR / "hand_landmarker.task"
```

These point to the `.task` model files inside `data/assets/mediapipe/`.
Used by `PoseEngine.__init__()` to locate and validate model files at startup.

---

### 3. `.github/workflows/build_release.yml` — Added Model Download Step

```yaml
# NEW STEP ADDED (after "Install Dependencies", before "Build with PyInstaller"):
- name: Download MediaPipe Task Models
  run: python utils/download_models.py
```

This ensures the CI/CD pipeline downloads the 3 model files before PyInstaller bundles them.
The existing `--add-data "data/assets;data/assets"` flag already bundles the entire assets
directory, so the `.task` files are automatically included in the installer.

---

### 4. `.gitignore` — Added `.task` Pattern

```gitignore
# ML / binary (CHANGED: added *.task)
*.onnx
*.pt
*.engine
*.bin
*.task
```

The 3 model files total ~21MB. They are downloaded on demand via `download_models.py`
rather than committed to the repository.

---

### 5. `README.md` — Updated Setup Instructions

- Added step 5: `python utils/download_models.py` (download models before first run)
- Updated PyInstaller command to match the CI/CD workflow format
- Changed `python3 main.py` to `python main.py` (Windows convention)

---

## Files Created

### 6. `utils/download_models.py` — New File

Downloads 3 `.task` model files from Google's public storage:

| File                         | URL Source                                              | Size   |
|------------------------------|---------------------------------------------------------|--------|
| `face_landmarker.task`       | `storage.googleapis.com/mediapipe-models/face_landmarker/...`   | 3.8 MB |
| `pose_landmarker_full.task`  | `storage.googleapis.com/mediapipe-models/pose_landmarker/...`   | 9.4 MB |
| `hand_landmarker.task`       | `storage.googleapis.com/mediapipe-models/hand_landmarker/...`   | 7.8 MB |

Features:
- Skips already-downloaded files (safe to re-run)
- Cleans up partial downloads on failure
- Can be called from CLI (`python utils/download_models.py`) or imported

---

## Files NOT Changed (Verified)

The adapter pattern ensured these files required **zero modifications**:

| File | Why No Changes Needed |
|------|-----------------------|
| `trackers/strategies/neck.py` | Accesses `results.pose_landmarks.landmark[11/12]` — adapter preserves this |
| `trackers/strategies/ear.py` | Accesses `results.face_landmarks.landmark[1/168/234/454/177/401]` — preserved |
| `trackers/strategies/nose.py` | Accesses `results.face_landmarks.landmark[1/49/168/234/279/454]` — preserved |
| `trackers/strategies/forehead.py` | Accesses `results.face_landmarks.landmark[1/10/127/151/168/234/356/454]` — preserved |
| `trackers/strategies/wrist.py` | Accesses `results.left/right_hand_landmarks.landmark[0/5/9/17]` + pose — preserved |
| `trackers/strategies/ring.py` | Same hand landmarks as wrist + finger indices — preserved |
| `trackers/strategies/waist.py` | Accesses `results.pose_landmarks.landmark[23/24]` — preserved |
| `trackers/strategies/collection.py` | Delegates to sub-strategies — no direct landmark access |
| `trackers/strategies/strategy_2d.py` | Uses `hasattr(results, 'face_landmarks')` + `.landmark` — preserved |
| `trackers/strategies/base.py` | Abstract base, no landmark access |
| `graphics/overlay_2d.py` | Uses `getattr(results.face_landmarks, 'landmark', None)` — preserved |
| `workers/ai.py` | Uses `bool(results.face_landmarks)` — `LandmarkListAdapter.__bool__` handles this |
| `workers/camera.py` | No MediaPipe interaction |
| `graphics/renderer.py` | No MediaPipe interaction |
| `graphics/shaders.py` | No MediaPipe interaction |
| All `ui/` files | No MediaPipe interaction |
| `model/database.py` | No MediaPipe interaction |
| `model/models.py` | No MediaPipe interaction |
| `utils/security.py` | No MediaPipe interaction |
| `utils/smoothing.py` | No MediaPipe interaction |
| `utils/mesh_loader.py` | No MediaPipe interaction |
| `utils/file_manager.py` | No MediaPipe interaction |
| `utils/logger.py` | No MediaPipe interaction |
| `installer.iss` | No changes needed — assets directory auto-bundled |

---

## How to Set Up After This Change

```bash
# 1. Install dependencies (unchanged)
pip install -r requirements.txt

# 2. Download model files (NEW — required once)
python utils/download_models.py

# 3. Run the app (unchanged)
python main.py
```

---

## Known Notes

- **Protobuf warning on import:** `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`
  This is a harmless warning from an older protobuf version interacting with TensorFlow. It does not
  affect functionality. Can be suppressed by upgrading protobuf: `pip install --upgrade protobuf`

- **Startup time:** Loading 3 separate models takes ~2-3 seconds vs ~1 second for the old single
  Holistic model. This is a one-time cost at PoseEngine initialization (first frame processed).

- **Hand detection independence:** The old Holistic used face/pose context to improve hand detection.
  The new separate models are fully independent. In rare edge cases, hand pickup may differ slightly.
