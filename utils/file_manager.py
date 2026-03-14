# utils/file_manager.py
import shutil
import os
import uuid
from pathlib import Path
from utils.paths import MODELS_DIR, THUMBNAILS_DIR, USER_DATA_ROOT
from utils.logger import logger

log = logger.bind(component="utils")

class LibraryManager:
    """
    Handles secure copying of user assets into the App's Library.
    """

    @staticmethod
    def import_asset_folder(source_path):
        """
        Copies an entire folder to the library.
        Returns TUPLE: (relative_folder_path, relative_obj_path)
        """
        source = Path(source_path)
        if not source.exists() or not source.is_dir():
            log.error(f"Error: Source is not a directory: {source}")
            return None, None

        # 1. Generate Unique Folder Name
        # e.g. "WeddingSet" -> "WeddingSet_a1b2c3"
        safe_name = "".join([c for c in source.name if c.isalnum() or c in (' ', '_', '-')]).strip()
        folder_name = f"{safe_name}_{uuid.uuid4().hex[:8]}"
        dest_dir = MODELS_DIR / folder_name
        
        try:
            # 2. Copy Everything (OBJ, MTL, JPG, Subfolders)
            shutil.copytree(source, dest_dir)
            
            # 3. Find the 'Primary' OBJ for Single Items
            # (For collections, this might just be the first one found, which is fine)
            found_obj = None
            # Recursive search for ANY .obj file
            for root, dirs, files in os.walk(dest_dir):
                for f in files:
                    if f.lower().endswith(".obj"):
                        found_obj = Path(root) / f
                        break
                if found_obj: break

            rel_folder = f"models/{folder_name}"
            rel_obj = None
            
            if found_obj:
                # Calculate path relative to USER_DATA_ROOT
                # e.g. "models/WeddingSet_a1b2c3/neck/necklace.obj"
                try:
                    rel_obj = str(found_obj.relative_to(USER_DATA_ROOT)).replace("\\", "/")
                except ValueError:
                    # Fallback if path manipulation fails
                    rel_obj = f"{rel_folder}/{found_obj.name}"
            
            return rel_folder, rel_obj

        except Exception as e:
            log.error(f"Error importing asset folder: {e}")
            return None, None

    @staticmethod
    def import_thumbnail(source_path):
        """Simple copy for single image files."""
        source = Path(source_path)
        if not source.exists(): return None
        
        clean_name = f"{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}"
        dest = THUMBNAILS_DIR / clean_name
        try:
            shutil.copy2(source, dest)
            return f"thumbnails/{clean_name}"
        except Exception:
            return None

    @staticmethod
    def import_2d_asset(source_path):
        """
        Copies a 2D PNG/JPG image used for the 2D tracking mode.
        """
        source = Path(source_path)
        if not source.exists(): return None
        
        # We can store these in the models directory or a new 'images' directory.
        # Let's reuse MODELS_DIR so everything is kept together cleanly.
        clean_name = f"2d_{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}"
        dest = MODELS_DIR / clean_name
        
        try:
            shutil.copy2(source, dest)
            return f"models/{clean_name}"
        except Exception as e:
            log.error(f"Error importing 2D asset: {e}")
            return None

    @staticmethod
    def _safe_delete_file(file_path):
        """Internal helper to securely delete a single file with logging."""
        if not file_path: return
        
        target = USER_DATA_ROOT / file_path
        if not target.exists():
            log.warning(f"Delete Skipped: File does not exist -> {target}")
            return
            
        # Security: Prevent escaping the app directory
        if USER_DATA_ROOT.resolve() not in target.resolve().parents:
            log.warning(f"Security Block: Attempted to delete external file -> {target}")
            return

        try:
            os.remove(target)
            log.info(f"[SUCCESS] Deleted File: {target.name}")
        except PermissionError:
            log.error(f"[LOCKED] Windows blocked deletion (File in use?): {target.name}")
        except Exception as e:
            log.error(f"[ERROR] Failed to delete {target.name}: {e}")

    @staticmethod
    def delete_all_item_assets(target_item):
        """
        Takes a JewelryItem object and systematically purges every file 
        (3D folder, 2D PNG, and Thumbnail) associated with it.
        """
        log.info(f"Purging assets for Item {target_item.id}: {target_item.name}")
        
        # 1. Delete the Thumbnail
        if target_item.thumbnail_path:
            LibraryManager._safe_delete_file(target_item.thumbnail_path)
            
        # 2. Delete the 2D Image (THIS WAS THE MISSING STEP!)
        if target_item.image_2d_path:
            LibraryManager._safe_delete_file(target_item.image_2d_path)

        # 3. Delete the 3D Model Folder
        if target_item.model_path:
            full_path = USER_DATA_ROOT / target_item.model_path
            
            # If DB path is a file (Single Item), target its parent folder
            if full_path.is_file():
                folder_to_delete = full_path.parent
            # If DB path is a folder (Collection), target it directly
            else:
                folder_to_delete = full_path
                
            # Security: Ensure we are inside 'data/models'
            if MODELS_DIR.resolve() not in folder_to_delete.resolve().parents and MODELS_DIR.resolve() != folder_to_delete.resolve():
                log.warning("Security Block: Attempted to delete external folder.")
                return

            if folder_to_delete.exists():
                try:
                    shutil.rmtree(folder_to_delete, ignore_errors=True)
                    log.info(f"[SUCCESS] Deleted 3D Folder: {folder_to_delete.name}")
                except Exception as e:
                    log.error(f"[ERROR] Failed to delete 3D Folder {folder_to_delete.name}: {e}")