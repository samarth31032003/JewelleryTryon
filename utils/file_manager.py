# utils/file_manager.py
import shutil
import os
import uuid
from pathlib import Path
from utils.paths import MODELS_DIR, THUMBNAILS_DIR, USER_DATA_ROOT

class LibraryManager:
    """
    Handles secure copying of user assets into the App's Library.
    """

    @staticmethod
    def import_file(source_path, target_folder_type="model"):
        """
        Copies a file to the library and returns the RELATIVE path for the DB.
        target_folder_type: 'model' or 'thumbnail'
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Determine Target Directory
        if target_folder_type == "model":
            dest_dir = MODELS_DIR
        else:
            dest_dir = THUMBNAILS_DIR

        # Generate a safe unique name to prevent overwrites (e.g., ring_uuid.obj)
        # We keep the original extension
        clean_name = f"{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}"
        destination = dest_dir / clean_name

        try:
            shutil.copy2(source, destination)
            # Return relative path string (e.g., "models/ring_a1b2.obj")
            # We use forward slashes for cross-platform DB compatibility
            rel_path = f"{target_folder_type}s/{clean_name}"
            return rel_path
        except Exception as e:
            print(f"Error importing file: {e}")
            return None

    @staticmethod
    def import_collection(source_path):
        """
        Copies an entire folder (Collection) to the library.
        Returns relative path: 'models/CollectionName_uuid'
        """
        source = Path(source_path)
        if not source.exists() or not source.is_dir():
            print(f"Error: Source collection not found: {source}")
            return None

        # Create unique folder name
        folder_name = f"{source.name}_{uuid.uuid4().hex[:8]}"
        destination = MODELS_DIR / folder_name

        try:
            # Copy the entire directory tree
            shutil.copytree(source, destination)
            return f"models/{folder_name}"
        except Exception as e:
            print(f"Error importing collection: {e}")
            return None

    @staticmethod
    def delete_file(file_path):
        """
        Safely deletes a file/folder if it exists inside the User Data Root.
        Accepts absolute paths (e.g. C:/.../models/file.obj).
        """
        if not file_path: return
        
        target = Path(file_path).resolve()
        library_root = USER_DATA_ROOT.resolve()

        # SECURITY CHECK:
        # Ensure the target path is actually inside our 'library' folder.
        # This prevents deleting system files if a bad path is passed.
        if library_root not in target.parents:
            print(f"Security Block: Attempted to delete external file: {target}")
            return

        try:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target) # Delete folder (Collections)
                else:
                    os.remove(target)     # Delete file (Single items)
                print(f"Deleted library item: {target.name}")
        except Exception as e:
            print(f"Error deleting {target}: {e}")