# trackers/occluder_shared.py
import numpy as np
from utils.paths import OCCLUDER_HEAD_PATH, OCCLUDER_BODY_PATH

class OccluderManager:
    """
    Central manager for the shared 'Shadow Catcher' models.
    Prevents redundancy in Ear, Nose, Forehead, and Neck strategies.
    """
    
    # PATHS (You must place these .obj files in your data folder)
    PATH_HEAD = OCCLUDER_HEAD_PATH
    PATH_BODY = OCCLUDER_BODY_PATH

    @staticmethod
    def get_shared_sliders():
        """
        Common sliders for Head and Body scaling.
        We separate them so you can tune Head Width independently of Shoulder Width.
        """
        return [
            # HEAD SETTINGS (For Ear, Nose, Forehead)
            ("Occ_Head_Scale", "Head Size", 50, 150, 100, 1.0),
            ("Occ_Head_X", "Head X", -500, 500, 0, 1.0),
            ("Occ_Head_Y", "Head Y", -500, 500, 0, 1.0),
            ("Occ_Head_Z", "Head Z", -500, 500, 0, 1.0),
            
            # BODY SETTINGS (For Necklace)
            ("Occ_Body_Scale", "Body Size", 50, 200, 100, 1.0),
            ("Occ_Body_Y", "Body Height", -500, 500, 0, 1.0),
            ("Occ_Body_Z", "Body Depth", -500, 500, 0, 1.0),
        ]

    @staticmethod
    def get_head_matrix(settings, pos, rot_mat):
        """Generates the matrix for the Skull/Face occluder."""
        # 1. Base Position
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        
        # 2. User Offsets (High Precision)
        ox = settings.get("Occ_Head_X", 0) * 0.0001
        oy = settings.get("Occ_Head_Y", 0) * 0.0001
        oz = settings.get("Occ_Head_Z", 0) * 0.0001
        
        T_off = np.eye(4, dtype=np.float32)
        T_off[:3, 3] = [ox, oy, oz]
        
        # 3. Scale
        # Base scale logic: 100 = 1.0x. Adjust multiplier based on your OBJ size.
        # Assuming OBJ is approx real-scale (unit size ~20cm)
        raw_s = settings.get("Occ_Head_Scale", 100)
        s = raw_s * 0.01 
        S = np.diag([s, s, s, 1.0])
        
        # 4. Pipeline
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ rot_mat @ T_off @ S

    @staticmethod
    def get_body_matrix(settings, pos, rot_mat):
        """Generates the matrix for the Neck/Shoulder occluder."""
        T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
        
        # Body specific offsets (Usually just Height and Depth)
        oy = settings.get("Occ_Body_Y", 0) * 0.0001
        oz = settings.get("Occ_Body_Z", 0) * 0.0001
        
        T_off = np.eye(4, dtype=np.float32)
        T_off[:3, 3] = [0, oy, oz]
        
        raw_s = settings.get("Occ_Body_Scale", 100)
        s = raw_s * 0.01
        S = np.diag([s, s, s, 1.0])
        
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        return cv_to_gl @ T @ rot_mat @ T_off @ S