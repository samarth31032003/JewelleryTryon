# trackers/strategies/base.py
import numpy as np
import cv2
from utils.smoothing import OneEuroFilter, RotationFilter

class TrackingStrategy:
    def __init__(self):
        # 1. Common Smoothing Filters
        # Tweaked values: High beta = fast response, Low beta = smooth
        self.filter_rvec = RotationFilter(min_cutoff=0.5, beta=1.0)
        self.filter_tvec = OneEuroFilter(min_cutoff=0.1, beta=1.0)
        
        # 2. Camera Matrix Cache
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # 2. Settings Store (Values for sliders)
        self.settings = {} 
        self.mode = "3d"

    def get_slider_definitions(self):
        """
        Returns a list of sliders this strategy needs.
        Format: [ (Key, Label, Min, Max, Default, Scale_Factor), ... ]
        """
        return []

    def update_settings(self, new_settings):
        """Receives dictionary of slider values from UI"""
        self.settings.update(new_settings)

    def update_camera(self, w, h):
        """Standard generic camera calibration"""
        if self.camera_matrix is None or self.camera_matrix[0,2] != w/2:
            focal_length = w 
            center = (w / 2, h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )

    def _build_2d_matrix(self, pos_3d, roll_angle, scale):
        """Builds a flat matrix facing the camera, only allowing Z-roll."""
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos_3d
        
        c, s = np.cos(roll_angle), np.sin(roll_angle)
        R = np.eye(4, dtype=np.float32)
        R[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        
        S = np.diag([scale, scale, scale, 1.0])
        
        cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        return cv_to_gl @ T @ R @ S

    def process_frame(self, results, width, height):
        """
        The Master Function.
        Returns a list of dictionaries:
        [
           {
             'type': 'mesh',       # or 'occluder'
             'matrix': 4x4_numpy,
             'id': 'left_hand_bracelet'
           },
           ...
        ]
        """
        raise NotImplementedError