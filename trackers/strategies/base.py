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