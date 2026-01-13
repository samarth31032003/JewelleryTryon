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
        
        # 3. State Memory
        self.last_valid_rvec = None
        self.last_valid_tvec = None

    def update_camera(self, w, h):
        """Standard generic camera calibration"""
        if self.camera_matrix is None or self.camera_matrix[0,2] != w/2:
            focal_length = w  # Approximation
            center = (w / 2, h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )

    def calculate_pose(self, results, width, height):
        """
        MUST be overridden by children (Wrist, Neck, etc.)
        Returns: (rvec, tvec, info_dict)
        """
        raise NotImplementedError("Strategies must implement calculate_pose")