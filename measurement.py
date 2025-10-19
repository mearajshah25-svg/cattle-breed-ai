"""
measurement.py
Extract body measurements from keypoints for Animal Type Classification
"""

import numpy as np
import math
from typing import Dict, Tuple

class BodyMeasurementExtractor:
    """
    Extract standardized body measurements for ATC scoring
    """
    
    # Reference measurements for calibration (in cm)
    # These would be set based on a known reference object in image
    PIXEL_TO_CM_RATIO = 1.0  # Will be calibrated per image
    
    def __init__(self, reference_length_cm=None, reference_length_pixels=None):
        """
        Initialize measurement extractor
        
        Args:
            reference_length_cm: Known length in cm (e.g., measuring tape)
            reference_length_pixels: Corresponding length in pixels
        """
        if reference_length_cm and reference_length_pixels:
            self.PIXEL_TO_CM_RATIO = reference_length_cm / reference_length_pixels
        
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       vertex: Tuple[float, float], 
                       point2: Tuple[float, float]) -> float:
        """
        Calculate angle at vertex formed by three points (in degrees)
        """
        # Vector from vertex to point1
        v1 = (point1[0] - vertex[0], point1[1] - vertex[1])
        # Vector from vertex to point2
        v2 = (point2[0] - vertex[0], point2[1] - vertex[1])
        
        # Calculate angle using dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def extract_measurements(self, keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Extract all relevant body measurements for ATC
        
        Returns measurements in pixels (convert to cm using ratio)
        """
        measurements = {}
        
        try:
            # 1. Body Length (nose to tail base)
            if 'nose' in keypoints and 'tail_base' in keypoints:
                measurements['body_length'] = self.calculate_distance(
                    keypoints['nose'], keypoints['tail_base']
                ) * self.PIXEL_TO_CM_RATIO
            
            # 2. Height at Withers (withers to ground approximation)
            if 'withers' in keypoints and 'front_hoof' in keypoints:
                measurements['height_at_withers'] = abs(
                    keypoints['withers'][1] - keypoints['front_hoof'][1]
                ) * self.PIXEL_TO_CM_RATIO
            
            # 3. Body Depth (withers to chest bottom)
            if 'withers' in keypoints and 'chest_bottom' in keypoints:
                measurements['body_depth'] = abs(
                    keypoints['withers'][1] - keypoints['chest_bottom'][1]
                ) * self.PIXEL_TO_CM_RATIO
            
            # 4. Chest Width (approximate from body width at chest)
            if 'shoulder_front' in keypoints:
                # In side view, estimate based on body proportions
                # Real implementation would need front/rear view
                measurements['chest_width'] = measurements.get('body_depth', 0) * 0.8
            
            # 5. Rump Length (hip to tail base)
            if 'hip' in keypoints and 'tail_base' in keypoints:
                measurements['rump_length'] = self.calculate_distance(
                    keypoints['hip'], keypoints['tail_base']
                ) * self.PIXEL_TO_CM_RATIO
            
            # 6. Rump Width (estimated from body proportions)
            if 'hip' in keypoints:
                measurements['rump_width'] = measurements.get('body_depth', 0) * 0.85
            
            # 7. Rump Angle (angle of rump slope)
            if all(k in keypoints for k in ['hip', 'rump_top', 'tail_base']):
                measurements['rump_angle'] = self.calculate_angle(
                    keypoints['hip'], keypoints['rump_top'], keypoints['tail_base']
                )
            
            # 8. Back Line Straightness (withers-back-hip alignment)
            if all(k in keypoints for k in ['withers', 'back_mid', 'hip']):
                measurements['back_straightness'] = self.calculate_angle(
                    keypoints['withers'], keypoints['back_mid'], keypoints['hip']
                )
            
            # 9. Leg Length (front)
            if 'shoulder_front' in keypoints and 'front_hoof' in keypoints:
                measurements['front_leg_length'] = self.calculate_distance(
                    keypoints['shoulder_front'], keypoints['front_hoof']
                ) * self.PIXEL_TO_CM_RATIO
            
            # 10. Leg Length (rear)
            if 'hip_back' in keypoints and 'back_hoof' in keypoints:
                measurements['rear_leg_length'] = self.calculate_distance(
                    keypoints['hip_back'], keypoints['back_hoof']
                ) * self.PIXEL_TO_CM_RATIO
            
            # 11. Body Capacity Index (body depth / height ratio)
            if 'body_depth' in measurements and 'height_at_withers' in measurements:
                measurements['body_capacity_index'] = (
                    measurements['body_depth'] / measurements['height_at_withers'] * 100
                )
            
            # 12. Body Proportion (body length / height ratio)
            if 'body_length' in measurements and 'height_at_withers' in measurements:
                measurements['body_proportion'] = (
                    measurements['body_length'] / measurements['height_at_withers']
                )
            
        except Exception as e:
            print(f"Error extracting measurements: {e}")
        
        return measurements
    
    def calculate_derived_scores(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate derived scores and indices for ATC
        """
        scores = {}
        
        # Body Conformation Score (based