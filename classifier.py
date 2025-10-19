"""
classifier.py
Animal Type Classification (ATC) Scoring System
Based on standard cattle/buffalo evaluation criteria
"""

import numpy as np
from typing import Dict, Tuple
from datetime import datetime

class ATCClassifier:
    """
    Classify and score animals based on body measurements
    Following standard ATC scoring guidelines (1-9 scale or percentage)
    """
    
    # ATC scoring categories and their weights
    SCORING_CATEGORIES = {
        'general_appearance': 0.15,
        'dairy_character': 0.20,
        'body_capacity': 0.20,
        'feet_and_legs': 0.15,
        'udder': 0.30  # Only for females
    }
    
    # Breed-specific ideal measurements (example for Holstein-Friesian)
    BREED_STANDARDS = {
        'cattle': {
            'body_length': {'min': 145, 'ideal': 160, 'max': 175},  # cm
            'height_at_withers': {'min': 130, 'ideal': 145, 'max': 160},
            'body_depth': {'min': 70, 'ideal': 80, 'max': 90},
            'chest_width': {'min': 45, 'ideal': 55, 'max': 65},
            'rump_angle': {'min': 15, 'ideal': 25, 'max': 35},
            'back_straightness': {'min': 165, 'ideal': 175, 'max': 185}
        },
        'buffalo': {
            'body_length': {'min': 150, 'ideal': 165, 'max': 180},
            'height_at_withers': {'min': 125, 'ideal': 140, 'max': 155},
            'body_depth': {'min': 75, 'ideal': 85, 'max': 95},
            'chest_width': {'min': 50, 'ideal': 60, 'max': 70},
            'rump_angle': {'min': 20, 'ideal': 30, 'max': 40},
            'back_straightness': {'min': 160, 'ideal': 170, 'max': 180}
        }
    }
    
    def __init__(self, breed_type='cattle', gender='female'):
        """
        Initialize classifier
        
        Args:
            breed_type: 'cattle' or 'buffalo'
            gender: 'male' or 'female'
        """
        self.breed_type = breed_type
        self.gender = gender
        self.standards = self.BREED_STANDARDS.get(breed_type, self.BREED_STANDARDS['cattle'])
    
    def normalize_measurement(self, value: float, standard: Dict[str, float]) -> float:
        """
        Normalize measurement to 0-100 scale based on breed standards
        
        Args:
            value: Measured value
            standard: Dict with 'min', 'ideal', 'max' values
        
        Returns:
            Score 0-100
        """
        min_val = standard['min']
        ideal_val = standard['ideal']
        max_val = standard['max']
        
        if value < min_val:
            # Below minimum - poor score
            score = max(0, 50 - (min_val - value) * 2)
        elif value <= ideal_val:
            # Between min and ideal - score increases to 100
            score = 50 + ((value - min_val) / (ideal_val - min_val)) * 50
        elif value <= max_val:
            # Between ideal and max - score decreases from 100
            score = 100 - ((value - ideal_val) / (max_val - ideal_val)) * 50
        else:
            # Above maximum - poor score
            score = max(0, 50 - (value - max_val) * 2)
        
        return max(0, min(100, score))
    
    def score_general_appearance(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Score general appearance (15% weight)
        - Overall balance and symmetry
        - Body length and proportion
        - Style and attractiveness
        """
        scores = {}
        
        # Body proportion
        if 'body_proportion' in measurements:
            ideal_proportion = 1.15 if self.breed_type == 'cattle' else 1.20
            deviation = abs(measurements['body_proportion'] - ideal_proportion)
            scores['proportion'] = max(0, 100 - deviation * 50)
        
        # Body length score
        if 'body_length' in measurements:
            scores['body_length'] = self.normalize_measurement(
                measurements['body_length'],
                self.standards['body_length']
            )
        
        # Overall balance
        if 'back_straightness' in measurements:
            scores['balance'] = self.normalize_measurement(
                measurements['back_straightness'],
                self.standards['back_straightness']
            )
        
        # Average score
        category_score = np.mean(list(scores.values())) if scores else 0
        
        return {
            'category': 'General Appearance',
            'score': category_score,
            'weight': self.SCORING_CATEGORIES['general_appearance'],
            'weighted_score': category_score * self.SCORING_CATEGORIES['general_appearance'],
            'details': scores
        }
    
    def score_dairy_character(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Score dairy character (20% weight)
        - Angularity and sharpness
        - Refinement of head and bone
        - Body depth relative to stature
        """
        scores = {}
        
        # Body depth (indicator of dairy capacity)
        if 'body_depth' in measurements:
            scores['body_depth'] = self.normalize_measurement(
                measurements['body_depth'],
                self.standards['body_depth']
            )
        
        # Body capacity index
        if 'body_capacity_index' in measurements:
            # Higher capacity index indicates better dairy character
            ideal_index = 55 if self.breed_type == 'cattle' else 58
            deviation = abs(measurements['body_capacity_index'] - ideal_index)
            scores['capacity_index'] = max(0, 100 - deviation * 2)
        
        # Chest width
        if 'chest_width' in measurements:
            scores['chest_width'] = self.normalize_measurement(
                measurements['chest_width'],
                self.standards['chest_width']
            )
        
        category_score = np.mean(list(scores.values())) if scores else 0
        
        return {
            'category': 'Dairy Character',
            'score': category_score,
            'weight': self.SCORING_CATEGORIES['dairy_character'],
            'weighted_score': category_score * self.SCORING_CATEGORIES['dairy_character'],
            'details': scores
        }
    
    def score_body_capacity(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Score body capacity (20% weight)
        - Barrel depth and width
        - Spring of rib
        - Body capacity for feed intake
        """
        scores = {}
        
        # Height at withers
        if 'height_at_withers' in measurements:
            scores['stature'] = self.normalize_measurement(
                measurements['height_at_withers'],
                self.standards['height_at_withers']
            )
        
        # Body depth
        if 'body_depth' in measurements:
            scores['depth'] = self.normalize_measurement(
                measurements['body_depth'],
                self.standards['body_depth']
            )
        
        # Chest capacity
        if 'chest_width' in measurements:
            scores['chest'] = self.normalize_measurement(
                measurements['chest_width'],
                self.standards['chest_width']
            )
        
        category_score = np.mean(list(scores.values())) if scores else 0
        
        return {
            'category': 'Body Capacity',
            'score': category_score,
            'weight': self.SCORING_CATEGORIES['body_capacity'],
            'weighted_score': category_score * self.SCORING_CATEGORIES['body_capacity'],
            'details': scores
        }
    
    def score_feet_and_legs(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Score feet and legs (15% weight)
        - Leg straightness and placement
        - Foot angle and depth
        - Bone quality
        """
        scores = {}
        
        # Front leg length
        if 'front_leg_length' in measurements:
            ideal_length = self.standards['height_at_withers']['ideal'] * 0.55
            actual = measurements['front_leg_length']
            deviation = abs(actual - ideal_length) / ideal_length * 100
            scores['front_legs'] = max(0, 100 - deviation)
        
        # Rear leg length
        if 'rear_leg_length' in measurements:
            ideal_length = self.standards['height_at_withers']['ideal'] * 0.60
            actual = measurements['rear_leg_length']
            deviation = abs(actual - ideal_length) / ideal_length * 100
            scores['rear_legs'] = max(0, 100 - deviation)
        
        # Leg symmetry
        if 'front_leg_length' in measurements and 'rear_leg_length' in measurements:
            leg_ratio = measurements['rear_leg_length'] / measurements['front_leg_length']
            ideal_ratio = 1.09
            deviation = abs(leg_ratio - ideal_ratio)
            scores['leg_balance'] = max(0, 100 - deviation * 100)
        
        category_score = np.mean(list(scores.values())) if scores else 75  # Default good score
        
        return {
            'category': 'Feet and Legs',
            'score': category_score,
            'weight': self.SCORING_CATEGORIES['feet_and_legs'],
            'weighted_score': category_score * self.SCORING_CATEGORIES['feet_and_legs'],
            'details': scores
        }
    
    def score_rump(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Score rump structure (included in udder/reproductive system scoring)
        - Rump angle
        - Rump width and length
        - Pin bone placement
        """
        scores = {}
        
        # Rump angle
        if 'rump_angle' in measurements:
            scores['rump_angle'] = self.normalize_measurement(
                measurements['rump_angle'],
                self.standards['rump_angle']
            )
        
        # Rump length
        if 'rump_length' in measurements:
            ideal_length = 45 if self.breed_type == 'cattle' else 48
            actual = measurements['rump_length']
            deviation = abs(actual - ideal_length) / ideal_length * 100
            scores['rump_length'] = max(0, 100 - deviation)
        
        # Rump width
        if 'rump_width' in measurements:
            ideal_width = 50 if self.breed_type == 'cattle' else 53
            actual = measurements['rump_width']
            deviation = abs(actual - ideal_width) / ideal_width * 100
            scores['rump_width'] = max(0, 100 - deviation)
        
        return scores
    
    def calculate_final_score(self, measurements: Dict[str, float]) -> Dict:
        """
        Calculate final ATC score
        
        Returns complete scoring breakdown
        """
        # Score each category
        general = self.score_general_appearance(measurements)
        dairy = self.score_dairy_character(measurements)
        capacity = self.score_body_capacity(measurements)
        legs = self.score_feet_and_legs(measurements)
        
        # Rump score (30% weight for females, combined with reproductive traits)
        rump_scores = self.score_rump(measurements)
        rump_score = np.mean(list(rump_scores.values())) if rump_scores else 75
        
        rump_result = {
            'category': 'Rump Structure',
            'score': rump_score,
            'weight': 0.30,
            'weighted_score': rump_score * 0.30,
            'details': rump_scores
        }
        
        # Compile all scores
        all_categories = [general, dairy, capacity, legs, rump_result]
        
        # Calculate final weighted score
        total_weighted = sum(cat['weighted_score'] for cat in all_categories)
        
        # Convert to standard ATC scale (1-9 or percentage)
        final_percentage = total_weighted
        final_linear_score = (final_percentage / 100) * 8 + 1  # Convert to 1-9 scale
        
        # Classification grade
        if final_percentage >= 90:
            grade = 'Excellent'
        elif final_percentage >= 80:
            grade = 'Very Good'
        elif final_percentage >= 70:
            grade = 'Good Plus'
        elif final_percentage >= 60:
            grade = 'Good'
        elif final_percentage >= 50:
            grade = 'Fair'
        else:
            grade = 'Poor'
        
        return {
            'animal_info': {
                'breed_type': self.breed_type,
                'gender': self.gender,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'category_scores': all_categories,
            'final_scores': {
                'percentage': round(final_percentage, 2),
                'linear_scale': round(final_linear_score, 2),
                'grade': grade
            },
            'recommendations': self.generate_recommendations(measurements, all_categories)
        }
    
    def generate_recommendations(self, measurements: Dict[str, float], 
                                category_scores: list) -> list:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        # Check each category
        for category in category_scores:
            if category['score'] < 70:
                recommendations.append(
                    f"⚠️ {category['category']} needs improvement (Score: {category['score']:.1f})"
                )
        
        # Specific measurement recommendations
        if 'body_depth' in measurements:
            if measurements['body_depth'] < self.standards['body_depth']['min']:
                recommendations.append("📏 Body depth below standard - focus on nutrition")
        
        if 'rump_angle' in measurements:
            if measurements['rump_angle'] < 20 or measurements['rump_angle'] > 35:
                recommendations.append("🔄 Rump angle outside ideal range - consider in breeding selection")
        
        if not recommendations:
            recommendations.append("✅ All parameters within acceptable ranges")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Sample measurements from previous modules
    sample_measurements = {
        'body_length': 158.5,
        'height_at_withers': 142.3,
        'body_depth': 78.6,
        'chest_width': 54.2,
        'rump_length': 44.8,
        'rump_width': 49.5,
        'rump_angle': 26.5,
        'back_straightness': 173.2,
        'front_leg_length': 78.3,
        'rear_leg_length': 85.4,
        'body_capacity_index': 55.2,
        'body_proportion': 1.14
    }
    
    # Initialize classifier
    classifier = ATCClassifier(breed_type='cattle', gender='female')
    
    # Calculate final score
    result = classifier.calculate_final_score(sample_measurements)
    
    # Print report
    print("\n" + "="*60)
    print("🐄 ANIMAL TYPE CLASSIFICATION (ATC) REPORT")
    print("="*60)
    
    print(f"\n📋 Animal Information:")
    for key, value in result['animal_info'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 Category Scores:")
    for cat in result['category_scores']:
        print(f"\n   {cat['category']}:")
        print(f"      Raw Score: {cat['score']:.2f}/100")
        print(f"      Weight: {cat['weight']*100:.0f}%")
        print(f"      Weighted: {cat['weighted_score']:.2f}")
        if cat['details']:
            print(f"      Details: {', '.join([f'{k}: {v:.1f}' for k, v in cat['details'].items()])}")
    
    print(f"\n🎯 Final Scores:")
    print(f"   Percentage: {result['final_scores']['percentage']}%")
    print(f"   Linear Scale (1-9): {result['final_scores']['linear_scale']}")
    print(f"   Grade: {result['final_scores']['grade']}")
    
    print(f"\n💡 Recommendations:")
    for rec in result['recommendations']:
        print(f"   {rec}")
    
    print("\n" + "="*60)