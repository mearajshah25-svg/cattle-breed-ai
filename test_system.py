"""
test_system.py
Complete system test script for Animal Type Classification
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import ImagePreprocessor
from keypoint_detector import AnimalKeypointDetector
from measurement import BodyMeasurementExtractor
from classifier import ATCClassifier


class ATCSystemTester:
    """Complete system testing and demonstration"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor(target_size=(640, 480))
        self.detector = AnimalKeypointDetector()
        self.extractor = BodyMeasurementExtractor(
            reference_length_cm=50,
            reference_length_pixels=100
        )
        self.classifier = None
        
    def generate_test_image(self, output_path='data/raw/test_sample.jpg'):
        """Generate a test image for demonstration"""
        # Create a blank canvas
        img = np.ones((600, 800, 3), dtype=np.uint8) * 240
        
        # Draw a simple cow shape
        # Body (ellipse)
        cv2.ellipse(img, (400, 350), (200, 120), 0, 0, 360, (139, 69, 19), -1)
        
        # Head (circle)
        cv2.circle(img, (220, 300), 60, (139, 69, 19), -1)
        
        # Ears
        pts_ear1 = np.array([[200, 260], [180, 230], [210, 250]], np.int32)
        pts_ear2 = np.array([[240, 260], [260, 230], [230, 250]], np.int32)
        cv2.fillPoly(img, [pts_ear1, pts_ear2], (139, 69, 19))
        
        # Legs
        leg_positions = [(300, 350), (350, 350), (450, 350), (500, 350)]
        for x, y in leg_positions:
            cv2.rectangle(img, (x-15, y), (x+15, y+200), (139, 69, 19), -1)
            cv2.circle(img, (x, y+200), 15, (80, 40, 10), -1)  # Hooves
        
        # Tail
        pts_tail = np.array([[580, 330], [620, 380], [600, 340]], np.int32)
        cv2.fillPoly(img, [pts_tail], (139, 69, 19))
        
        # Add reference line (50cm = 100px)
        cv2.line(img, (50, 50), (150, 50), (255, 0, 0), 3)
        cv2.putText(img, '50cm', (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"✅ Generated test image: {output_path}")
        return output_path
    
    def test_preprocessing(self, image_path):
        """Test image preprocessing"""
        print("\n" + "="*60)
        print("🔍 TESTING: Image Preprocessing")
        print("="*60)
        
        try:
            img, scale, offset = self.preprocessor.preprocess_pipeline(
                image_path, 
                enhance=True, 
                denoise_img=False
            )
            
            print(f"✅ Image loaded successfully")
            print(f"   Shape: {img.shape}")
            print(f"   Scale: {scale:.4f}")
            print(f"   Offset: {offset}")
            
            # Save processed image
            output_path = 'data/processed/test_preprocessed.jpg'
            os.makedirs('data/processed', exist_ok=True)
            self.preprocessor.save_processed_image(img, output_path)
            print(f"✅ Saved preprocessed image: {output_path}")
            
            return img, scale, offset
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None, None, None
    
    def test_keypoint_detection(self, image):
        """Test keypoint detection"""
        print("\n" + "="*60)
        print("🎯 TESTING: Keypoint Detection")
        print("="*60)
        
        try:
            keypoints = self.detector.detect_keypoints_basic(image)
            
            if keypoints:
                print(f"✅ Detected {len(keypoints)} keypoints:")
                for name, (x, y) in list(keypoints.items())[:5]:
                    print(f"   {name}: ({x:.1f}, {y:.1f})")
                print(f"   ... and {len(keypoints)-5} more")
                
                # Visualize
                img_with_keypoints = self.detector.visualize_keypoints(image, keypoints)
                output_path = 'data/processed/test_keypoints.jpg'
                os.makedirs('data/processed', exist_ok=True)
                cv2.imwrite(output_path, cv2.cvtColor(img_with_keypoints, cv2.COLOR_RGB2BGR))
                print(f"✅ Saved keypoint visualization: {output_path}")
                
                return keypoints
            else:
                print("❌ No keypoints detected")
                return None
                
        except Exception as e:
            print(f"❌ Keypoint detection failed: {e}")
            return None
    
    def test_measurement_extraction(self, keypoints):
        """Test measurement extraction"""
        print("\n" + "="*60)
        print("📏 TESTING: Measurement Extraction")
        print("="*60)
        
        try:
            measurements = self.extractor.extract_measurements(keypoints)
            
            print(f"✅ Extracted {len(measurements)} measurements:")
            for key, value in list(measurements.items())[:8]:
                unit = 'cm' if 'angle' not in key and 'index' not in key and 'proportion' not in key else ('°' if 'angle' in key else '')
                print(f"   {key}: {value:.2f} {unit}")
            
            if len(measurements) > 8:
                print(f"   ... and {len(measurements)-8} more")
            
            return measurements
            
        except Exception as e:
            print(f"❌ Measurement extraction failed: {e}")
            return None
    
    def test_classification(self, measurements, breed_type='cattle', gender='female'):
        """Test ATC classification"""
        print("\n" + "="*60)
        print("⭐ TESTING: ATC Classification")
        print("="*60)
        
        try:
            self.classifier = ATCClassifier(breed_type=breed_type, gender=gender)
            result = self.classifier.calculate_final_score(measurements)
            
            print(f"\n✅ Classification completed!")
            print(f"\n📊 RESULTS:")
            print(f"   Animal Type: {result['animal_info']['breed_type'].title()}")
            print(f"   Gender: {result['animal_info']['gender'].title()}")
            print(f"\n   Final Score: {result['final_scores']['percentage']}%")
            print(f"   Linear Scale: {result['final_scores']['linear_scale']}/9")
            print(f"   Grade: {result['final_scores']['grade']}")
            
            print(f"\n📈 Category Breakdown:")
            for cat in result['category_scores']:
                print(f"   {cat['category']}: {cat['score']:.1f}/100 (weight: {cat['weight']*100:.0f}%)")
            
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations']:
                print(f"   {rec}")
            
            return result
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            return None
    
    def run_full_test(self, image_path=None, breed_type='cattle', gender='female'):
        """Run complete system test"""
        print("\n" + "="*70)
        print("🚀 ANIMAL TYPE CLASSIFICATION SYSTEM - FULL TEST")
        print("="*70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate test image if none provided
        if image_path is None or not os.path.exists(image_path):
            print("\n⚠️  No test image provided. Generating sample image...")
            image_path = self.generate_test_image()
        
        # Step 1: Preprocessing
        img, scale, offset = self.test_preprocessing(image_path)
        if img is None:
            print("\n❌ Test failed at preprocessing stage")
            return False
        
        # Step 2: Keypoint Detection
        keypoints = self.test_keypoint_detection(img)
        if keypoints is None:
            print("\n❌ Test failed at keypoint detection stage")
            return False
        
        # Step 3: Measurement Extraction
        measurements = self.test_measurement_extraction(keypoints)
        if measurements is None:
            print("\n❌ Test failed at measurement extraction stage")
            return False
        
        # Step 4: Classification
        result = self.test_classification(measurements, breed_type, gender)
        if result is None:
            print("\n❌ Test failed at classification stage")
            return False
        
        # Success!
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📁 Output files generated:")
        print("   - data/processed/test_preprocessed.jpg")
        print("   - data/processed/test_keypoints.jpg")
        
        return True
    
    def test_batch_processing(self, image_folder='data/raw'):
        """Test batch processing of multiple images"""
        print("\n" + "="*60)
        print("🔄 TESTING: Batch Processing")
        print("="*60)
        
        if not os.path.exists(image_folder):
            os.makedirs(image_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("⚠️  No images found in folder. Generating test images...")
            for i in range(3):
                self.generate_test_image(f'data/raw/test_sample_{i+1}.jpg')
            image_files = [f for f in os.listdir(image_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        results = []
        print(f"📷 Processing {len(image_files)} images...\n")
        
        for i, filename in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {filename}")
            image_path = os.path.join(image_folder, filename)
            
            try:
                img, _, _ = self.preprocessor.preprocess_pipeline(image_path)
                keypoints = self.detector.detect_keypoints_basic(img)
                
                if keypoints:
                    measurements = self.extractor.extract_measurements(keypoints)
                    classifier = ATCClassifier(breed_type='cattle', gender='female')
                    result = classifier.calculate_final_score(measurements)
                    
                    results.append({
                        'filename': filename,
                        'score': result['final_scores']['percentage'],
                        'grade': result['final_scores']['grade'],
                        'status': 'success'
                    })
                    print(f"   ✅ Score: {result['final_scores']['percentage']}% - {result['final_scores']['grade']}")
                else:
                    results.append({
                        'filename': filename,
                        'status': 'failed',
                        'error': 'Keypoint detection failed'
                    })
                    print(f"   ❌ Failed: Keypoint detection")
            
            except Exception as e:
                results.append({
                    'filename': filename,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"   ❌ Failed: {e}")
        
        # Summary
        print(f"\n📊 Batch Processing Summary:")
        successful = sum(1 for r in results if r.get('status') == 'success')
        print(f"   Total: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        
        if successful > 0:
            avg_score = np.mean([r['score'] for r in results if r.get('status') == 'success'])
            print(f"   Average Score: {avg_score:.2f}%")
        
        return results


def print_menu():
    """Print test menu"""
    print("\n" + "="*70)
    print("🐄 ANIMAL TYPE CLASSIFICATION SYSTEM - TEST MENU")
    print("="*70)
    print("1. Run Full System Test (Single Image)")
    print("2. Test Preprocessing Only")
    print("3. Test Keypoint Detection Only")
    print("4. Test Measurement Extraction Only")
    print("5. Test Classification Only")
    print("6. Run Batch Processing Test")
    print("7. Generate Sample Test Image")
    print("8. Run All Tests")
    print("0. Exit")
    print("="*70)


def main():
    """Main test runner"""
    tester = ATCSystemTester()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == '0':
            print("\n👋 Exiting. Thank you!")
            break
        
        elif choice == '1':
            image_path = input("\nEnter image path (or press Enter for sample): ").strip()
            breed = input("Enter breed type (cattle/buffalo) [cattle]: ").strip() or 'cattle'
            gender = input("Enter gender (male/female) [female]: ").strip() or 'female'
            
            if not image_path:
                image_path = None
            
            tester.run_full_test(image_path, breed, gender)
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            image_path = input("\nEnter image path: ").strip()
            if os.path.exists(image_path):
                tester.test_preprocessing(image_path)
            else:
                print("❌ Image not found!")
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            image_path = input("\nEnter image path: ").strip()
            if os.path.exists(image_path):
                img, _, _ = tester.preprocessor.preprocess_pipeline(image_path)
                tester.test_keypoint_detection(img)
            else:
                print("❌ Image not found!")
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            print("\n⚠️  This requires keypoints. Running full pipeline...")
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                img, _, _ = tester.preprocessor.preprocess_pipeline(image_path)
                keypoints = tester.detector.detect_keypoints_basic(img)
                if keypoints:
                    tester.test_measurement_extraction(keypoints)
            else:
                print("❌ Image not found!")
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            print("\n⚠️  This requires measurements. Using sample data...")
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
            breed = input("Enter breed type (cattle/buffalo) [cattle]: ").strip() or 'cattle'
            gender = input("Enter gender (male/female) [female]: ").strip() or 'female'
            tester.test_classification(sample_measurements, breed, gender)
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            folder = input("\nEnter folder path [data/raw]: ").strip() or 'data/raw'
            tester.test_batch_processing(folder)
            input("\nPress Enter to continue...")
        
        elif choice == '7':
            output = input("\nEnter output path [data/raw/test_sample.jpg]: ").strip()
            output = output or 'data/raw/test_sample.jpg'
            tester.generate_test_image(output)
            input("\nPress Enter to continue...")
        
        elif choice == '8':
            print("\n🚀 Running all tests...")
            
            # Generate test image
            test_img = tester.generate_test_image('data/raw/test_all.jpg')
            
            # Run full test
            success = tester.run_full_test(test_img, 'cattle', 'female')
            
            if success:
                # Run batch test
                tester.test_batch_processing('data/raw')
            
            input("\nPress Enter to continue...")
        
        else:
            print("\n❌ Invalid choice. Please try again.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🐄 ANIMAL TYPE CLASSIFICATION SYSTEM")
    print("   Test & Demonstration Script")
    print("="*70)
    print("\nThis script will test all components of the ATC system:")
    print("  ✓ Image preprocessing")
    print("  ✓ Keypoint detection")
    print("  ✓ Measurement extraction")
    print("  ✓ ATC classification and scoring")
    print("  ✓ Batch processing")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()