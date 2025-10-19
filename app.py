"""
app.py
Flask API for Animal Type Classification System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import cv2
import numpy as np
from datetime import datetime
import json
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
from preprocessing import ImagePreprocessor
from keypoint_detector import AnimalKeypointDetector
from measurement import BodyMeasurementExtractor
from classifier import ATCClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize components
preprocessor = ImagePreprocessor(target_size=(640, 480))
detector = AnimalKeypointDetector()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Animal Type Classification API',
        'version': '1.0',
        'endpoints': {
            '/health': 'Health check',
            '/api/classify': 'POST - Classify animal from image',
            '/api/analyze': 'POST - Detailed analysis with visualization'
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/api/classify', methods=['POST'])
def classify_animal():
    """
    Main classification endpoint
    
    Expects:
    - image file
    - breed_type (cattle/buffalo)
    - gender (male/female)
    - reference_length_cm (optional, for calibration)
    - reference_length_pixels (optional, for calibration)
    """
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
    
    try:
        # Get parameters
        breed_type = request.form.get('breed_type', 'cattle')
        gender = request.form.get('gender', 'female')
        reference_cm = request.form.get('reference_length_cm', None)
        reference_px = request.form.get('reference_length_pixels', None)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        img, scale, offset = preprocessor.preprocess_pipeline(filepath)
        
        # Detect keypoints
        keypoints = detector.detect_keypoints_basic(img)
        
        if not keypoints:
            return jsonify({'error': 'Failed to detect animal keypoints'}), 400
        
        # Extract measurements
        if reference_cm and reference_px:
            extractor = BodyMeasurementExtractor(
                reference_length_cm=float(reference_cm),
                reference_length_pixels=float(reference_px)
            )
        else:
            extractor = BodyMeasurementExtractor()
        
        measurements = extractor.extract_measurements(keypoints)
        
        # Classify and score
        classifier = ATCClassifier(breed_type=breed_type, gender=gender)
        result = classifier.calculate_final_score(measurements)
        
        # Save visualization
        img_with_keypoints = detector.visualize_keypoints(img, keypoints)
        viz_filename = f"viz_{filename}"
        viz_path = os.path.join(app.config['PROCESSED_FOLDER'], viz_filename)
        cv2.imwrite(viz_path, cv2.cvtColor(img_with_keypoints, cv2.COLOR_RGB2BGR))
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'original_filename': file.filename,
                'processed_filename': filename,
                'visualization_path': viz_filename
            },
            'classification_result': result,
            'measurements': measurements,
            'keypoints_detected': len(keypoints)
        }
        
        # Save result to JSON
        result_filename = f"result_{timestamp}.json"
        result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
        with open(result_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/batch-classify', methods=['POST'])
def batch_classify():
    """
    Batch classification endpoint for multiple images
    """
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    breed_type = request.form.get('breed_type', 'cattle')
    gender = request.form.get('gender', 'female')
    
    results = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process
                img, _, _ = preprocessor.preprocess_pipeline(filepath)
                keypoints = detector.detect_keypoints_basic(img)
                
                if keypoints:
                    extractor = BodyMeasurementExtractor()
                    measurements = extractor.extract_measurements(keypoints)
                    classifier = ATCClassifier(breed_type=breed_type, gender=gender)
                    result = classifier.calculate_final_score(measurements)
                    
                    results.append({
                        'filename': file.filename,
                        'score': result['final_scores']['percentage'],
                        'grade': result['final_scores']['grade']
                    })
                else:
                    errors.append({'filename': file.filename, 'error': 'Keypoints not detected'})
                    
            except Exception as e:
                errors.append({'filename': file.filename, 'error': str(e)})
    
    return jsonify({
        'success': True,
        'processed': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors
    }), 200


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get statistics about processed animals
    """
    processed_dir = app.config['PROCESSED_FOLDER']
    
    if not os.path.exists(processed_dir):
        return jsonify({
            'total_processed': 0,
            'message': 'No processed data found'
        }), 200
    
    result_files = [f for f in os.listdir(processed_dir) if f.startswith('result_') and f.endswith('.json')]
    
    stats = {
        'total_processed': len(result_files),
        'grade_distribution': {},
        'average_score': 0,
        'recent_evaluations': []
    }
    
    scores = []
    
    for result_file in result_files[-10:]:  # Last 10 results
        try:
            with open(os.path.join(processed_dir, result_file), 'r') as f:
                data = json.load(f)
                score = data['classification_result']['final_scores']['percentage']
                grade = data['classification_result']['final_scores']['grade']
                
                scores.append(score)
                stats['grade_distribution'][grade] = stats['grade_distribution'].get(grade, 0) + 1
                
                stats['recent_evaluations'].append({
                    'timestamp': data['timestamp'],
                    'score': score,
                    'grade': grade
                })
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    if scores:
        stats['average_score'] = round(sum(scores) / len(scores), 2)
    
    return jsonify(stats), 200


if __name__ == '__main__':
    print("🚀 Starting Animal Type Classification API...")
    print("📍 Server will run on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")
    print("  POST /api/classify - Classify single image")
    print("  POST /api/batch-classify - Classify multiple images")
    print("  GET  /api/stats  - Get statistics")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)