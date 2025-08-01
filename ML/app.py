
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import os
from PIL import Image
import json

# Import our virtual try-on system
from virtual_tryon_main import VirtualTryOnSystem

app = Flask(__name__)
CORS(app)

# Initialize the virtual try-on system
try_on_system = VirtualTryOnSystem()

# Configuration
UPLOAD_FOLDER = 'uploads'
CLOTHING_FOLDER = 'clothing_items'
OUTPUT_FOLDER = 'outputs'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, CLOTHING_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if 'data:image' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(img_bytes))

        # Convert to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return cv_image
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def image_to_base64(cv_image):
    """Convert OpenCV image to base64 string"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/clothing-items')
def get_clothing_items():
    """Get available clothing items"""
    clothing_items = []

    for filename in os.listdir(CLOTHING_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            item_info = {
                'id': filename,
                'name': filename.split('.')[0].replace('_', ' ').title(),
                'category': 'shirt',  # Default category
                'image_path': f'/clothing/{filename}'
            }
            clothing_items.append(item_info)

    return jsonify(clothing_items)

@app.route('/clothing/<filename>')
def serve_clothing(filename):
    """Serve clothing item images"""
    return send_file(os.path.join(CLOTHING_FOLDER, filename))

@app.route('/api/try-on', methods=['POST'])
def try_on():
    """Virtual try-on endpoint"""
    try:
        data = request.json

        # Get user image from base64
        user_image_b64 = data.get('user_image')
        clothing_item_id = data.get('clothing_item_id')
        clothing_type = data.get('clothing_type', 'shirt')

        if not user_image_b64 or not clothing_item_id:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Convert user image from base64
        user_image = base64_to_image(user_image_b64)
        if user_image is None:
            return jsonify({'error': 'Invalid user image'}), 400

        # Load clothing item
        clothing_path = os.path.join(CLOTHING_FOLDER, clothing_item_id)
        if not os.path.exists(clothing_path):
            return jsonify({'error': 'Clothing item not found'}), 404

        clothing_image = cv2.imread(clothing_path)
        if clothing_image is None:
            return jsonify({'error': 'Could not load clothing item'}), 400

        # Process the virtual try-on
        result_image = try_on_system.process_frame(user_image, clothing_image, clothing_type)

        # Convert result to base64
        result_b64 = image_to_base64(result_image)
        if result_b64 is None:
            return jsonify({'error': 'Failed to process result image'}), 500

        return jsonify({
            'success': True,
            'result_image': result_b64,
            'message': 'Virtual try-on completed successfully'
        })

    except Exception as e:
        print(f"Error in try-on endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/pose-detection', methods=['POST'])
def detect_pose():
    """Pose detection endpoint"""
    try:
        data = request.json
        user_image_b64 = data.get('user_image')

        if not user_image_b64:
            return jsonify({'error': 'Missing user image'}), 400

        # Convert user image from base64
        user_image = base64_to_image(user_image_b64)
        if user_image is None:
            return jsonify({'error': 'Invalid user image'}), 400

        # Detect pose
        annotated_image, pose_landmarks = try_on_system.detect_pose(user_image)

        # Convert annotated image to base64
        annotated_b64 = image_to_base64(annotated_image)

        return jsonify({
            'success': True,
            'annotated_image': annotated_b64,
            'pose_landmarks': pose_landmarks,
            'has_pose': pose_landmarks is not None
        })

    except Exception as e:
        print(f"Error in pose detection: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/body-segmentation', methods=['POST'])
def segment_body():
    """Body segmentation endpoint"""
    try:
        data = request.json
        user_image_b64 = data.get('user_image')

        if not user_image_b64:
            return jsonify({'error': 'Missing user image'}), 400

        # Convert user image from base64
        user_image = base64_to_image(user_image_b64)
        if user_image is None:
            return jsonify({'error': 'Invalid user image'}), 400

        # Segment body
        segmented_image, mask = try_on_system.segment_body(user_image)

        # Convert images to base64
        segmented_b64 = image_to_base64(segmented_image)

        return jsonify({
            'success': True,
            'segmented_image': segmented_b64,
            'has_body': np.any(mask > 0)
        })

    except Exception as e:
        print(f"Error in body segmentation: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Virtual Try-On Web Application...")
    print("Available endpoints:")
    print("- GET  /                     - Main application")
    print("- GET  /api/clothing-items   - Get clothing items")
    print("- POST /api/try-on           - Virtual try-on")
    print("- POST /api/pose-detection   - Pose detection")
    print("- POST /api/body-segmentation - Body segmentation")

    app.run(debug=True, host='0.0.0.0', port=5000)
