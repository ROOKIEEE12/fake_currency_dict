import os
import io
import base64
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# Import TensorFlow for pre-trained validation
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Configuration
app = Flask(__name__)
MODEL_PATH = 'currency_model.pkl'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MIN_RF_CONFIDENCE = 0.85  # Higher threshold for the Random Forest

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Load the Custom Currency Classifier (Random Forest)
print("Loading Custom Currency Model...")
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Custom Model loaded successfully!")
    else:
        print("Warning: Custom Model not found. Please train it.")
        model = None
    CLASS_NAMES = ['Fake', 'Real'] 
except Exception as e:
    print(f"Error loading custom model: {e}")
    model = None

# 2. Load Pre-trained Validation Model (to filter non-currency images like cats)
print("Loading Validation Model (MobileNetV2)...")
try:
    # Use MobileNetV2 (trained on ImageNet) as a "gatekeeper"
    validator_model = MobileNetV2(weights='imagenet')
    print("Validation Model loaded successfully!")
except Exception as e:
    print(f"Error loading validation model: {e}")
    validator_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_likely_currency(img_bytes):
    """
    Uses MobileNetV2 to check if the image is likely a banknote OR if it's an animal/object.
    """
    if validator_model is None: return True # Fallback if model fails to load
    
    try:
        # Prepare image for MobileNetV2 (224x224)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Run prediction
        preds = validator_model.predict(x)
        decoded = decode_predictions(preds, top=5)[0]
        
        # Log for debugging
        print(f"Validator Top Predictions: {[d[1] for d in decoded]}")

        # List of things that are DEFINITELY NOT currency but common in accidental uploads
        # This includes cats, dogs, persons, and general objects that aren't flat/rectangular
        rejected_keywords = [
            'cat', 'dog', 'tiger', 'lion', 'animal', 'pet', 
            'person', 'face', 'human', # Keep these for major non-curreny
            'tree', 'plant', 'car', 'vehicle', 'building',
            'furniture', 'table', 'chair', 'cup', 'plate'
        ]
        
        # Keywords that might indicate the note is on a surface we should ignore
        suspicious_keywords = ['clothing', 'shirt', 'velvet', 'linen', 'jeans']

        # List of things that MIGHT be identified as currency or currency-like shapes
        currency_like_keywords = [
            'packet', 'wallet', 'envelope', 'notebook', 'book', 
            'paper_towel', 'ticket', 'coupon', 'mail', 'folder'
        ]

        for _, label, score in decoded:
            label_lower = label.lower()
            
            # If it explicitly sees an animal or person, reject immediately
            if any(k in label_lower for k in rejected_keywords):
                print(f"REJECTED: Validated as {label_lower}")
                return False, label.replace('_', ' ')

        # MobileNetV2 doesn't have a direct 'banknote' class, but 'packet' is often used.
        # If the image is very unknown or generic, we'll let it pass to the RF but 
        # the RF will have a higher threshold check.
        return True, None
    except Exception as e:
        print(f"Validator error: {e}")
        return True, None

def preprocess_for_rf(img_bytes):
    """Preprocessing for the custom Random Forest model (64x64)"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((64, 64))
        return np.array(img).flatten().reshape(1, -1)
    except Exception as e:
        print(f"RF Preprocessing error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'System warming up... Custom model not loaded.'}), 500

    image_bytes = None
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            image_bytes = file.read()
    elif 'image_data' in request.form:
        image_data = request.form['image_data']
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)

    if not image_bytes:
        return jsonify({'error': 'No valid image received'}), 400

    try:
        # STEP 1: Validate if it's actually a note (Prevention Logic)
        is_val, detected_label = is_likely_currency(image_bytes)
        if not is_val:
            return jsonify({
                'label': 'Invalid Photo',
                'invalid': True,
                'is_real': False,
                'confidence': 0.99,
                'message': f"Detection Failed: System identifies this as a '{detected_label}'. Please upload a clear photo of a currency note."
            })

        # STEP 2: Custom RF Prediction
        processed_img = preprocess_for_rf(image_bytes)
        if processed_img is None: return jsonify({'error': 'Processing Failed'}), 400

        probs = model.predict_proba(processed_img)[0]
        prediction_idx = np.argmax(probs)
        confidence = probs[prediction_idx]

        # STEP 3: Threshold check for RF
        # If it's something weird that the validator missed, the RF might still fail 
        # unless it has high confidence.
        if confidence < MIN_RF_CONFIDENCE:
            return jsonify({
                'label': 'Unknown Note',
                'invalid': True,
                'is_real': False,
                'confidence': float(confidence),
                'message': "Confidence too low to verify. Ensure the note is flat, well-lit, and fills the scanner area."
            })

        label = CLASS_NAMES[prediction_idx]
        return jsonify({
            'label': f"{label} Currency",
            'is_real': label == 'Real',
            'invalid': False,
            'confidence': float(confidence),
            'message': f"Scan Complete. Neural analysis suggests a {label.upper()} note."
        })

    except Exception as e:
        print(f"General error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_to_dataset', methods=['POST'])
def save_to_dataset():
    """Saves the last uploaded image to the training dataset for refinement."""
    try:
        data = request.json
        label = data.get('label') # 'real' or 'fake'
        image_data = data.get('image_data')

        if not label or not image_data:
            return jsonify({'error': 'Missing data'}), 400

        if "," in image_data:
            _, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        else:
            image_bytes = base64.b64decode(image_data)

        # Save to archive/data/data/<label>/custom/
        target_dir = os.path.join('archive', 'data', 'data', label, 'custom')
        os.makedirs(target_dir, exist_ok=True)
        
        import time
        filename = f"user_{int(time.time())}.jpg"
        filepath = os.path.join(target_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
            
        return jsonify({'success': True, 'message': f'Image saved to {label} dataset.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Triggers the training script to update the model with new data."""
    try:
        import subprocess
        import sys
        # Use the same python executable from the venv
        python_exe = sys.executable
        # Run the training script
        print(f"Starting Model Retraining with {python_exe}...")
        result = subprocess.run([python_exe, 'train_currency_model.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Reload model
            global model
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                print("Model Retrained and Reloaded Successfully!")
                return jsonify({'success': True, 'message': 'Model retrained successfully!'})
            else:
                return jsonify({'error': 'Training finished but model file not found.'}), 500
        else:
            print(f"Retraining Failed: {result.stderr}")
            return jsonify({'error': f'Retraining failed: {result.stderr}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Increase timeout and use a different port to avoid conflicts
    app.run(debug=True, port=8000, use_reloader=False)
