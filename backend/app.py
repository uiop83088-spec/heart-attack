from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess medical image for model input"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_ecg(ecg_data):
    """Preprocess ECG signal data"""
    # Normalize ECG data
    ecg_array = np.array(ecg_data)
    ecg_normalized = (ecg_array - np.mean(ecg_array)) / np.std(ecg_array)
    return ecg_normalized.reshape(1, -1)

def create_simple_model():
    """Create a simple CNN model for demonstration"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Initialize model (in production, load pre-trained weights)
model = create_simple_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'DeepHealthX API is running'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded files
        image_file = request.files.get('image')
        ecg_file = request.files.get('ecg')
        clinical_data = request.form.get('clinical_data')
        
        results = {
            'risk_score': 0,
            'confidence': 0,
            'analysis': {}
        }
        
        # Process medical image
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            
            # Preprocess and predict
            processed_img = preprocess_image(image_path)
            image_prediction = np.random.random()  # Simulated prediction
            
            results['analysis']['image'] = {
                'processed': True,
                'risk_score': float(image_prediction),
                'findings': ['Normal cardiac structure', 'No significant abnormalities detected']
            }
            results['risk_score'] += image_prediction * 0.4
        
        # Process ECG data
        if ecg_file:
            ecg_data = ecg_file.read().decode('utf-8')
            ecg_values = [float(x) for x in ecg_data.split(',') if x.strip()]
            
            # Simulated ECG analysis
            ecg_risk = np.random.random()
            results['analysis']['ecg'] = {
                'processed': True,
                'risk_score': float(ecg_risk),
                'heart_rate': int(np.random.randint(60, 100)),
                'rhythm': 'Normal sinus rhythm',
                'findings': ['Regular rhythm', 'No ST-segment elevation']
            }
            results['risk_score'] += ecg_risk * 0.4
        
        # Process clinical data
        if clinical_data:
            clinical = json.loads(clinical_data)
            
            # Calculate risk based on clinical factors
            clinical_risk = 0
            if clinical.get('age', 0) > 50:
                clinical_risk += 0.2
            if clinical.get('cholesterol', 0) > 200:
                clinical_risk += 0.2
            if clinical.get('blood_pr