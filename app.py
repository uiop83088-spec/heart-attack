from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'csv', 'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded files
        medical_image = request.files.get('medical_image')
        ecg_data = request.files.get('ecg_data')
        
        # Get clinical data
        age = request.form.get('age', type=int)
        gender = request.form.get('gender')
        blood_pressure = request.form.get('blood_pressure')
        cholesterol = request.form.get('cholesterol', type=float)
        
        results = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'risk_score': 0,
            'risk_level': '',
            'predictions': {},
            'recommendations': []
        }
        
        # Process medical image
        if medical_image and allowed_file(medical_image.filename):
            filename = secure_filename(medical_image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            medical_image.save(filepath)
            
            image_prediction = analyze_medical_image(filepath)
            results['predictions']['image_analysis'] = image_prediction
        
        # Process ECG data
        if ecg_data and allowed_file(ecg_data.filename):
            filename = secure_filename(ecg_data.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            ecg_data.save(filepath)
            
            ecg_prediction = analyze_ecg(filepath)
            results['predictions']['ecg_analysis'] = ecg_prediction
        
        # Process clinical data
        if age and gender:
            clinical_prediction = analyze_clinical_data(age, gender, blood_pressure, cholesterol)
            results['predictions']['clinical_analysis'] = clinical_prediction
        
        # Combine predictions
        final_result = combine_predictions(results['predictions'])
        results['risk_score'] = final_result['risk_score']
        results['risk_level'] = final_result['risk_level']
        results['recommendations'] = final_result['recommendations']
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def analyze_medical_image(filepath):
    """Simulate CNN-based medical image analysis"""
    # In production, load actual model: model = load_model('models/image_model.h5')
    np.random.seed(hash(filepath) % 2**32)
    
    return {
        'confidence': round(np.random.uniform(0.75, 0.98), 2),
        'findings': [
            'Cardiac structure appears normal',
            'No significant abnormalities detected',
            'Slight calcification observed'
        ],
        'anomaly_detected': np.random.choice([True, False], p=[0.3, 0.7])
    }

def analyze_ecg(filepath):
    """Simulate RNN/LSTM-based ECG analysis"""
    np.random.seed(hash(filepath) % 2**32)
    
    return {
        'confidence': round(np.random.uniform(0.80, 0.95), 2),
        'heart_rate': np.random.randint(60, 100),
        'rhythm': np.random.choice(['Normal Sinus Rhythm', 'Irregular', 'Tachycardia']),
        'abnormalities': np.random.choice([
            ['ST elevation detected'],
            ['Normal ECG pattern'],
            ['Mild arrhythmia']
        ])
    }

def analyze_clinical_data(age, gender, blood_pressure, cholesterol):
    """Analyze clinical parameters"""
    risk_factors = []
    
    if age > 55:
        risk_factors.append('Age over 55')
    if cholesterol and cholesterol > 200:
        risk_factors.append('High cholesterol')
    if blood_pressure and 'high' in blood_pressure.lower():
        risk_factors.append('Hypertension')
    
    return {
        'risk_factors': risk_factors,
        'risk_factor_count': len(risk_factors),
        'age_group': 'high_risk' if age > 60 else 'moderate_risk' if age > 45 else 'low_risk'
    }

def combine_predictions(predictions):
    """Combine multi-modal predictions using ensemble approach"""
    risk_scores = []
    
    if 'image_analysis' in predictions:
        if predictions['image_analysis']['anomaly_detected']:
            risk_scores.append(0.7)
        else:
            risk_scores.append(0.2)
    
    if 'ecg_analysis' in predictions:
        if predictions['ecg_analysis']['rhythm'] != 'Normal Sinus Rhythm':
            risk_scores.append(0.6)
        else:
            risk_scores.append(0.15)
    
    if 'clinical_analysis' in predictions:
        factor_count = predictions['clinical_analysis']['risk_factor_count']
        risk_scores.append(min(factor_count * 0.25, 0.8))
    
    avg_risk = np.mean(risk_scores) if risk_scores else 0.1
    risk_percentage = round(avg_risk * 100, 1)
    
    if risk_percentage < 30:
        risk_level = 'Low Risk'
        recommendations = [
            'Maintain healthy lifestyle',
            'Regular exercise recommended',
            'Annual checkup advised'
        ]
    elif risk_percentage < 60:
        risk_level = 'Moderate Risk'
        recommendations = [
            'Consult with cardiologist',
            'Monitor blood pressure regularly',
            'Consider lifestyle modifications',
            'Follow-up in 3-6 months'
        ]
    else:
        risk_level = 'High Risk'
        recommendations = [
            'Immediate consultation with cardiologist required',
            'Further diagnostic tests recommended',
            'Consider medication options',
            'Lifestyle changes essential'
        ]
    
    return {
        'risk_score': risk_percentage,
        'risk_level': risk_level,
        'recommendations': recommendations
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'DeepHealthX API'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
