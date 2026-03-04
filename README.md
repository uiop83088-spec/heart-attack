# DeepHealthX: Multi-Modal Deep Learning for Early Detection of Heart Diseases

A full-stack web application that uses deep learning to analyze multiple data modalities for early heart disease detection.

## Features

- **Multi-Modal Analysis**: Combines medical imaging, ECG data, and clinical information
- **Deep Learning Models**: CNN for images, RNN/LSTM for ECG, ensemble prediction
- **Real-time Predictions**: Instant risk assessment and recommendations
- **Modern UI**: Responsive design with smooth animations

## Tech Stack

### Frontend
- HTML5, CSS3, JavaScript
- Responsive design with CSS Grid/Flexbox
- Fetch API for backend communication

### Backend
- Node.js with Express
- TensorFlow.js for deep learning inference
- Sharp for image processing
- Real-time ECG signal analysis

### Machine Learning
- **Image Analysis**: MobileNet CNN for medical image feature extraction
- **ECG Analysis**: Custom signal processing algorithms for heart rhythm detection
- **Clinical Data**: Risk factor analysis and ensemble prediction

## Installation

### Prerequisites
- Node.js 14+ and npm

### Setup

1. Install dependencies:
```bash
npm install
```

2. Start the server:
```bash
npm start
```

Or for development with auto-reload:
```bash
npm run dev
```

3. Access the application at `http://localhost:5000`

## Real ML Features

### Medical Image Analysis
- Uses TensorFlow.js with MobileNet architecture
- Analyzes actual image pixels and structure
- Detects anomalies based on activation patterns
- Provides confidence scores and technical details

### ECG Signal Processing
- Real-time heart rate calculation from ECG data
- Peak detection (R-wave identification)
- Rhythm analysis (Normal Sinus, Bradycardia, Tachycardia)
- ST segment elevation/depression detection
- Heart Rate Variability (HRV) calculation

### Clinical Data Integration
- Multi-modal ensemble prediction
- Combines image, ECG, and clinical risk factors
- Generates personalized recommendations

## Usage

1. Navigate to the "Try the AI Model" section
2. Upload medical images (PNG, JPG, DICOM)
3. Upload ECG data (CSV or TXT format)
4. Enter clinical information (age, gender, blood pressure, cholesterol)
5. Click "Analyze with AI" to get predictions
6. View risk assessment and recommendations

## API Endpoints

### POST /api/predict
Analyzes uploaded data and returns risk assessment

**Request:**
- Form data with files and clinical parameters

**Response:**
```json
{
  "success": true,
  "risk_score": 45.2,
  "risk_level": "Moderate Risk",
  "predictions": {
    "image_analysis": {...},
    "ecg_analysis": {...},
    "clinical_analysis": {...}
  },
  "recommendations": [...]
}
```

### GET /api/health
Health check endpoint

## Model Training

To train custom models on your own medical imaging data:

1. Navigate to the training directory:
```bash
cd training
pip install -r requirements.txt
```

2. Prepare your dataset (see training/README.md for details)

3. Train models:
```bash
python train_model.py
```

4. Models will be exported to `models/nodejs/` for integration

See `training/README.md` for detailed instructions on:
- Downloading medical imaging datasets
- Training custom CNN models
- Training ECG LSTM models
- Exporting models for Node.js deployment

## Project Structure

```
DeepHealthX/
├── index.html          # Main webpage
├── styles.css          # Styling
├── script.js           # Frontend logic
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── uploads/            # Uploaded files (auto-created)
└── README.md          # Documentation
```

## Future Enhancements

- Database integration for patient records
- User authentication and authorization
- Model training pipeline
- Real-time monitoring dashboard
- Mobile application
- HIPAA compliance features

## Disclaimer

This is a demonstration project. Not intended for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## License

MIT License
