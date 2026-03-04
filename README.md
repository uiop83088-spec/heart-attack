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
- Multer for file uploads
- CORS enabled for cross-origin requests
- Support for TensorFlow.js or ONNX models

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

## Model Integration

To integrate actual deep learning models:

1. Train your models (CNN, RNN/LSTM)
2. Save models in `models/` directory
3. Update `app.py` functions:
   - `analyze_medical_image()` - Load CNN model
   - `analyze_ecg()` - Load RNN/LSTM model
   - `analyze_clinical_data()` - Load classification model

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
