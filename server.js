const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { MedicalImageAnalyzer, ECGAnalyzer } = require('./ml-models');

const app = express();

// Initialize ML models
const imageAnalyzer = new MedicalImageAnalyzer();
const ecgAnalyzer = new ECGAnalyzer();

// Load models on startup
imageAnalyzer.loadModel().catch(err => console.error('Failed to load image model:', err));

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('.'));

// Configure multer for memory storage (Vercel doesn't support disk storage)
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 16 * 1024 * 1024 }, // 16MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|dcm|csv|txt/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype) || file.mimetype === 'text/plain';
        
        if (extname || mimetype) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type'));
        }
    }
});

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/api/predict', upload.fields([
    { name: 'medical_image', maxCount: 1 },
    { name: 'ecg_data', maxCount: 1 }
]), async (req, res) => {
    try {
        const { age, gender, blood_pressure, cholesterol } = req.body;
        const files = req.files;
        
        const results = {
            success: true,
            timestamp: new Date().toISOString(),
            risk_score: 0,
            risk_level: '',
            predictions: {},
            recommendations: []
        };
        
        // Process medical image with real ML analysis
        if (files && files.medical_image) {
            const imageFile = files.medical_image[0];
            results.predictions.image_analysis = await imageAnalyzer.analyze(imageFile.buffer);
        }
        
        // Process ECG data with real signal analysis
        if (files && files.ecg_data) {
            const ecgFile = files.ecg_data[0];
            results.predictions.ecg_analysis = await ecgAnalyzer.analyze(ecgFile.buffer);
        }
        
        // Process clinical data
        if (age && gender) {
            results.predictions.clinical_analysis = analyzeClinicalData(
                parseInt(age), gender, blood_pressure, parseFloat(cholesterol)
            );
        }
        
        // Combine predictions
        const finalResult = combinePredictions(results.predictions);
        results.risk_score = finalResult.risk_score;
        results.risk_level = finalResult.risk_level;
        results.recommendations = finalResult.recommendations;
        
        res.json(results);
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(400).json({ success: false, error: error.message });
    }
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'healthy', service: 'DeepHealthX API' });
});

// Clinical data analysis (kept as is)


function analyzeClinicalData(age, gender, blood_pressure, cholesterol) {
    const risk_factors = [];
    
    if (age > 55) {
        risk_factors.push('Age over 55');
    }
    if (cholesterol && cholesterol > 200) {
        risk_factors.push('High cholesterol');
    }
    if (blood_pressure && blood_pressure.includes('high')) {
        risk_factors.push('Hypertension');
    }
    
    return {
        risk_factors: risk_factors,
        risk_factor_count: risk_factors.length,
        age_group: age > 60 ? 'high_risk' : age > 45 ? 'moderate_risk' : 'low_risk'
    };
}

function combinePredictions(predictions) {
    const risk_scores = [];
    
    if (predictions.image_analysis) {
        risk_scores.push(predictions.image_analysis.anomaly_detected ? 0.7 : 0.2);
    }
    
    if (predictions.ecg_analysis) {
        risk_scores.push(predictions.ecg_analysis.rhythm !== 'Normal Sinus Rhythm' ? 0.6 : 0.15);
    }
    
    if (predictions.clinical_analysis) {
        const factor_count = predictions.clinical_analysis.risk_factor_count;
        risk_scores.push(Math.min(factor_count * 0.25, 0.8));
    }
    
    const avg_risk = risk_scores.length > 0 
        ? risk_scores.reduce((a, b) => a + b, 0) / risk_scores.length 
        : 0.1;
    
    const risk_percentage = parseFloat((avg_risk * 100).toFixed(1));
    
    let risk_level, recommendations;
    
    if (risk_percentage < 30) {
        risk_level = 'Low Risk';
        recommendations = [
            'Maintain healthy lifestyle',
            'Regular exercise recommended',
            'Annual checkup advised'
        ];
    } else if (risk_percentage < 60) {
        risk_level = 'Moderate Risk';
        recommendations = [
            'Consult with cardiologist',
            'Monitor blood pressure regularly',
            'Consider lifestyle modifications',
            'Follow-up in 3-6 months'
        ];
    } else {
        risk_level = 'High Risk';
        recommendations = [
            'Immediate consultation with cardiologist required',
            'Further diagnostic tests recommended',
            'Consider medication options',
            'Lifestyle changes essential'
        ];
    }
    
    return {
        risk_score: risk_percentage,
        risk_level: risk_level,
        recommendations: recommendations
    };
}

// Export for Vercel serverless
module.exports = app;

// For local development
if (require.main === module) {
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
        console.log(`DeepHealthX server running on http://localhost:${PORT}`);
    });
}
