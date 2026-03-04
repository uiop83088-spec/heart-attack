const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('.'));

// Create uploads directory
const uploadDir = 'uploads';
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({
    storage: storage,
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
]), (req, res) => {
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
        
        // Process medical image
        if (files.medical_image) {
            const imagePath = files.medical_image[0].path;
            results.predictions.image_analysis = analyzeMedicalImage(imagePath);
        }
        
        // Process ECG data
        if (files.ecg_data) {
            const ecgPath = files.ecg_data[0].path;
            results.predictions.ecg_analysis = analyzeECG(ecgPath);
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

// Analysis functions
function analyzeMedicalImage(filepath) {
    // Simulate CNN-based medical image analysis
    const seed = filepath.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const random = () => {
        const x = Math.sin(seed) * 10000;
        return x - Math.floor(x);
    };
    
    return {
        confidence: parseFloat((0.75 + random() * 0.23).toFixed(2)),
        findings: [
            'Cardiac structure appears normal',
            'No significant abnormalities detected',
            'Slight calcification observed'
        ],
        anomaly_detected: random() > 0.7
    };
}

function analyzeECG(filepath) {
    // Simulate RNN/LSTM-based ECG analysis
    const seed = filepath.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const random = () => {
        const x = Math.sin(seed * 2) * 10000;
        return x - Math.floor(x);
    };
    
    const rhythms = ['Normal Sinus Rhythm', 'Irregular', 'Tachycardia'];
    const abnormalities = [
        ['ST elevation detected'],
        ['Normal ECG pattern'],
        ['Mild arrhythmia']
    ];
    
    return {
        confidence: parseFloat((0.80 + random() * 0.15).toFixed(2)),
        heart_rate: Math.floor(60 + random() * 40),
        rhythm: rhythms[Math.floor(random() * rhythms.length)],
        abnormalities: abnormalities[Math.floor(random() * abnormalities.length)]
    };
}

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

// Start server
app.listen(PORT, () => {
    console.log(`DeepHealthX server running on http://localhost:${PORT}`);
    console.log(`API endpoint: http://localhost:${PORT}/api/predict`);
});
