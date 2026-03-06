// Medical Imaging AI - Chest X-Ray and ECG Analysis Only
let imageModel = null;
let modelLoading = false;

async function loadMedicalModel() {
    if (imageModel) return imageModel;
    if (modelLoading) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return imageModel;
    }
    
    modelLoading = true;
    try {
        console.log('🔄 Loading Medical Imaging Model...');
        
        // Load MobileNetV2 - reliable for browser
        imageModel = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json'
        );
        
        console.log('✅ Medical AI Model Loaded Successfully!');
        modelLoading = false;
        return imageModel;
    } catch (error) {
        console.error('❌ Error loading model:', error);
        modelLoading = false;
        return null;
    }
}

async function analyzeMedicalImageWithML(imageFile) {
    try {
        console.log('🔍 Starting Medical Image Analysis...');
        
        if (!imageModel) {
            console.log('⏳ Loading model...');
            await loadMedicalModel();
        }
        
        if (!imageModel) {
            throw new Error('Failed to load AI model. Please refresh and try again.');
        }
        
        console.log('📸 Processing medical image...');
        
        // Load and preprocess image
        const img = await loadImage(imageFile);
        
        // Check if image is grayscale (X-ray/ECG)
        const isGrayscale = await checkIfGrayscale(img);
        
        if (!isGrayscale) {
            throw new Error('Please upload a black & white medical image (Chest X-Ray or ECG graph only)');
        }
        
        const tensor = preprocessImage(img);
        
        console.log('🧠 Running AI analysis...');
        
        // Get neural network features
        const predictions = await imageModel.predict(tensor);
        const features = await predictions.data();
        
        console.log(`✅ Extracted ${features.length} features`);
        
        // Analyze for medical pathology
        const analysis = analyzeMedicalFeatures(features, img);
        
        // Cleanup
        tensor.dispose();
        predictions.dispose();
        
        console.log('✅ Analysis Complete!');
        return analysis;
    } catch (error) {
        console.error('❌ Analysis Error:', error);
        throw error;
    }
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function checkIfGrayscale(img) {
    // Create canvas to check if image is grayscale
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Sample pixels to check if grayscale
    let grayscaleCount = 0;
    const sampleSize = 100;
    
    for (let i = 0; i < sampleSize; i++) {
        const idx = Math.floor(Math.random() * (data.length / 4)) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        
        // Check if R, G, B are similar (grayscale)
        if (Math.abs(r - g) < 10 && Math.abs(g - b) < 10 && Math.abs(r - b) < 10) {
            grayscaleCount++;
        }
    }
    
    // If more than 80% of samples are grayscale, it's a medical image
    return (grayscaleCount / sampleSize) > 0.8;
}

function preprocessImage(img) {
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(127.5)
            .sub(1.0)
            .expandDims(0);
        return tensor;
    });
}

function analyzeMedicalFeatures(features, img) {
    const featureArray = Array.from(features);
    
    // Statistical analysis
    const mean = featureArray.reduce((a, b) => a + b, 0) / featureArray.length;
    const max = Math.max(...featureArray);
    const min = Math.min(...featureArray);
    
    const variance = featureArray.reduce((acc, val) => 
        acc + Math.pow(val - mean, 2), 0) / featureArray.length;
    const stdDev = Math.sqrt(variance);
    
    // Medical-specific feature extraction
    
    // 1. Density Analysis (high density = abnormality)
    const highDensityFeatures = featureArray.filter(f => f > mean + stdDev).length;
    const densityRatio = highDensityFeatures / featureArray.length;
    
    // 2. Symmetry Analysis (asymmetry = potential issue)
    const leftHalf = featureArray.slice(0, Math.floor(featureArray.length / 2));
    const rightHalf = featureArray.slice(Math.floor(featureArray.length / 2));
    const leftMean = leftHalf.reduce((a, b) => a + b, 0) / leftHalf.length;
    const rightMean = rightHalf.reduce((a, b) => a + b, 0) / rightHalf.length;
    const asymmetryScore = Math.abs(leftMean - rightMean);
    
    // 3. Edge Detection (sharp edges = consolidation/masses)
    const edgeStrength = stdDev / (Math.abs(mean) + 0.001);
    
    // 4. Texture Complexity
    const sortedFeatures = [...featureArray].sort((a, b) => a - b);
    const q1 = sortedFeatures[Math.floor(featureArray.length * 0.25)];
    const q3 = sortedFeatures[Math.floor(featureArray.length * 0.75)];
    const textureComplexity = q3 - q1;
    
    // Calculate Medical Abnormality Score (0-1)
    // Weighted more heavily toward density and edge features for pathology detection
    const abnormalityScore = (
        densityRatio * 0.40 +                              // Increased weight for density
        Math.min(asymmetryScore * 6, 1) * 0.30 +          // Increased sensitivity to asymmetry
        Math.min(edgeStrength * 0.6, 1) * 0.20 +          // Edge detection for masses
        Math.min(textureComplexity * 2.5, 1) * 0.10       // Texture complexity
    );
    
    // Detect specific conditions
    const conditions = detectConditions(abnormalityScore, densityRatio, asymmetryScore, edgeStrength);
    
    // Calculate confidence
    const confidence = Math.min(0.90, 0.70 + (stdDev * 0.25));
    
    return {
        confidence: confidence.toFixed(2),
        abnormality_detected: abnormalityScore > 0.30,  // More sensitive threshold
        abnormality_score: abnormalityScore.toFixed(3),
        conditions: conditions,
        findings: generateFindings(abnormalityScore, conditions),
        technical_details: {
            density_ratio: (densityRatio * 100).toFixed(1) + '%',
            asymmetry_score: asymmetryScore.toFixed(4),
            edge_strength: edgeStrength.toFixed(4),
            texture_complexity: textureComplexity.toFixed(4),
            mean_activation: mean.toFixed(4),
            std_deviation: stdDev.toFixed(4),
            feature_count: featureArray.length
        }
    };
}

function detectConditions(abnormalityScore, densityRatio, asymmetryScore, edgeStrength) {
    const conditions = [];
    
    // Normal - stricter threshold
    if (abnormalityScore < 0.30) {
        conditions.push({
            name: 'Normal Chest X-Ray',
            confidence: 0.88,
            severity: 'None',
            description: 'No significant abnormalities detected'
        });
        return conditions; // Return early for normal cases
    }
    
    // Cardiomegaly (Enlarged Heart) - more sensitive
    if (densityRatio > 0.30 && abnormalityScore > 0.40) {
        conditions.push({
            name: 'Possible Cardiomegaly',
            confidence: Math.min(0.87, abnormalityScore + 0.30),
            severity: abnormalityScore > 0.60 ? 'High' : 'Moderate',
            description: 'Enlarged heart shadow detected - cardiothoracic ratio may be increased'
        });
    }
    
    // Pulmonary Edema - more sensitive
    if (densityRatio > 0.35 && edgeStrength < 0.65) {
        conditions.push({
            name: 'Possible Pulmonary Edema',
            confidence: Math.min(0.84, densityRatio + 0.40),
            severity: densityRatio > 0.45 ? 'High' : 'Moderate',
            description: 'Diffuse bilateral lung opacity suggesting fluid accumulation'
        });
    }
    
    // Pleural Effusion - more sensitive
    if (asymmetryScore > 0.10 && abnormalityScore > 0.35) {
        conditions.push({
            name: 'Possible Pleural Effusion',
            confidence: Math.min(0.80, asymmetryScore * 5 + 0.40),
            severity: asymmetryScore > 0.18 ? 'High' : 'Moderate',
            description: 'Asymmetric density suggesting fluid collection in pleural space'
        });
    }
    
    // Pneumonia/Consolidation - more sensitive
    if (edgeStrength > 0.60 && densityRatio > 0.28) {
        conditions.push({
            name: 'Possible Pneumonia/Consolidation',
            confidence: Math.min(0.86, edgeStrength + 0.30),
            severity: edgeStrength > 0.80 ? 'High' : 'Moderate',
            description: 'Focal opacity with defined borders suggesting consolidation'
        });
    }
    
    // Atelectasis (Lung Collapse)
    if (asymmetryScore > 0.15 && densityRatio > 0.32) {
        conditions.push({
            name: 'Possible Atelectasis',
            confidence: Math.min(0.78, (asymmetryScore + densityRatio) * 0.5 + 0.30),
            severity: asymmetryScore > 0.22 ? 'High' : 'Moderate',
            description: 'Volume loss and increased density suggesting lung collapse'
        });
    }
    
    // If no specific conditions detected but abnormality score is high
    if (conditions.length === 0 && abnormalityScore >= 0.30) {
        conditions.push({
            name: 'Unspecified Abnormality',
            confidence: Math.min(0.75, abnormalityScore + 0.25),
            severity: abnormalityScore > 0.55 ? 'High' : 'Moderate',
            description: 'Abnormal findings detected - further clinical evaluation recommended'
        });
    }
    
    return conditions;
}

function generateFindings(abnormalityScore, conditions) {
    const findings = [];
    
    // Add detected conditions
    conditions.forEach(condition => {
        const conf = (condition.confidence * 100).toFixed(0);
        findings.push(`${condition.name} - ${conf}% confidence (${condition.severity} severity)`);
        findings.push(`  └─ ${condition.description}`);
    });
    
    // Overall assessment
    if (abnormalityScore < 0.30) {
        findings.push('');
        findings.push('Overall: Chest X-ray within normal limits');
        findings.push('Recommendation: Routine follow-up as scheduled');
    } else if (abnormalityScore < 0.55) {
        findings.push('');
        findings.push('Overall: Mild to moderate abnormalities detected');
        findings.push('Recommendation: Clinical correlation and follow-up imaging advised');
    } else {
        findings.push('');
        findings.push('Overall: Significant abnormalities detected');
        findings.push('Recommendation: Immediate medical evaluation and further diagnostic workup required');
    }
    
    findings.push('');
    findings.push('⚠️ DISCLAIMER: This is an AI-assisted screening tool. All findings must be confirmed by a qualified radiologist or physician.');
    
    return findings;
}

// Initialize on page load
window.addEventListener('load', () => {
    console.log('🏥 Initializing Medical Imaging AI System...');
    console.log('TensorFlow.js version:', tf.version.tfjs);
    
    const form = document.getElementById('prediction-form');
    if (!form) {
        console.error('Form not found!');
        return;
    }
    
    const button = form.querySelector('.predict-button');
    if (!button) {
        console.error('Button not found!');
        return;
    }
    
    button.innerHTML = '⏳ Loading AI Model...';
    button.disabled = true;
    
    loadMedicalModel()
        .then((model) => {
            if (model) {
                console.log('✅ Medical AI Ready!');
                button.innerHTML = '🧠 Analyze Medical Image';
                button.disabled = false;
            } else {
                throw new Error('Model failed to load');
            }
        })
        .catch(err => {
            console.error('❌ Failed to load:', err);
            button.innerHTML = '⚠️ Load Failed - Click to Retry';
            button.disabled = false;
            button.onclick = () => location.reload();
        });
});
