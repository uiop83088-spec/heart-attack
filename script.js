// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Image preview
document.getElementById('medical-image').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const preview = document.getElementById('image-preview');
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" style="max-width: 300px; max-height: 300px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                <p style="margin-top: 0.5rem; font-size: 0.9rem;">✓ ${file.name}</p>`;
        };
        reader.readAsDataURL(file);
    }
});

// Deep Learning Analysis - Medical Chest X-Ray Only
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultsDiv = document.getElementById('results');
    const predictButton = this.querySelector('.predict-button');
    const imageFile = document.getElementById('medical-image').files[0];
    
    if (!imageFile) {
        alert('⚠️ Please upload a chest X-ray image first');
        return;
    }
    
    // Check file size
    if (imageFile.size > 10 * 1024 * 1024) {
        alert('⚠️ Image too large. Please use an image smaller than 10MB');
        return;
    }
    
    predictButton.innerHTML = '🏥 Analyzing Chest X-Ray with Medical AI...';
    predictButton.disabled = true;
    
    try {
        // Check if TensorFlow.js is loaded
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js not loaded. Please refresh the page.');
        }
        
        // Check if medical analysis function exists
        if (typeof analyzeMedicalChestXray === 'undefined') {
            throw new Error('Medical analysis system not loaded. Please refresh the page.');
        }
        
        console.log('Starting medical chest X-ray analysis...');
        const mlResult = await analyzeMedicalChestXray(imageFile);
        
        if (!mlResult) {
            throw new Error('Medical analysis returned no results.');
        }
        
        console.log('Medical analysis successful:', mlResult);
        
        // Display results
        displayMLResults(mlResult);
        resultsDiv.classList.remove('hidden');
        
        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert('❌ Analysis Error:\n\n' + error.message + '\n\nPlease try:\n1. Refresh the page\n2. Check your internet connection\n3. Use a chest X-ray image');
    } finally {
        predictButton.innerHTML = '🧠 Analyze Chest X-Ray with AI';
        predictButton.disabled = false;
    }
});

function displayMLResults(mlResult) {
    // Calculate risk score from ML analysis
    const riskScore = (parseFloat(mlResult.anomaly_score) * 100).toFixed(1);
    
    document.getElementById('risk-percentage').textContent = riskScore;
    
    let riskLevel, riskClass;
    if (riskScore < 35) {
        riskLevel = 'Low Risk';
        riskClass = 'low-risk';
    } else if (riskScore < 60) {
        riskLevel = 'Moderate Risk';
        riskClass = 'moderate-risk';
    } else {
        riskLevel = 'High Risk';
        riskClass = 'high-risk';
    }
    
    const riskLevelEl = document.getElementById('risk-level');
    riskLevelEl.textContent = riskLevel;
    riskLevelEl.className = riskClass;
    
    // Display detailed ML analysis
    const detailDiv = document.getElementById('predictions-detail');
    const confidence = (parseFloat(mlResult.confidence) * 100).toFixed(1);
    
    let detailHTML = `
        <h4>🏥 Medical AI Analysis - Chest X-Ray Pathology Detection</h4>
        <div class="prediction-item">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <strong>Deep Learning Medical Analysis</strong>
                <span class="ml-badge">DenseNet121 Medical</span>
            </div>
            <p><strong>Overall Confidence:</strong> ${confidence}%</p>
            <p><strong>Pathology Status:</strong> ${mlResult.anomaly_detected ? '⚠️ Abnormalities Detected' : '✓ Normal Chest X-Ray'}</p>
            <p><strong>Abnormality Score:</strong> ${mlResult.anomaly_score} (0-1 scale)</p>
            
            <h5 style="margin-top: 1.5rem;">🔍 Detected Conditions:</h5>
            <div style="background: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
                ${mlResult.detected_conditions.map(cond => `
                    <div style="margin-bottom: 0.5rem;">
                        <strong>${cond.name}</strong><br>
                        <small>Confidence: ${(cond.confidence * 100).toFixed(0)}% | Severity: ${cond.severity}</small>
                    </div>
                `).join('<hr style="margin: 0.5rem 0;">')}
            </div>
            
            <h5 style="margin-top: 1.5rem;">📋 Clinical Findings:</h5>
            <ul>${mlResult.findings.map(f => `<li>${f}</li>`).join('')}</ul>
            
            <h5 style="margin-top: 1.5rem;">🔬 Technical Analysis:</h5>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; font-size: 0.9rem;">
                <p><strong>Feature Count:</strong> ${mlResult.technical_details.feature_count}</p>
                <p><strong>Mean Activation:</strong> ${mlResult.technical_details.mean_activation}</p>
                <p><strong>High Activation Ratio:</strong> ${mlResult.technical_details.high_activation_ratio}</p>
                <p><strong>Asymmetry Score:</strong> ${mlResult.technical_details.asymmetry_score}</p>
                <p><strong>Edge Strength:</strong> ${mlResult.technical_details.edge_strength}</p>
                <p><strong>Texture Complexity:</strong> ${mlResult.technical_details.texture_complexity}</p>
                <p><strong>Processing:</strong> Client-side Neural Network</p>
            </div>
        </div>
    `;
    
    detailDiv.innerHTML = detailHTML;
    
    // Display recommendations based on risk
    const recDiv = document.getElementById('recommendations');
    let recommendations;
    
    if (riskScore < 30) {
        recommendations = [
            'Continue regular health monitoring',
            'Maintain healthy lifestyle habits',
            'Schedule routine checkup within 12 months',
            'No immediate action required'
        ];
    } else if (riskScore < 60) {
        recommendations = [
            'Consult with a cardiologist for detailed evaluation',
            'Consider additional diagnostic tests',
            'Monitor symptoms closely',
            'Follow-up within 3-6 months recommended'
        ];
    } else {
        recommendations = [
            '⚠️ Immediate consultation with cardiologist strongly recommended',
            'Further diagnostic imaging may be required',
            'Do not delay medical attention',
            'Consider emergency evaluation if experiencing symptoms'
        ];
    }
    
    recDiv.innerHTML = '<h4>Medical Recommendations:</h4><ul>' + 
        recommendations.map(r => `<li>${r}</li>`).join('') + '</ul>' +
        '<p style="margin-top: 1rem; font-size: 0.9rem; color: #666;"><em>Note: This is an AI-assisted analysis. Always consult qualified healthcare professionals for medical diagnosis.</em></p>';
}

// File input labels
document.querySelectorAll('input[type="file"]').forEach(input => {
    input.addEventListener('change', function() {
        const label = this.nextElementSibling;
        if (this.files.length > 0) {
            label.textContent = this.files[0].name;
        }
    });
});

// Contact form handler
document.querySelector('.contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    alert('Thank you for your interest! We will get back to you soon.');
    this.reset();
});

// Add scroll effect to navbar
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%)';
    } else {
        navbar.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    }
});

// Animate elements on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('section').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(20px)';
    section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(section);
});
