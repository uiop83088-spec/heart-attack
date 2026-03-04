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

// Prediction form handler with client-side ML
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const resultsDiv = document.getElementById('results');
    const predictButton = this.querySelector('.predict-button');
    
    predictButton.textContent = 'Analyzing with MobileNetV2...';
    predictButton.disabled = true;
    
    try {
        // Get medical image file for client-side ML analysis
        const imageFile = formData.get('medical_image');
        let clientMLResult = null;
        
        if (imageFile && imageFile.size > 0 && typeof analyzeMedicalImageWithML !== 'undefined') {
            try {
                predictButton.textContent = 'Running MobileNetV2 analysis...';
                clientMLResult = await analyzeMedicalImageWithML(imageFile);
            } catch (mlError) {
                console.error('Client-side ML error:', mlError);
            }
        }
        
        // Send to server for ECG and clinical analysis
        predictButton.textContent = 'Processing ECG and clinical data...';
        const apiUrl = window.location.hostname === 'localhost' 
            ? 'http://localhost:5000/api/predict'
            : '/api/predict';
            
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Merge client-side ML results with server results
        if (clientMLResult && data.success) {
            data.predictions.image_analysis = {
                ...data.predictions.image_analysis,
                ml_analysis: clientMLResult,
                model: 'MobileNetV2',
                processing: 'client-side'
            };
        }
        
        if (data.success) {
            displayResults(data);
            resultsDiv.classList.remove('hidden');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Connection error: ' + error.message);
        console.error(error);
    } finally {
        predictButton.textContent = 'Analyze with AI';
        predictButton.disabled = false;
    }
});

function displayResults(data) {
    document.getElementById('risk-percentage').textContent = data.risk_score;
    document.getElementById('risk-level').textContent = data.risk_level;
    
    const riskLevel = document.getElementById('risk-level');
    riskLevel.className = '';
    if (data.risk_score < 30) {
        riskLevel.classList.add('low-risk');
    } else if (data.risk_score < 60) {
        riskLevel.classList.add('moderate-risk');
    } else {
        riskLevel.classList.add('high-risk');
    }
    
    // Display detailed predictions
    const detailDiv = document.getElementById('predictions-detail');
    let detailHTML = '<h4>Detailed Analysis:</h4>';
    
    if (data.predictions.image_analysis) {
        const imgAnalysis = data.predictions.image_analysis;
        const confidence = imgAnalysis.confidence * 100;
        
        detailHTML += `<div class="prediction-item">
            <strong>Medical Image Analysis:</strong> 
            ${imgAnalysis.ml_analysis ? '<span class="ml-badge">MobileNetV2</span>' : ''}
            Confidence ${confidence.toFixed(1)}%`;
        
        // Show ML-specific findings if available
        if (imgAnalysis.ml_analysis) {
            detailHTML += `<ul>${imgAnalysis.ml_analysis.findings.map(f => `<li>${f}</li>`).join('')}</ul>`;
            detailHTML += `<p><small>Model: MobileNetV2 | Processing: Client-side | 
                Anomaly Score: ${imgAnalysis.ml_analysis.anomaly_score}</small></p>`;
        } else {
            detailHTML += `<ul>${imgAnalysis.findings.map(f => `<li>${f}</li>`).join('')}</ul>`;
        }
        
        detailHTML += `</div>`;
    }
    
    if (data.predictions.ecg_analysis) {
        detailHTML += `<div class="prediction-item">
            <strong>ECG Analysis:</strong> ${data.predictions.ecg_analysis.rhythm}
            <p>Heart Rate: ${data.predictions.ecg_analysis.heart_rate} bpm</p>
            <ul>${data.predictions.ecg_analysis.abnormalities.map(a => `<li>${a}</li>`).join('')}</ul>
        </div>`;
    }
    
    if (data.predictions.clinical_analysis) {
        detailHTML += `<div class="prediction-item">
            <strong>Clinical Factors:</strong> ${data.predictions.clinical_analysis.risk_factor_count} risk factors identified
            <ul>${data.predictions.clinical_analysis.risk_factors.map(r => `<li>${r}</li>`).join('')}</ul>
        </div>`;
    }
    
    detailDiv.innerHTML = detailHTML;
    
    // Display recommendations
    const recDiv = document.getElementById('recommendations');
    recDiv.innerHTML = '<h4>Recommendations:</h4><ul>' + 
        data.recommendations.map(r => `<li>${r}</li>`).join('') + '</ul>';
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
