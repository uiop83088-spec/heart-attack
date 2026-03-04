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

// Prediction form handler
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const resultsDiv = document.getElementById('results');
    const predictButton = this.querySelector('.predict-button');
    
    predictButton.textContent = 'Analyzing...';
    predictButton.disabled = true;
    
    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            resultsDiv.classList.remove('hidden');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Connection error. Make sure the backend server is running.');
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
        detailHTML += `<div class="prediction-item">
            <strong>Medical Image:</strong> Confidence ${data.predictions.image_analysis.confidence * 100}%
            <ul>${data.predictions.image_analysis.findings.map(f => `<li>${f}</li>`).join('')}</ul>
        </div>`;
    }
    
    if (data.predictions.ecg_analysis) {
        detailHTML += `<div class="prediction-item">
            <strong>ECG Analysis:</strong> ${data.predictions.ecg_analysis.rhythm}
            <p>Heart Rate: ${data.predictions.ecg_analysis.heart_rate} bpm</p>
        </div>`;
    }
    
    if (data.predictions.clinical_analysis) {
        detailHTML += `<div class="prediction-item">
            <strong>Clinical Factors:</strong> ${data.predictions.clinical_analysis.risk_factor_count} risk factors identified
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
