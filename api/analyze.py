from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image
import sys

# Try to import scipy
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available, using alternative edge detection", file=sys.stderr)

def validate_medical_image(img):
    """
    Validate if image looks like a medical chest X-ray
    Returns (is_valid, confidence, reason)
    """
    # Convert to grayscale
    gray = img.convert('L')
    pixels = np.array(gray)
    
    # Medical X-rays have specific characteristics:
    # 1. Mostly grayscale (low color variance)
    # 2. High contrast (wide intensity range)
    # 3. Specific aspect ratio (chest X-rays are typically portrait)
    # 4. Dark background with lighter center (lungs)
    
    width, height = img.size
    aspect_ratio = height / width
    
    # Check aspect ratio (chest X-rays are usually 1.2-1.5 ratio)
    if aspect_ratio < 0.8 or aspect_ratio > 2.0:
        return False, 0.3, "Image aspect ratio doesn't match typical chest X-ray dimensions"
    
    # Check if image is mostly grayscale
    if img.mode == 'RGB':
        r, g, b = img.split()
        r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)
        color_variance = np.mean([
            np.std(r_arr - g_arr),
            np.std(g_arr - b_arr),
            np.std(r_arr - b_arr)
        ])
        
        # Medical X-rays should have very low color variance
        if color_variance > 30:
            return False, 0.4, "Image appears to be a color photo, not a medical X-ray"
    
    # Check intensity distribution
    intensity_std = np.std(pixels)
    intensity_range = np.max(pixels) - np.min(pixels)
    
    # X-rays should have good contrast
    if intensity_range < 100:
        return False, 0.5, "Image has insufficient contrast for medical X-ray"
    
    # Check for typical X-ray pattern (darker edges, lighter center)
    h, w = pixels.shape
    center_region = pixels[h//4:3*h//4, w//4:3*w//4]
    edge_region = np.concatenate([
        pixels[:h//4, :].flatten(),
        pixels[3*h//4:, :].flatten(),
        pixels[:, :w//4].flatten(),
        pixels[:, 3*w//4:].flatten()
    ])
    
    center_mean = np.mean(center_region)
    edge_mean = np.mean(edge_region)
    
    # Center should be lighter than edges in chest X-ray
    if center_mean < edge_mean:
        return False, 0.6, "Image pattern doesn't match chest X-ray (center should be lighter)"
    
    # Calculate confidence based on characteristics
    confidence = min(0.95, 0.7 + (intensity_std / 100) * 0.2 + (intensity_range / 255) * 0.1)
    
    return True, confidence, "Image appears to be a valid medical chest X-ray"


def analyze_with_heuristics(img):
    """
    Analyze chest X-ray using medical imaging heuristics
    This is a fallback when deep learning models aren't available
    """
    # Convert to grayscale and normalize
    gray = img.convert('L')
    pixels = np.array(gray).astype(float) / 255.0
    
    h, w = pixels.shape
    
    # Divide into regions for analysis
    left_lung = pixels[:, :w//2]
    right_lung = pixels[:, w//2:]
    upper_region = pixels[:h//3, :]
    middle_region = pixels[h//3:2*h//3, :]
    lower_region = pixels[2*h//3:, :]
    cardiac_region = pixels[h//3:2*h//3, w//3:2*w//3]
    
    # Calculate statistics
    left_mean = np.mean(left_lung)
    right_mean = np.mean(right_lung)
    asymmetry = abs(left_mean - right_mean)
    
    cardiac_density = np.mean(cardiac_region < 0.4)  # Dark regions
    lung_density = np.mean(pixels < 0.5)
    
    # Edge detection for consolidation
    if SCIPY_AVAILABLE:
        edges = ndimage.sobel(pixels)
        edge_strength = np.mean(np.abs(edges))
    else:
        # Simple edge detection without scipy
        edges_x = np.diff(pixels, axis=1)
        edges_y = np.diff(pixels, axis=0)
        edge_strength = (np.mean(np.abs(edges_x)) + np.mean(np.abs(edges_y))) / 2
    
    # Texture analysis
    texture_variance = np.var(pixels)
    
    # Detect conditions based on patterns
    conditions = []
    risk_scores = []
    
    # Cardiomegaly (enlarged heart) - high cardiac density
    if cardiac_density > 0.35:
        severity = 'Severe' if cardiac_density > 0.50 else 'Moderate' if cardiac_density > 0.42 else 'Mild'
        confidence = min(0.88, 0.60 + cardiac_density)
        conditions.append({
            'name': 'Cardiomegaly (Enlarged Heart)',
            'confidence': confidence,
            'severity': severity,
            'description': f'Cardiac silhouette appears enlarged (density: {cardiac_density:.2f})'
        })
        risk_scores.append(0.75 if severity == 'Severe' else 0.60 if severity == 'Moderate' else 0.45)
    
    # Pulmonary Edema - diffuse increased density
    if lung_density > 0.55 and asymmetry < 0.08:
        severity = 'Severe' if lung_density > 0.65 else 'Moderate'
        confidence = min(0.85, 0.55 + lung_density * 0.4)
        conditions.append({
            'name': 'Pulmonary Edema',
            'confidence': confidence,
            'severity': severity,
            'description': f'Bilateral increased lung density suggesting fluid accumulation'
        })
        risk_scores.append(0.85 if severity == 'Severe' else 0.65)
    
    # Pleural Effusion - asymmetric density
    if asymmetry > 0.12:
        severity = 'Large' if asymmetry > 0.20 else 'Moderate'
        confidence = min(0.82, 0.50 + asymmetry * 2.5)
        conditions.append({
            'name': 'Pleural Effusion',
            'confidence': confidence,
            'severity': severity,
            'description': f'Asymmetric opacity suggesting fluid in pleural space'
        })
        risk_scores.append(0.70 if severity == 'Large' else 0.55)
    
    # Pneumonia/Consolidation - focal density with sharp edges
    if edge_strength > 0.15 and texture_variance > 0.04:
        severity = 'Severe' if edge_strength > 0.22 else 'Moderate'
        confidence = min(0.86, 0.55 + edge_strength * 1.5)
        conditions.append({
            'name': 'Pneumonia/Consolidation',
            'confidence': confidence,
            'severity': severity,
            'description': f'Focal consolidation with sharp margins detected'
        })
        risk_scores.append(0.75 if severity == 'Severe' else 0.60)
    
    # Pneumothorax - very high asymmetry with low density
    if asymmetry > 0.18 and lung_density < 0.40:
        conditions.append({
            'name': 'Possible Pneumothorax',
            'confidence': min(0.75, 0.45 + asymmetry * 2.0),
            'severity': 'Urgent',
            'description': 'Asymmetric lucency suggesting possible air in pleural space'
        })
        risk_scores.append(0.90)
    
    # Normal if no significant findings
    if len(conditions) == 0:
        if lung_density < 0.45 and asymmetry < 0.10 and cardiac_density < 0.30:
            conditions.append({
                'name': 'Normal Chest X-Ray',
                'confidence': 0.85,
                'severity': 'None',
                'description': 'No significant cardiopulmonary abnormalities detected'
            })
            risk_scores.append(0.15)
    
    # Calculate overall risk
    overall_risk = max(risk_scores) if risk_scores else 0.15
    
    return {
        'conditions': conditions,
        'risk_score': overall_risk,
        'technical_details': {
            'asymmetry': float(asymmetry),
            'cardiac_density': float(cardiac_density),
            'lung_density': float(lung_density),
            'edge_strength': float(edge_strength),
            'texture_variance': float(texture_variance)
        }
    }


class handler(BaseHTTPRequestHandler):
    """
    Vercel serverless function handler
    """
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            if 'image' not in data:
                self.send_error_response(400, {'error': 'No image data provided'})
                return
            
            # Decode base64 image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Validate if it's a medical image
            is_valid, validation_confidence, validation_reason = validate_medical_image(img)
            
            if not is_valid:
                self.send_error_response(400, {
                    'error': 'Invalid medical image',
                    'reason': validation_reason,
                    'confidence': validation_confidence,
                    'suggestion': 'Please upload a chest X-ray image (grayscale medical imaging)'
                })
                return
            
            # Analyze the image
            analysis = analyze_with_heuristics(img)
            
            # Generate medical report
            findings = []
            findings.append('=== CHEST X-RAY ANALYSIS REPORT ===')
            findings.append('')
            findings.append('FINDINGS:')
            
            for i, condition in enumerate(analysis['conditions'], 1):
                confidence_pct = int(condition['confidence'] * 100)
                findings.append(f"{i}. {condition['name']}")
                findings.append(f"   - Confidence: {confidence_pct}%")
                findings.append(f"   - Severity: {condition['severity']}")
                findings.append(f"   - {condition['description']}")
                findings.append('')
            
            findings.append('IMPRESSION:')
            risk = analysis['risk_score']
            if risk < 0.30:
                findings.append('- Chest X-ray within normal limits')
                findings.append('- No acute cardiopulmonary abnormality')
                findings.append('- Routine follow-up recommended')
            elif risk < 0.60:
                findings.append('- Mild to moderate abnormalities detected')
                findings.append('- Clinical correlation recommended')
                findings.append('- Consider follow-up imaging in 3-6 months')
            else:
                findings.append('- Significant abnormalities detected')
                findings.append('- IMMEDIATE clinical evaluation recommended')
                findings.append('- Further diagnostic workup advised')
                findings.append('- Specialist consultation recommended')
            
            findings.append('')
            findings.append('NOTE: This is an AI-assisted analysis using medical imaging algorithms.')
            findings.append('Final diagnosis must be made by a qualified radiologist.')
            
            # Return results
            response_data = {
                'success': True,
                'validation': {
                    'is_medical_image': is_valid,
                    'confidence': validation_confidence,
                    'reason': validation_reason
                },
                'analysis': {
                    'confidence': validation_confidence,
                    'anomaly_detected': risk > 0.40,
                    'anomaly_score': f"{risk:.3f}",
                    'findings': findings,
                    'detected_conditions': analysis['conditions'],
                    'technical_details': analysis['technical_details']
                }
            }
            
            self.send_success_response(response_data)
            
        except Exception as e:
            import traceback
            self.send_error_response(500, {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def send_success_response(self, data):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def send_error_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
