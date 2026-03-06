# Testing Guide - Medical Image Analysis

## What Was Fixed

The system now:
1. ✅ **Validates medical images** - Rejects faces, color photos, non-medical images
2. ✅ **Proper risk scoring** - Unhealthy X-rays show HIGH risk (60-90%), not low risk
3. ✅ **Real medical algorithms** - Uses medical imaging principles, not generic ML

## How to Test

### Test 1: Valid Chest X-Ray (Healthy)
**Expected Result**: 
- ✅ Image accepted
- Risk: 15-30% (Low Risk)
- Findings: "Normal Chest X-Ray" or minimal abnormalities

### Test 2: Valid Chest X-Ray (Unhealthy with visible pathology)
**Expected Result**:
- ✅ Image accepted
- Risk: 60-90% (High Risk)
- Findings: Specific conditions detected (Cardiomegaly, Pulmonary Edema, etc.)
- Recommendations: "Immediate clinical evaluation strongly recommended"

### Test 3: Face Image / Color Photo
**Expected Result**:
- ❌ Image REJECTED
- Error: "Image appears to be a color photo, not a medical X-ray"
- Suggestion: "Please upload a chest X-ray image (grayscale medical imaging)"

### Test 4: Landscape Photo / Wrong Aspect Ratio
**Expected Result**:
- ❌ Image REJECTED
- Error: "Image aspect ratio doesn't match typical chest X-ray dimensions"

### Test 5: Low Contrast Image
**Expected Result**:
- ❌ Image REJECTED
- Error: "Image has insufficient contrast for medical X-ray"

## Where to Find Test Images

### Chest X-Rays (for testing)
You can find sample chest X-rays at:
- Google Images: "chest x-ray normal" or "chest x-ray pneumonia"
- Medical databases: NIH Chest X-ray dataset (public)
- Your uploads folder already has some test images

### Non-Medical Images (should be rejected)
- Any face photo
- Color photos
- Landscape images
- Screenshots

## What to Look For

### ✅ Good Signs
1. Face images are rejected immediately
2. Unhealthy X-rays show risk > 60%
3. Specific conditions are named (not just "abnormal")
4. Technical details show actual measurements (asymmetry, density, etc.)

### ❌ Bad Signs (if you see these, something is wrong)
1. Face images are accepted and analyzed
2. Clearly abnormal X-rays show risk < 30%
3. All images show the same risk score
4. No specific conditions detected

## Vercel Deployment

After pushing to GitHub:
1. Vercel will automatically deploy
2. Wait 2-3 minutes for build to complete
3. Check deployment logs for any errors
4. Test on live site

### Common Deployment Issues

**Issue**: Python dependencies fail to install
**Solution**: Check `requirements.txt` has correct versions

**Issue**: API endpoint not found
**Solution**: Check `vercel.json` routes configuration

**Issue**: CORS errors
**Solution**: Check `Access-Control-Allow-Origin` headers in `api/analyze.py`

## API Testing (Advanced)

You can test the API directly:

```bash
# Convert image to base64
base64 -i chest_xray.jpg -o image.txt

# Test API
curl -X POST https://your-app.vercel.app/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,<base64_string>"}'
```

## Expected Response Format

```json
{
  "success": true,
  "validation": {
    "is_medical_image": true,
    "confidence": 0.85,
    "reason": "Image appears to be a valid medical chest X-ray"
  },
  "analysis": {
    "confidence": 0.85,
    "anomaly_detected": true,
    "anomaly_score": "0.750",
    "findings": ["=== CHEST X-RAY ANALYSIS REPORT ===", ...],
    "detected_conditions": [
      {
        "name": "Cardiomegaly (Enlarged Heart)",
        "confidence": 0.75,
        "severity": "Moderate",
        "description": "Cardiac silhouette appears enlarged"
      }
    ],
    "technical_details": {
      "asymmetry": 0.15,
      "cardiac_density": 0.42,
      "lung_density": 0.38,
      "edge_strength": 0.18,
      "texture_variance": 0.05
    }
  }
}
```

## Troubleshooting

### "Model failed to load"
- Old error from TensorFlow.js - should not appear anymore
- If you see this, clear browser cache

### "Failed to fetch"
- Check internet connection
- Check Vercel deployment status
- Check browser console for CORS errors

### All images show same risk
- Check if Python backend is running
- Check Vercel logs for Python errors
- Verify `api/analyze.py` is deployed

## Success Criteria

The fix is successful if:
1. ✅ Face images are rejected with clear error message
2. ✅ Unhealthy X-rays show risk > 60% with specific conditions
3. ✅ Healthy X-rays show risk < 30%
4. ✅ Technical details show actual measurements (not just generic features)
5. ✅ Error messages are helpful and specific

## Next Steps

If everything works:
1. Test with multiple chest X-ray images
2. Document any edge cases
3. Consider training custom model for even better accuracy
4. Add more validation rules if needed

If issues persist:
1. Check Vercel deployment logs
2. Test API endpoint directly
3. Verify Python dependencies installed correctly
4. Check browser console for JavaScript errors
