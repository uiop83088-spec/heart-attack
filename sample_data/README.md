# Sample Data for Testing DeepHealthX

## Medical Images

### Where to Get Sample Medical Images:

1. **Free Medical Image Databases:**
   - NIH Chest X-Ray: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Kaggle Chest X-Ray: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   - OpenI: https://openi.nlm.nih.gov/

2. **For Testing (Use Any Image):**
   - You can upload ANY image to test the system
   - The AI will analyze it, but results are only meaningful for actual medical images
   - Try uploading a photo of your chest area (clothed) for demo purposes

### Recommended Test Images:
- Download a chest X-ray from the links above
- Or use the sample images in this folder

## ECG Data

### Sample ECG Files Included:

1. **normal_ecg.csv** - Normal heart rhythm
2. **abnormal_ecg.csv** - Irregular rhythm
3. **tachycardia_ecg.csv** - Fast heart rate

### ECG File Format:

ECG files should be CSV or TXT with comma or space-separated values:

```
0.5, 0.6, 0.7, 0.9, 1.2, 0.8, 0.5
0.6, 0.7, 0.8, 1.0, 1.3, 0.9, 0.6
```

Each number represents the electrical signal at a point in time.

### Where to Get Real ECG Data:

1. **PhysioNet:** https://physionet.org/
   - MIT-BIH Arrhythmia Database
   - PTB Diagnostic ECG Database

2. **Kaggle:** https://www.kaggle.com/datasets/shayanfazeli/heartbeat

## Clinical Information

Fill in the form with:
- **Age:** Your age in years (e.g., 45)
- **Gender:** Male or Female
- **Blood Pressure:** Format like "120/80" or "high"
- **Cholesterol:** Number in mg/dL (e.g., 200)

## Quick Test

1. Upload any image as "Medical Image"
2. Upload one of the sample ECG files
3. Fill in clinical info
4. Click "Analyze with AI"

The system will:
- Run MobileNetV2 on the image
- Analyze ECG signal patterns
- Calculate risk based on clinical factors
- Provide recommendations
