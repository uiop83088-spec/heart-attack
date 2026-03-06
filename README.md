# DeepHealthX - AI-Powered Heart Disease Detection

Multi-Modal Deep Learning system for early detection of heart diseases using medical chest X-ray analysis.

## 🚀 Features

- **Real Medical Image Analysis**: Uses medical imaging algorithms to analyze chest X-rays
- **Image Validation**: Automatically validates if uploaded images are medical chest X-rays
- **Pathology Detection**: Detects conditions like:
  - Cardiomegaly (Enlarged Heart)
  - Pulmonary Edema
  - Pleural Effusion
  - Pneumonia/Consolidation
  - Pneumothorax
- **Risk Assessment**: Provides risk scores and clinical recommendations
- **Serverless Architecture**: Runs on Vercel with Python backend

## 🏗️ Architecture

### Frontend
- HTML/CSS/JavaScript
- Client-side image validation
- Real-time results display

### Backend
- **Python Serverless Function** (`api/analyze.py`)
  - Medical image validation
  - Heuristic-based pathology detection
  - Uses PIL, NumPy, SciPy for image analysis
- **Node.js Server** (`server.js`)
  - File upload handling
  - Static file serving

## 🔬 How It Works

1. **Image Upload**: User uploads a chest X-ray image
2. **Client Validation**: Basic checks (size, format, dimensions)
3. **Server Validation**: Validates if image is a medical chest X-ray by checking:
   - Aspect ratio (chest X-rays are portrait)
   - Grayscale characteristics
   - Contrast and intensity distribution
   - Center-to-edge brightness pattern
4. **Medical Analysis**: Analyzes image using medical imaging algorithms:
   - Asymmetry detection (left vs right lung)
   - Cardiac density analysis
   - Lung density patterns
   - Edge detection for consolidation
   - Texture analysis
5. **Pathology Detection**: Identifies specific conditions based on patterns
6. **Risk Scoring**: Calculates overall risk and provides recommendations

## 🛠️ Technology Stack

- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Backend**: Python 3.x (Vercel Serverless)
- **Libraries**: 
  - Pillow (PIL) - Image processing
  - NumPy - Numerical computations
  - SciPy - Scientific computing (edge detection)
- **Deployment**: Vercel

## 📦 Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/uiop83088-spec/heart-attack.git
cd heart-attack
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Install Python dependencies (for local testing):
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
npm start
```

5. Open http://localhost:5000

### Deployment to Vercel

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel
```

3. Follow the prompts to deploy

## 🧪 Testing

Upload chest X-ray images to test the analysis. The system will:
- ✅ Accept valid chest X-rays
- ❌ Reject non-medical images (photos, faces, etc.)
- ❌ Reject images with wrong aspect ratios
- ❌ Reject color photos

## 📊 Model Training (Optional)

For custom model training with actual deep learning:

1. Navigate to training directory:
```bash
cd training
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Download datasets:
```bash
python download_datasets.py
```

4. Train model:
```bash
python train_model.py --model mobilenetv2 --epochs 50
```

5. Export to TensorFlow.js format for browser deployment

See `training/README.md` for detailed instructions.

## ⚠️ Important Notes

- This is an AI-assisted analysis tool, NOT a medical diagnostic device
- Always consult qualified healthcare professionals for medical diagnosis
- The system uses heuristic algorithms, not trained deep learning models
- For production use, integrate actual medical imaging models (CheXNet, ChestX-ray14)

## 🔮 Future Improvements

1. **Integrate Real Medical Models**:
   - Use pre-trained CheXNet or ChestX-ray14 models
   - Requires larger server infrastructure (not Vercel serverless)

2. **Multi-Modal Analysis**:
   - Add ECG signal analysis
   - Integrate clinical data (age, symptoms, vitals)

3. **Enhanced Validation**:
   - Use ML-based image classification to validate medical images
   - Detect image orientation and auto-correct

4. **DICOM Support**:
   - Support medical DICOM format
   - Extract metadata from DICOM files

## 📝 License

MIT License

## 👥 Contributors

- Your Name

## 🙏 Acknowledgments

- Medical imaging algorithms based on radiological principles
- Inspired by CheXNet and ChestX-ray14 research
