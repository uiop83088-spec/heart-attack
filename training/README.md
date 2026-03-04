# DeepHealthX Model Training

Train custom deep learning models for heart disease detection.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
```
data/
├── medical_images/
│   ├── normal/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── abnormal/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── ecg_signals/
    ├── signals.npy
    └── labels.npy
```

## Available Datasets

### Medical Imaging Datasets:
1. **Chest X-Ray Dataset** (Kaggle)
   - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   
2. **NIH Chest X-rays** (NIH Clinical Center)
   - https://nihcc.app.box.com/v/ChestXray-NIHCC
   
3. **MIMIC-CXR** (MIT)
   - https://physionet.org/content/mimic-cxr/2.0.0/

### ECG Datasets:
1. **MIT-BIH Arrhythmia Database**
   - https://physionet.org/content/mitdb/1.0.0/
   
2. **PTB Diagnostic ECG Database**
   - https://physionet.org/content/ptbdb/1.0.0/
   
3. **ECG Heartbeat Categorization** (Kaggle)
   - https://www.kaggle.com/datasets/shayanfazeli/heartbeat

## Training Models

### 1. Train Image Classification Model

```python
from train_model import HeartDiseaseModelTrainer

# Initialize trainer
trainer = HeartDiseaseModelTrainer(image_size=224, num_classes=2)

# Create model (choose architecture)
trainer.model = trainer.create_cnn_model(architecture='mobilenet')
# Options: 'mobilenet', 'resnet', 'custom'

# Prepare data
train_gen, val_gen = trainer.prepare_data('data/medical_images')

# Train
history = trainer.train(train_gen, val_gen, epochs=50)

# Save model
trainer.save_model('models/heart_disease_cnn.h5')

# Export for Node.js
trainer.export_for_nodejs('models/heart_disease_cnn.h5')
```

### 2. Train ECG Model

```python
from train_model import ECGModelTrainer
import numpy as np

# Initialize trainer
ecg_trainer = ECGModelTrainer(sequence_length=1000)

# Create model
ecg_trainer.model = ecg_trainer.create_cnn_lstm_model()

# Load data
signals = np.load('data/ecg_signals/signals.npy')
labels = np.load('data/ecg_signals/labels.npy')

# Prepare data
X_train, y_train, X_val, y_val = ecg_trainer.prepare_ecg_data(signals, labels)

# Train
history = ecg_trainer.train(X_train, y_train, X_val, y_val, epochs=100)

# Save
ecg_trainer.model.save('models/ecg_lstm.h5')
```

## Model Architectures

### Image Models:

1. **MobileNetV2** (Recommended for production)
   - Fast inference
   - Small model size (~14MB)
   - Good accuracy (85-90%)

2. **ResNet50**
   - Higher accuracy (90-95%)
   - Larger model size (~98MB)
   - Slower inference

3. **Custom CNN**
   - Lightweight
   - Customizable
   - Good for specific use cases

### ECG Models:

1. **LSTM**
   - Good for temporal patterns
   - Handles variable-length sequences

2. **CNN-LSTM Hybrid** (Recommended)
   - Best accuracy
   - Combines spatial and temporal features
   - Robust to noise

## Integration with Node.js

After training, export models for Node.js:

```python
trainer.export_for_nodejs('models/heart_disease_cnn.h5', 'models/nodejs')
```

This creates:
- `model.json` - Model architecture
- `group1-shard1of1.bin` - Model weights
- `metadata.json` - Preprocessing info

Update `ml-models.js` to load the trained model:

```javascript
const tf = require('@tensorflow/tfjs-node');

async function loadModel() {
    const model = await tf.loadLayersModel('file://./models/nodejs/model.json');
    return model;
}
```

## Performance Metrics

Track these metrics during training:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision and recall

## Tips for Better Models

1. **Data Augmentation**: Increase training data variety
2. **Transfer Learning**: Use pre-trained models
3. **Regularization**: Prevent overfitting (Dropout, L2)
4. **Learning Rate Scheduling**: Adjust learning rate during training
5. **Early Stopping**: Stop when validation loss stops improving
6. **Ensemble Methods**: Combine multiple models

## Deployment

Once trained, models can be:
1. Deployed to Node.js backend (current setup)
2. Converted to TensorFlow Lite for mobile
3. Deployed to cloud (AWS SageMaker, Google AI Platform)
4. Used in browser with TensorFlow.js

## License

MIT License - See LICENSE file
