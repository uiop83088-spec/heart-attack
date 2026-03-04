"""
DeepHealthX - Medical Image Model Training
Train custom CNN models for heart disease detection from medical images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json

class HeartDiseaseModelTrainer:
    def __init__(self, image_size=224, num_classes=2):
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = None
        
    def create_cnn_model(self, architecture='mobilenet'):
        """
        Create CNN model for heart disease detection
        Architectures: 'mobilenet', 'resnet', 'custom'
        """
        if architecture == 'mobilenet':
            return self._create_mobilenet_model()
        elif architecture == 'resnet':
            return self._create_resnet_model()
        else:
            return self._create_custom_cnn()
    
    def _create_mobilenet_model(self):
        """Transfer learning with MobileNetV2"""
        base_model = MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self):
        """Transfer learning with ResNet50"""
        base_model = ResNet50(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_custom_cnn(self):
        """Custom CNN architecture for medical imaging"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.image_size, self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def prepare_data(self, data_dir, validation_split=0.2, batch_size=32):
        """
        Prepare data generators for training
        Expected directory structure:
        data_dir/
            normal/
                image1.jpg
                image2.jpg
            abnormal/
                image1.jpg
                image2.jpg
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50, 
              learning_rate=0.001, save_path='models/heart_disease_model.h5'):
        """Train the model"""
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate model performance"""
        results = self.model.evaluate(test_generator)
        metrics = dict(zip(self.model.metrics_names, results))
        return metrics
    
    def save_model(self, path='models/heart_disease_model.h5'):
        """Save trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
        
        # Also save as TensorFlow.js format
        tfjs_path = path.replace('.h5', '_tfjs')
        os.system(f'tensorflowjs_converter --input_format keras {path} {tfjs_path}')
        print(f"TensorFlow.js model saved to {tfjs_path}")
    
    def export_for_nodejs(self, model_path, output_dir='models/nodejs'):
        """Export model in format compatible with Node.js"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to TensorFlow.js format
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(self.model, output_dir)
        
        # Save metadata
        metadata = {
            'image_size': self.image_size,
            'num_classes': self.num_classes,
            'class_names': ['normal', 'abnormal'],
            'preprocessing': {
                'rescale': 1./255,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model exported for Node.js to {output_dir}")


class ECGModelTrainer:
    """Train LSTM/RNN models for ECG signal analysis"""
    
    def __init__(self, sequence_length=1000, num_features=1):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
    
    def create_lstm_model(self):
        """Create LSTM model for ECG classification"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequence_length, self.num_features)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(5, activation='softmax')  # 5 classes: Normal, AFib, etc.
        ])
        
        return model
    
    def create_cnn_lstm_model(self):
        """Hybrid CNN-LSTM for ECG analysis"""
        model = models.Sequential([
            # CNN layers for feature extraction
            layers.Conv1D(64, 3, activation='relu', 
                         input_shape=(self.sequence_length, self.num_features)),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.MaxPooling1D(2),
            
            # LSTM layers for temporal patterns
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(5, activation='softmax')
        ])
        
        return model
    
    def prepare_ecg_data(self, ecg_signals, labels, validation_split=0.2):
        """Prepare ECG data for training"""
        # Normalize signals
        ecg_signals = (ecg_signals - np.mean(ecg_signals)) / np.std(ecg_signals)
        
        # Reshape for LSTM input
        ecg_signals = ecg_signals.reshape(-1, self.sequence_length, self.num_features)
        
        # Split data
        split_idx = int(len(ecg_signals) * (1 - validation_split))
        
        X_train = ecg_signals[:split_idx]
        y_train = labels[:split_idx]
        X_val = ecg_signals[split_idx:]
        y_val = labels[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train ECG model"""
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history


# Example usage
if __name__ == "__main__":
    print("DeepHealthX Model Training")
    print("=" * 50)
    
    # Train Image Model
    print("\n1. Training Medical Image Model...")
    image_trainer = HeartDiseaseModelTrainer(image_size=224, num_classes=2)
    image_trainer.model = image_trainer.create_cnn_model(architecture='mobilenet')
    
    # Uncomment when you have data:
    # train_gen, val_gen = image_trainer.prepare_data('data/medical_images')
    # history = image_trainer.train(train_gen, val_gen, epochs=50)
    # image_trainer.save_model('models/heart_disease_cnn.h5')
    # image_trainer.export_for_nodejs('models/heart_disease_cnn.h5')
    
    print("Image model architecture created!")
    print(image_trainer.model.summary())
    
    # Train ECG Model
    print("\n2. Training ECG Model...")
    ecg_trainer = ECGModelTrainer(sequence_length=1000)
    ecg_trainer.model = ecg_trainer.create_cnn_lstm_model()
    
    print("ECG model architecture created!")
    print(ecg_trainer.model.summary())
    
    print("\n" + "=" * 50)
    print("To train models, prepare your data and uncomment training code")
    print("Data structure:")
    print("  data/medical_images/")
    print("    normal/")
    print("    abnormal/")
    print("  data/ecg_signals/")
    print("    signals.npy")
    print("    labels.npy")
