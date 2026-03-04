"""
Download and prepare medical imaging datasets for training
"""

import os
import urllib.request
import zipfile
import kaggle

def download_chest_xray_dataset():
    """Download Chest X-Ray dataset from Kaggle"""
    print("Downloading Chest X-Ray dataset from Kaggle...")
    print("Note: You need Kaggle API credentials (~/.kaggle/kaggle.json)")
    
    try:
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='data/raw',
            unzip=True
        )
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use Kaggle API:")
        print("1. Create account at kaggle.com")
        print("2. Go to Account > API > Create New API Token")
        print("3. Place kaggle.json in ~/.kaggle/")

def download_ecg_dataset():
    """Download ECG dataset from Kaggle"""
    print("Downloading ECG Heartbeat dataset from Kaggle...")
    
    try:
        kaggle.api.dataset_download_files(
            'shayanfazeli/heartbeat',
            path='data/raw/ecg',
            unzip=True
        )
        print("ECG dataset downloaded successfully!")
    except Exception as e:
        print(f"Error: {e}")

def prepare_directory_structure():
    """Create directory structure for training"""
    directories = [
        'data/raw',
        'data/medical_images/normal',
        'data/medical_images/abnormal',
        'data/ecg_signals',
        'models',
        'models/nodejs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

if __name__ == "__main__":
    print("DeepHealthX Dataset Preparation")
    print("=" * 50)
    
    # Create directories
    prepare_directory_structure()
    
    # Download datasets
    print("\nDownloading datasets...")
    print("This may take a while depending on your internet connection.")
    
    choice = input("\nDownload Chest X-Ray dataset? (y/n): ")
    if choice.lower() == 'y':
        download_chest_xray_dataset()
    
    choice = input("\nDownload ECG dataset? (y/n): ")
    if choice.lower() == 'y':
        download_ecg_dataset()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("Next steps:")
    print("1. Organize images into data/medical_images/normal and abnormal folders")
    print("2. Run: python train_model.py")
