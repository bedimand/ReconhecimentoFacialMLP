import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible (3.7+)"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)

def check_required_packages():
    """Check if required packages are installed"""
    try:
        import cv2
        import numpy
        import torch
        import insightface
        print("Required packages are already installed")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "opencv-python", "numpy", "torch", "insightface", "onnxruntime"])
        print("Required packages installed successfully")

def download_insightface_models():
    """Download InsightFace models for face detection and landmark detection"""
    print("\nDownloading InsightFace models...")
    
    # Create models directory
    models_dir = Path("models/insightface")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import insightface to trigger model download
        import insightface
        from insightface.app import FaceAnalysis
        
        # Initialize FaceAnalysis to download models
        app = FaceAnalysis(name="buffalo_s", root=str(models_dir), 
                          allowed_modules=['detection', 'landmark_2d_106'])
        
        # Prepare the model (this will download if not present)
        app.prepare(ctx_id=-1)  # Use CPU for download
        
        print("InsightFace models downloaded successfully")
        
        # List downloaded models
        print("\nDownloaded models:")
        for model_file in models_dir.glob("**/*"):
            if model_file.is_file():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  - {model_file.relative_to(models_dir)} ({size_mb:.2f} MB)")
    
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Please check your internet connection and try again")
        sys.exit(1)

def main():
    """Main function to download all required models"""
    print("=" * 50)
    print("Face Recognition System - Model Downloader")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check and install required packages
    check_required_packages()
    
    # Download InsightFace models
    download_insightface_models()
    
    print("\n" + "=" * 50)
    print("All models downloaded successfully!")
    print("You can now run face_detector.py and real_time_recognition.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
