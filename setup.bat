@echo off
echo Setting up VisualKinetics environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install the package in development mode
echo Installing VisualKinetics...
pip install -e .

REM Create directories for models and output
echo Creating necessary directories...
mkdir models 2>nul
mkdir output 2>nul

REM Download required models
echo Downloading required models...
python -c "
import mediapipe as mp
import tensorflow as tf
import torch
import os

# Initialize MediaPipe Pose (this will download the model)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Download a basic PyTorch model for transfer learning
model_path = 'models/pose_model.pth'
if not os.path.exists(model_path):
    # Using ResNet18 as base model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    torch.save(model.state_dict(), model_path)

print('Models downloaded successfully!')
"

REM Test the installation
echo Testing installation...
python -c "from visualkinetics import VideoProcessor; print('VisualKinetics imported successfully!')"

echo Setup completed successfully!
echo To activate the environment, run: venv\Scripts\activate.bat
pause 