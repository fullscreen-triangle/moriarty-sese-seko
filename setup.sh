#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up VisualKinetics environment...${NC}"

# Use Python 3.10 explicitly
PYTHON_CMD="/opt/local/bin/python3.10"

# Check if Python 3.10 is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python 3.10 is not found at $PYTHON_CMD. Please ensure it is installed."
    exit 1
fi

# Create and activate virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv venv

# Determine the correct activate script based on OS
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    echo "Unsupported operating system"
    exit 1
fi

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install the package in development mode
echo -e "${BLUE}Installing VisualKinetics...${NC}"
pip install -e .

# Create directories for models and output
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p models output

# Download required models
echo -e "${BLUE}Downloading required models...${NC}"
python - << EOF
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
model_path = "models/pose_model.pth"
if not os.path.exists(model_path):
    # Using ResNet18 as base model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    torch.save(model.state_dict(), model_path)

print("Models downloaded successfully!")
EOF

# Test the installation
echo -e "${BLUE}Testing installation...${NC}"
python - << EOF
from visualkinetics import VideoProcessor
print("VisualKinetics imported successfully!")
EOF

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}To activate the environment, run: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)${NC}" 