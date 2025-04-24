# Pose Module Documentation

This document provides a comprehensive overview of the pose module located in `src/core/pose`. The module focuses on human pose detection, estimation, and visualization.

## Table of Contents

1. [Overview](#overview)
2. [Module Components](#module-components)
3. [File Descriptions](#file-descriptions)
4. [Key Classes and Functions](#key-classes-and-functions)
5. [Usage Examples](#usage-examples)

## Overview

The pose module provides functionality for human detection, pose estimation, keypoint tracking, and visualization. It is designed to work with images and video data, allowing for real-time pose analysis and visualization. The module integrates with Ray for distributed processing to improve performance.

## Module Components

The module consists of several components:

- **Human Detection**: Detecting human figures in images/videos
- **Pose Estimation**: Estimating body keypoints and skeletal structure
- **Keypoint Tracking**: Tracking keypoints over time with filtering
- **Pose Visualization**: Visualizing poses in 2D and 3D
- **Distance Calculation**: Calculating distances between detected people
- **Angle Calculation**: Calculating joint angles for biomechanical analysis

## File Descriptions

### `pose.py`

Contains the core `DistributedPose` class for representing human poses with keypoints, along with utility functions for pose tracking and similarity comparison. The class handles keypoint representation, bounding box calculation, and rendering.

### `pose_detector.py`

Implements the `PoseDetector` class which handles the detection of human poses in images. It includes functions for model loading, inference, and returning keypoint coordinates with confidence scores.

### `human_detector.py`

Contains the `HumanDetector` class for detecting humans in images using a pre-trained Faster R-CNN model. It handles bounding box detection, calibration for distance measurements, and visualization of detections.

### `keypoints.py`

Provides functions for keypoint extraction, processing, and grouping from model outputs. It handles the identification of body part connections and filtering of pose entries.

### `one_euro_filter.py`

Implements the One Euro Filter algorithm for smoothing keypoint movements over time. This helps reduce jitter in pose tracking while maintaining responsiveness.

### `pose_visualizer.py`

Contains the `PoseVisualizer` class for generating 2D and 3D visualizations of detected poses. It supports rendering joint connections, calculating angles, and generating plots.

### `skeleton.py`

Implements the `SkeletonDrawer` class for rendering skeletal structures based on detected keypoints. It handles joint pair connections, angle calculations, and visualization.

### `athlete_detection.py`

Focuses on athlete-specific detection and tracking, providing specialized functionality for sports applications.

### `pose_data_to_llm.py`

Handles the conversion of pose data to formats suitable for large language models, enabling natural language understanding of pose information.

### `load_state.py`

Provides utilities for loading model states and checkpoints.

### `loss.py`

Contains loss functions used for training pose detection models.

### `get_parameters.py`

Utilities for retrieving model parameters and configuration settings.

### `conv.py`

Implements convolution operations specialized for pose detection networks.

## Key Classes and Functions

### `DistributedPose` (in `pose.py`)

The core class representing a human pose with keypoints.

```python
class DistributedPose:
    num_kpts = 18
    kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                'r_eye', 'l_eye', 'r_ear', 'l_ear']
    
    def __init__(self, keypoints, confidence):
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = self.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(self.num_kpts)]
```

### `PoseDetector` (in `pose_detector.py`)

Handles the detection of poses in frames.

```python
class PoseDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        # Initialize detector
        
    def detect(self, frame: np.ndarray):
        # Detect poses in frame
```

### `HumanDetector` (in `human_detector.py`)

Detects humans in frames and calculates distances between them.

```python
class HumanDetector:
    def __init__(self, confidence_threshold: float = 0.5, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # Initialize detector
        
    def detect_humans(self, frame: np.ndarray) -> Tuple[List[DetectedPerson], Dict[Tuple[int, int], float]]:
        # Detect humans and return detections with distances
```

### `SkeletonDrawer` (in `skeleton.py`)

Draws skeletal structures and calculates joint angles.

```python
class SkeletonDrawer:
    def __init__(self):
        # Set up joint pairs
        
    def draw_skeleton(self, frame, detection):
        # Draw skeleton on frame
```

### `OneEuroFilter` (in `one_euro_filter.py`)

Implements the One Euro Filter algorithm for smoothing keypoint movements.

```python
class DistributedOneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):
        # Initialize filter
        
    def __call__(self, x):
        # Apply filter to new value
```

### `PoseVisualizer` (in `pose_visualizer.py`)

Visualizes poses in 2D and 3D.

```python
class PoseVisualizer:
    def __init__(self, distilled_model_path: str = "./distilled_model/final"):
        # Initialize visualizer
        
    def query_to_visualization(self, query: str, output_path: str = None) -> Dict[str, Any]:
        # Convert query to visualization
```

## Usage Examples

### Basic Pose Detection

```python
from src.core.pose.pose_detector import PoseDetector

# Initialize detector
detector = PoseDetector(model_path="path/to/model", confidence_threshold=0.5)

# Detect poses in frame
keypoints = detector.detect(frame)
```

### Human Detection with Distance Calculation

```python
from src.core.pose.human_detector import HumanDetector

# Initialize detector
detector = HumanDetector(confidence_threshold=0.5)

# Calibrate for distance measurement
detector.calibrate(known_distance_pixels=100, known_distance_meters=1.0)

# Detect humans and measure distances
detections, distances = detector.detect_humans(frame)

# Visualize detections
output_frame = detector.draw_detections(frame, detections, distances)
```

### Pose Visualization

```python
from src.core.pose.pose_visualizer import PoseVisualizer

# Initialize visualizer
visualizer = PoseVisualizer(distilled_model_path="path/to/model")

# Generate visualization from query
visualization_data = visualizer.query_to_visualization(
    query="A person standing with arms raised",
    output_path="output/visualization.json"
)
```

### Keypoint Tracking with Filtering

```python
from src.core.pose.pose import DistributedPose
from src.core.pose.one_euro_filter import OneEuroFilter

# Create pose from keypoints
pose = DistributedPose(keypoints, confidence)

# Apply filtering for smoothed tracking
filtered_poses = track_poses(previous_poses, current_poses, smooth=True)
```
