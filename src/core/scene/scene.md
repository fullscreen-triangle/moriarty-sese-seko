# Scene Module Documentation

This document provides a comprehensive overview of the scene module located in `src/core/scene`. The module focuses on video processing, scene detection, pose analysis, and metrics calculation for athletic performance analysis.

## Table of Contents

1. [Overview](#overview)
2. [Module Components](#module-components)
3. [File Descriptions](#file-descriptions)
4. [Key Classes and Functions](#key-classes-and-functions)
5. [Usage Examples](#usage-examples)

## Overview

The scene module provides functionality for processing and analyzing video content, with a focus on athletic performance analysis. It includes tools for scene detection, pose analysis, metrics calculation, video quality assessment, and data extraction. The module integrates with Ray for distributed processing to improve performance on large videos.

## Module Components

The module consists of several components:

- **Video Processing**: Processing video files for analysis
- **Scene Detection**: Identifying scene changes and transitions in videos
- **Pose Analysis**: Analyzing human poses in video frames
- **Metrics Calculation**: Computing performance metrics based on pose data
- **Data Extraction**: Extracting structured data from processed videos
- **Video Quality Assessment**: Evaluating video quality metrics
- **Video Reconstruction**: Reconstructing missing video segments

## File Descriptions

### `processor.py`

Contains the `VideoProcessor` class for processing videos using parallel execution with Ray. It handles frame extraction, batch processing, pose detection, and annotation drawing.

### `analyzer.py`

Implements the `PoseAnalyzer` class for analyzing pose landmarks and calculating various metrics like joint angles, ground contact, and body metrics.

### `metrics.py`

Contains the `MetricsCalculator` class for computing various performance metrics from pose data, including velocities, accelerations, and distances between athletes.

### `scene_detector.py`

Implements the `SceneDetector` class for detecting scene changes in videos using techniques like histogram comparison, optical flow analysis, and edge detection.

### `analyze_video.py`

Provides functions for the complete analysis pipeline, from video processing to model analysis using AI APIs.

### `data_extractor.py`

Contains the `VideoDataExtractor` class for extracting pose data and metrics from processed videos and storing them in a structured format.

### `process_videos.py`

A utility script for batch processing multiple videos with the video processor.

### `process_one_video.py`

A utility script for processing a single video with the video processor.

### `video_quality.py`

Implements the `VideoQualityAnalyzer` class for analyzing video quality metrics such as brightness, contrast, sharpness, noise, and saturation.

### `video_reconstructor.py`

Contains the `VideoReconstructor` class for reconstructing missing video segments using metrics profiles and multi-angle data.

### `video_manager.py`

Implements the `VideoFrameManager` class for managing video frames, including storage, compression, and retrieval.

## Key Classes and Functions

### `VideoProcessor` (in `processor.py`)

The main class for video processing that handles frame extraction, pose detection, and visualization.

```python
class VideoProcessor:
    def __init__(self, model_path=None, n_workers=None):
        """Initialize the video processor with MediaPipe pose model."""
        
    def process_video(self, video_path):
        """Process video and save annotated output."""
        
    def _prepare_frame_batches(self, cap, total_frames) -> List[List[np.ndarray]]:
        """Prepare batches of frames for parallel processing."""
        
    def _process_frame_batches(self, frame_batches: List[List[np.ndarray]]) -> List[Tuple[np.ndarray, Dict]]:
        """Process batches of frames in parallel using Ray."""
        
    @ray.remote
    def _process_batch(self, analyzer, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, Dict]]:
        """Process a batch of frames using a distributed analyzer."""
        
    def _draw_annotations(self, frame, results, metrics):
        """Draw all annotations on the frame."""
```

### `PoseAnalyzer` (in `analyzer.py`)

Analyzes pose landmarks and calculates various metrics.

```python
class PoseAnalyzer:
    def __init__(self):
        """Initialize the pose analyzer with necessary parameters."""
        
    def analyze_pose(self, landmarks) -> Dict:
        """Analyze pose landmarks and return comprehensive analysis."""
        
    def _landmarks_to_numpy(self, landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array."""
        
    def _calculate_joint_angles(self, points) -> JointAngles:
        """Calculate angles between joints."""
        
    def _detect_ground_contact(self, points) -> Dict[str, bool]:
        """Detect if feet are in contact with the ground."""
        
    def _calculate_body_metrics(self, points) -> Dict:
        """Calculate various body metrics."""
        
    def _angle_between_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points in degrees."""
```

### `MetricsCalculator` (in `metrics.py`)

Calculates performance metrics from pose data.

```python
@ray.remote
class MetricsCalculator:
    def __init__(self):
        """Initialize the metrics calculator."""
        
    def calculate_metrics(self, pose_data: Dict) -> Dict:
        """Calculate various metrics from pose data in parallel."""
        
    def _calculate_velocities(self, points: np.ndarray, dt: float) -> Dict[str, float]:
        """Calculate velocities of key body parts in parallel."""
        
    def _calculate_accelerations(self, points: np.ndarray, dt: float) -> Dict[str, float]:
        """Calculate accelerations of key body parts."""
        
    def _calculate_distances(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, float]:
        """Calculate distances between two athletes in parallel."""
        
    def set_fps(self, fps: float):
        """Update the FPS value for velocity calculations."""
```

### `SceneDetector` (in `scene_detector.py`)

Detects scene changes in videos.

```python
class SceneDetector:
    def __init__(self, config: dict):
        """Initialize scene detector with configuration."""
        
    def compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        """Compute difference metrics between consecutive frames."""
        
    def _compute_focus_measure(self, gray_frame: np.ndarray) -> float:
        """Compute focus measure for a frame."""
        
    def detect_scenes(self, frames_generator) -> List[SceneChange]:
        """Detect scene changes from a generator of frames."""
        
    def _analyze_metrics(self, metrics: Dict[str, float], frame_idx: int) -> Optional[SceneChange]:
        """Analyze metrics to detect scene changes."""
        
    def plot_metrics(self, sequence_name: str):
        """Plot scene detection metrics."""
        
    def reset_metrics(self):
        """Reset metrics history."""
```

### `VideoDataExtractor` (in `data_extractor.py`)

Extracts pose data and metrics from processed videos.

```python
class VideoDataExtractor:
    def __init__(self, output_dir="data_store"):
        """Initialize the data extractor with an output directory."""
        
    def extract_video_data(self, video_path, sample_rate=5, reprocess=False):
        """Extract pose data from processed video at the given sample rate."""
        
    def _process_frame(self, frame, frame_idx, pose=None):
        """Process a single frame to extract pose data and metrics."""
        
    def _extract_metrics_from_frame(self, frame):
        """Extract metrics from text annotations on the frame."""
        
    def extract_all_videos(self, video_dir="output", sample_rate=5):
        """Extract data from all processed videos in the given directory."""
```

### `VideoQualityAnalyzer` (in `video_quality.py`)

Analyzes video quality metrics.

```python
class VideoQualityAnalyzer:
    def __init__(self, config: dict):
        """Initialize video quality analyzer with configuration."""
        
    def compute_frame_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for a single frame."""
        
    def update_metrics_history(self, frame_idx: int, metrics: Dict[str, float]):
        """Update metrics history with new frame data."""
        
    def plot_metrics(self, sequence_name: str, output_dir: Optional[str] = None):
        """Plot quality metrics over time."""
```

### `VideoReconstructor` (in `video_reconstructor.py`)

Reconstructs missing video segments.

```python
class VideoReconstructor:
    def __init__(self, config: dict):
        """Initialize the video reconstructor with configuration."""
        
    def reconstruct_gaps(self, video_segments: List[Dict], gaps: List[GapInfo]) -> Tuple[np.ndarray, Dict]:
        """Reconstruct missing segments using available information."""
```

### `VideoFrameManager` (in `video_manager.py`)

Manages video frames for storage and retrieval.

```python
class VideoFrameManager:
    def __init__(self, storage_path: str, target_resolution: tuple, compression_level: int):
        """Initialize with storage path and compression parameters."""
        
    def process_video(self, video_path: str, sequence_name: str, frame_step: int = 1):
        """Process video and save frames to storage."""
        
    def get_frames(self, sequence_name: str):
        """Retrieve frames for a specific sequence."""
```

## Usage Examples

### Basic Video Processing

```python
from src.core.scene.processor import VideoProcessor

# Initialize the processor
processor = VideoProcessor()

# Process a video
output_path = processor.process_video("input_video.mp4")
print(f"Processed video saved to {output_path}")
```

### Scene Detection

```python
from src.core.scene.scene_detector import SceneDetector
import cv2

# Create a scene detector with configuration
config = {
    'scene_detection': {
        'hist_threshold': 0.5,
        'flow_threshold': 0.7,
        'edge_threshold': 0.6
    }
}
detector = SceneDetector(config)

# Open a video file
cap = cv2.VideoCapture("input_video.mp4")

# Function to generate frames
def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

# Detect scenes
scene_changes = detector.detect_scenes(generate_frames())
print(f"Detected {len(scene_changes)} scene changes")

# Plot metrics
detector.plot_metrics("input_video")
```

### Extracting Data from Processed Videos

```python
from src.core.scene.data_extractor import VideoDataExtractor

# Create data extractor
extractor = VideoDataExtractor(output_dir="data_store")

# Extract data from a single video
video_data = extractor.extract_video_data(
    "output/annotated_video.mp4",
    sample_rate=5  # Sample 1 frame every 5 frames
)

# Extract data from all processed videos
all_data = extractor.extract_all_videos(
    video_dir="output",
    sample_rate=10
)

print(f"Extracted data from {len(all_data)} videos")
```

### Analyzing Video Quality

```python
from src.core.scene.video_quality import VideoQualityAnalyzer
import cv2

# Create quality analyzer with configuration
config = {
    'output': {
        'plots_directory': 'quality_plots'
    }
}
analyzer = VideoQualityAnalyzer(config)

# Process video frames
cap = cv2.VideoCapture("input_video.mp4")
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Compute metrics for the frame
    metrics = analyzer.compute_frame_metrics(frame)
    
    # Update metrics history
    analyzer.update_metrics_history(frame_idx, metrics)
    
    frame_idx += 1

cap.release()

# Plot metrics
analyzer.plot_metrics("input_video")
```

### Running the Complete Analysis Pipeline

```python
from src.core.scene.analyze_video import run_pipeline

# Run the complete pipeline
success = run_pipeline(
    video_path="input_video.mp4",
    api="openai",
    sport_type="running",
    use_ray=True,
    output_dir="analysis_results"
)

if success:
    print("Analysis pipeline completed successfully!")
else:
    print("Analysis pipeline failed.")
```
