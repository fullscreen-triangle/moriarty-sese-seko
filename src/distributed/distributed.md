# Distributed Processing Documentation

This directory contains modules for distributed video processing using Ray.

## Ray Remote Functions

### `analyze_pose(landmark_data)`
- **Location**: `rayprocessor.py`
- **Type**: Ray remote function
- **Purpose**: Analyzes pose landmarks in a distributed manner
- **Parameters**:
  - `landmark_data`: Array of pose landmarks from MediaPipe
- **Returns**: Dictionary with landmark points and timestamp

### `calculate_metrics(pose_data, prev_data=None)`
- **Location**: `rayprocessor.py`
- **Type**: Ray remote function
- **Purpose**: Calculates biomechanical metrics from pose data
- **Parameters**:
  - `pose_data`: Current frame's pose data
  - `prev_data`: Previous frame's pose data (optional)
- **Returns**: Dictionary of calculated metrics including:
  - Shoulder width
  - Hip width
  - Velocity (if previous data is available)

### `process_frame_batch(batch_of_frames, fps)`
- **Location**: `rayprocessor.py`
- **Type**: Helper function
- **Purpose**: Processes a batch of video frames using MediaPipe and Ray
- **Parameters**:
  - `batch_of_frames`: List of frames to process
  - `fps`: Frames per second of the video
- **Returns**: List of tuples containing (annotated_frame, pose_data)

### `draw_annotations(frame, results, metrics, mp_pose)`
- **Location**: `rayprocessor.py`
- **Type**: Helper function
- **Purpose**: Draws pose landmarks and metrics on a video frame
- **Parameters**:
  - `frame`: The video frame
  - `results`: MediaPipe pose results
  - `metrics`: Dictionary of calculated metrics
  - `mp_pose`: MediaPipe pose module
- **Returns**: Annotated frame with landmarks and metrics

## RayVideoProcessor Class

### `__init__(model_path=None, n_workers=None)`
- **Location**: `rayprocessor.py`
- **Purpose**: Initializes the video processor with Ray for distributed computing
- **Parameters**:
  - `model_path`: Path to custom model (optional)
  - `n_workers`: Number of worker processes (defaults to CPU count)

### `process_video(video_path)`
- **Location**: `rayprocessor.py`
- **Purpose**: Processes a video using Ray distributed computing
- **Parameters**:
  - `video_path`: Path to the input video file
- **Returns**: Path to the processed output video
- **Features**:
  - Splits video into batches for parallel processing
  - Uses ThreadPoolExecutor for batch processing
  - Handles video I/O efficiently

### `load_custom_model(model_path)`
- **Location**: `rayprocessor.py`
- **Purpose**: Loads a custom model for additional analysis
- **Parameters**:
  - `model_path`: Path to the model file

## Process With Ray Module

### `process_all_videos(input_folder='public', output_folder='output')`
- **Location**: `process_with_ray.py`
- **Purpose**: Processes all videos in a folder using the RayVideoProcessor
- **Parameters**:
  - `input_folder`: Directory containing input videos
  - `output_folder`: Directory for output videos
- **Features**:
  - Automatically identifies mp4 files for processing
  - Reports progress and errors

## Requirements

This module requires the following packages for the LLM component:
- torch
- transformers
- datasets
- accelerate
- scikit-learn
- numpy
- pandas
- wandb
- tqdm
- sentencepiece
- tensorboard
- peft
- bitsandbytes
- tiktoken
- pyarrow

Additionally, the following packages are needed for video processing:
- Ray
- OpenCV (cv2)
- MediaPipe
- NumPy
