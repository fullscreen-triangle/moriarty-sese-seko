import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import ray
import multiprocessing as multiproc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
import time
import os
import pickle
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import cached MediaPipe function from pipeline
try:
    from src.pipeline import get_cached_mediapipe_pose
except ImportError:
    from pipeline import get_cached_mediapipe_pose

# Define remote functions outside of any class
@ray.remote(max_calls=50)  # Limit task reuse to prevent memory buildup
def analyze_pose(landmark_data):
    """Analyze pose landmarks remotely."""
    # Extract relevant points and calculate metrics
    # This is a simplified version of what PoseAnalyzer would do
    result = {
        "points": landmark_data,  # Already numpy array from caller
        "timestamp": time.time()
    }
    return result

@ray.remote(max_calls=50)
def calculate_metrics(pose_data, prev_data):
    """Calculate pose metrics remotely."""
    metrics = {
        "velocity": 0.0,
        "smoothness": 1.0,
        "stability": 1.0
    }
    
    if prev_data and pose_data:
        try:
            # Simple velocity calculation between consecutive frames
            current_points = np.array(pose_data.get("points", []))
            prev_points = np.array(prev_data.get("points", []))
            
            if len(current_points) == len(prev_points) and len(current_points) > 0:
                velocity = np.mean(np.linalg.norm(current_points - prev_points, axis=1))
                metrics["velocity"] = float(velocity)
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
    
    return metrics

def draw_annotations(frame, pose_results, metrics, mp_pose):
    """Draw annotations on frame using MediaPipe."""
    if pose_results and pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    
    # Draw metrics
    if metrics:
        y = 30
        for key, value in metrics.items():
            text = f"{key}: {value:.3f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
    
    return frame

# Conservative memory monitoring
class MemoryMonitor:
    @staticmethod
    def get_available_memory_gb():
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)
    
    @staticmethod
    def get_cpu_usage_percent():
        """Get current CPU usage as percentage."""
        return psutil.cpu_percent()
    
    @staticmethod
    def check_memory_available(min_gb=1.0):
        """Check if we have at least min_gb of memory available."""
        available = MemoryMonitor.get_available_memory_gb()
        cpu_usage = MemoryMonitor.get_cpu_usage_percent()
        
        # Check both memory and CPU usage
        memory_ok = available >= min_gb
        cpu_ok = cpu_usage < 85  # Don't proceed if CPU is over 85%
        
        if not memory_ok:
            logger.warning(f"Low memory: {available:.2f}GB available (need {min_gb}GB)")
        if not cpu_ok:
            logger.warning(f"High CPU usage: {cpu_usage:.1f}%")
            
        return memory_ok and cpu_ok

def process_frame_batch(batch_of_frames, fps):
    """Process a batch of frames locally (not as a Ray task) using cached MediaPipe models."""
    # Use cached MediaPipe models
    cached_models = get_cached_mediapipe_pose(model_complexity=1)  # Use lower complexity
    mp_pose = cached_models['mp_pose']
    pose = cached_models['pose']
    
    results = []
    prev_data = None
    
    for frame in batch_of_frames:
        # Check memory before processing each frame
        if not MemoryMonitor.check_memory_available(0.5):  # Need at least 500MB
            logger.warning("Skipping frame due to low memory")
            results.append((frame, None))
            continue
            
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        
        if pose_results.pose_landmarks:
            # Convert landmarks to numpy array for ray serialization - more efficient than lists
            landmarks_array = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                       for landmark in pose_results.pose_landmarks.landmark], 
                                       dtype=np.float32)
            
            # Submit for distributed analysis
            pose_data_id = analyze_pose.remote(landmarks_array)
            pose_data = ray.get(pose_data_id)
            
            # Calculate metrics
            metrics_id = calculate_metrics.remote(pose_data, prev_data)
            metrics = ray.get(metrics_id)
            
            # Draw annotations on the frame
            annotated_frame = draw_annotations(frame, pose_results, metrics, mp_pose)
            
            results.append((annotated_frame, pose_data))
            prev_data = pose_data
        else:
            results.append((frame, None))
    
    return results

class RayVideoProcessor:
    def __init__(self, model_path=None, n_workers=None):
        """Initialize the video processor with Ray for distributed computing and conservative resource limits."""
        # Don't initialize Ray here - let it be initialized by the caller with proper limits
        if not ray.is_initialized():
            # Initialize with very conservative limits
            total_memory = psutil.virtual_memory().total
            object_store_memory = int(total_memory * 0.1)  # Only 10% for object store
            
            ray.init(
                ignore_reinit_error=True,
                object_store_memory=object_store_memory,
                num_cpus=min(4, multiproc.cpu_count()),  # Cap at 4 CPUs
                num_gpus=0  # Disable GPU to avoid conflicts
            )
            logger.info("Ray initialized with conservative limits for video processing")
        
        # Conservative worker count
        total_cpus = multiproc.cpu_count()
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Use fewer workers based on memory constraints  
        memory_based_workers = max(1, int(total_memory_gb / 3))  # 1 worker per 3GB
        cpu_based_workers = max(1, total_cpus // 2)  # Use half of CPU cores
        
        # Take the minimum and cap at 4
        self.n_workers = min(memory_based_workers, cpu_based_workers, 4)
        
        if n_workers:
            self.n_workers = min(n_workers, 4)  # Never exceed 4 workers
        
        logger.info(f"Using {self.n_workers} workers for conservative processing")
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Custom model for additional analysis
        self.custom_model = None
        if model_path:
            self.load_custom_model(model_path)

    def process_video(self, video_path):
        """Process video using Ray for distributed computing with conservative resource usage."""
        logger.info(f"Starting conservative video processing: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        output_path = self.output_dir / f"annotated_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            raise Exception(f"Could not open output video file {output_path}")
        
        # Use very small batches to prevent memory issues
        batch_size = min(3, self.n_workers)  # Very small batches
        
        # Prepare frame batches for processing
        frame_batches = []
        current_batch = []
        frame_count = 0
        
        logger.info(f"Reading frames with conservative batch size: {batch_size}")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % 60 == 0:  # Progress every 60 frames
                logger.info(f"Reading frame {frame_count}/{total_frames}")
                
                # Check memory periodically
                if not MemoryMonitor.check_memory_available(1.0):
                    logger.warning("Low memory detected, reducing batch size")
                    batch_size = max(1, batch_size - 1)
                
            current_batch.append(frame)
            if len(current_batch) >= batch_size:
                frame_batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            frame_batches.append(current_batch)
            
        cap.release()  # Release the capture early
        
        logger.info(f"Prepared {len(frame_batches)} batches for conservative processing")
        
        # Process frame batches sequentially to avoid overwhelming the system
        processed_frames = 0
        
        for i, batch in enumerate(frame_batches):
            # Check memory before each batch
            if not MemoryMonitor.check_memory_available(0.8):
                logger.warning(f"Stopping processing at batch {i} due to low memory")
                break
                
            logger.info(f"Processing batch {i+1}/{len(frame_batches)} ({len(batch)} frames)")
            
            try:
                # Process batch with reduced parallelism
                batch_results = process_frame_batch(batch, fps)
                
                # Write frames immediately to save memory
                for frame, pose_data in batch_results:
                    if frame is not None:
                        out.write(frame)
                    processed_frames += 1
                    
                # Clear batch results to free memory
                del batch_results
                
                # Progress update
                if (i + 1) % 10 == 0:
                    progress = ((i + 1) / len(frame_batches)) * 100
                    memory_gb = MemoryMonitor.get_available_memory_gb()
                    cpu_percent = MemoryMonitor.get_cpu_usage_percent()
                    logger.info(f"Progress: {progress:.1f}%, Memory: {memory_gb:.2f}GB, CPU: {cpu_percent:.1f}%")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Write original frames on error
                for frame in batch:
                    if frame is not None:
                        out.write(frame)
                    processed_frames += 1
        
        # Release resources
        out.release()
        
        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info(f"Successfully processed video. Output saved to {output_path}")
            logger.info(f"Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            logger.info(f"Processed {processed_frames} frames")
            return str(output_path)
        else:
            raise Exception(f"Output file is missing or too small: {output_path}")
        
    def load_custom_model(self, model_path):
        """Load a custom model for additional analysis."""
        # Implement model loading
        pass 