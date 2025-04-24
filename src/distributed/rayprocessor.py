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

# Define remote functions outside of any class
@ray.remote
def analyze_pose(landmark_data):
    """Analyze pose landmarks remotely."""
    # Extract relevant points and calculate metrics
    # This is a simplified version of what PoseAnalyzer would do
    result = {
        "points": landmark_data,  # Already numpy array from caller
        "timestamp": time.time()
    }
    return result

@ray.remote
def calculate_metrics(pose_data, prev_data=None):
    """Calculate metrics from pose data remotely."""
    metrics = {}
    
    if not pose_data or "points" not in pose_data:
        return metrics
    
    points = pose_data["points"]
    
    # Calculate some basic metrics
    # Shoulder width (points 11 and 12 are shoulders in MediaPipe)
    if len(points) > 12:
        shoulder_width = np.linalg.norm(points[11] - points[12])
        metrics["shoulder_width"] = float(shoulder_width)
    
    # Hip width (points 23 and 24 are hips)
    if len(points) > 24:
        hip_width = np.linalg.norm(points[23] - points[24])
        metrics["hip_width"] = float(hip_width)
    
    # Calculate velocity if we have previous data
    if prev_data and "points" in prev_data and "timestamp" in prev_data and "timestamp" in pose_data:
        dt = pose_data["timestamp"] - prev_data["timestamp"]
        if dt > 0 and len(points) > 0 and len(prev_data["points"]) > 0:
            # Calculate velocity of center of mass (average of shoulders and hips)
            if len(points) > 24 and len(prev_data["points"]) > 24:
                com_current = np.mean(points[[11, 12, 23, 24]], axis=0)
                com_prev = np.mean(prev_data["points"][[11, 12, 23, 24]], axis=0)
                velocity = np.linalg.norm(com_current - com_prev) / dt
                metrics["velocity"] = float(velocity)
    
    return metrics

# Optimize serialization by using numpy arrays directly
class OptimizedNumpySerializer:
    """Custom serializer for numpy arrays to improve performance with Ray."""
    
    @staticmethod
    def serialize(obj):
        return pickle.dumps(obj)
    
    @staticmethod
    def deserialize(data):
        return pickle.loads(data)

# Register custom serializer with Ray
if not ray.is_initialized():
    ray.init()
ray.util.register_serializer(
    np.ndarray,
    serializer=OptimizedNumpySerializer.serialize,
    deserializer=OptimizedNumpySerializer.deserialize
)

class MemoryMonitor:
    """Monitor system memory and CPU resources for adaptive batch sizing."""
    
    @staticmethod
    def get_available_memory_gb():
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)
    
    @staticmethod
    def get_cpu_usage_percent():
        """Get current CPU usage as percentage."""
        return psutil.cpu_percent()
    
    @staticmethod
    def get_optimal_batch_size(frame_shape, min_batch=1, max_batch=30):
        """
        Calculate optimal batch size based on available memory and CPU resources.
        
        Args:
            frame_shape: Shape of a single frame (height, width, channels)
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            
        Returns:
            int: Optimal batch size
        """
        available_memory_gb = MemoryMonitor.get_available_memory_gb()
        cpu_usage = MemoryMonitor.get_cpu_usage_percent()
        
        # Calculate memory needed for a single frame (in GB)
        # Multiply by 4 to account for RGBA frames and intermediate results
        frame_memory_gb = np.prod(frame_shape) * 4 / (1024 ** 3)
        
        # Basic calculation: how many frames can we fit in 70% of available memory
        memory_based_batch = int(0.7 * available_memory_gb / frame_memory_gb)
        
        # Adjust based on CPU usage (reduce batch size when CPU is busy)
        cpu_factor = max(0.5, 1.0 - (cpu_usage / 200))  # CPU factor between 0.5 and 1.0
        
        # Calculate final batch size
        batch_size = int(memory_based_batch * cpu_factor)
        
        # Ensure batch size is within limits
        batch_size = max(min_batch, min(batch_size, max_batch))
        
        logger.info(f"Adaptive batch sizing: memory={available_memory_gb:.2f}GB, CPU={cpu_usage:.1f}%, " +
                   f"frame={frame_memory_gb:.4f}GB, batch_size={batch_size}")
        
        return batch_size

def process_frame_batch(batch_of_frames, fps):
    """Process a batch of frames locally (not as a Ray task)."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    results = []
    prev_data = None
    
    for frame in batch_of_frames:
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

def draw_annotations(frame, results, metrics, mp_pose):
    """Draw pose landmarks and metrics on the frame."""
    # Draw pose landmarks
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    
    # Draw metrics
    y_pos = 30
    for metric, value in metrics.items():
        cv2.putText(frame, f"{metric}: {value:.2f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
    return frame

class RayVideoProcessor:
    def __init__(self, model_path=None, n_workers=None):
        """Initialize the video processor with Ray for distributed computing."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            logger.info("Ray initialized for distributed processing")
        
        self.n_workers = n_workers or multiproc.cpu_count()
        
        # Dynamically adjust worker count based on system resources
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Allow one worker per 4GB of total memory, but at least 2 workers
        memory_based_workers = max(2, int(total_memory_gb / 4))
        
        # Choose the smaller of CPU count or memory-based worker count
        self.n_workers = min(self.n_workers, memory_based_workers)
        
        logger.info(f"Using {self.n_workers} workers based on system resources")
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Custom model for additional analysis
        self.custom_model = None
        if model_path:
            self.load_custom_model(model_path)

    def process_video(self, video_path):
        """Process video using Ray for distributed computing and save annotated output."""
        logger.info(f"Starting to process {video_path} with Ray distributed computing...")
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
        
        # Determine frame shape for memory calculations
        ret, test_frame = cap.read()
        if not ret:
            raise Exception("Could not read test frame from video")
        
        frame_shape = test_frame.shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Use adaptive batch sizing based on available memory and system resources
        initial_batch_size = MemoryMonitor.get_optimal_batch_size(frame_shape)
        
        logger.info(f"Initial adaptive batch size: {initial_batch_size}")
        
        # Prepare frame batches for processing
        frame_batches = []
        current_batch = []
        frame_count = 0
        
        logger.info("Reading frames and preparing batches...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Reading frame {frame_count}/{total_frames}")
                # Recalculate batch size every 100 frames based on current system state
                if frame_count % 1000 == 0:
                    new_batch_size = MemoryMonitor.get_optimal_batch_size(frame_shape)
                    if new_batch_size != initial_batch_size:
                        logger.info(f"Adjusting batch size from {initial_batch_size} to {new_batch_size}")
                        initial_batch_size = new_batch_size
                
            current_batch.append(frame)
            if len(current_batch) >= initial_batch_size:
                frame_batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            frame_batches.append(current_batch)
            
        cap.release()  # Release the capture early
        
        logger.info(f"Prepared {len(frame_batches)} batches for processing")
        logger.info(f"Starting parallel processing with {self.n_workers} workers")
        
        # Process frame batches in parallel
        # Monitor system resources during processing and adjust worker count if needed
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Process batches in smaller chunks to avoid memory issues
            remaining_batches = frame_batches.copy()
            
            while remaining_batches:
                # Check current memory status and adjust worker count if needed
                avail_memory_gb = MemoryMonitor.get_available_memory_gb()
                if avail_memory_gb < 1.0:  # Less than 1GB available
                    logger.warning(f"Low memory detected ({avail_memory_gb:.2f}GB available)")
                    # Reduce active workers temporarily
                    active_workers = max(1, self.n_workers // 2)
                    logger.info(f"Reducing active workers to {active_workers} due to low memory")
                else:
                    active_workers = self.n_workers
                
                # Take up to active_workers batches at a time
                current_batches = remaining_batches[:active_workers]
                remaining_batches = remaining_batches[active_workers:]
                
                future_results = [
                    executor.submit(process_frame_batch, batch, fps) 
                    for batch in current_batches
                ]
                
                # Gather results and write immediately to save memory
                for i, future in enumerate(future_results):
                    logger.info(f"Processing batch {i+1}/{len(current_batches)}...")
                    batch_results = future.result()
                    
                    # Write frames immediately instead of storing them
                    for frame, _ in batch_results:
                        if frame is not None:
                            out.write(frame)
                    
                    # Don't store all frames in memory
                    del batch_results
        
        # Release resources
        out.release()
        
        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info(f"Successfully processed video. Output saved to {output_path}")
            logger.info(f"Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return str(output_path)
        else:
            raise Exception(f"Output file is missing or too small: {output_path}")
        
    def load_custom_model(self, model_path):
        """Load a custom model for additional analysis."""
        # Implement model loading
        pass 