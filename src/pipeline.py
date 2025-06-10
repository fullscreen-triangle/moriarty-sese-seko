"""
Distributed Video Processing and LLM Training Pipeline

This module provides a comprehensive pipeline for:
1. Processing videos using distributed computing (Dask and Ray)
2. Converting pose data to LLM training data
3. Training LLMs or generating synthetic data with OpenAI and Claude models
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp_lib
import psutil  # For memory monitoring
import ray
import ssl
import urllib.request
import threading
import gc

from core.pose.pose_data_to_llm import PoseDataExtractor

# Fix SSL certificate verification issue for MediaPipe model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Global MediaPipe model cache to avoid repeated downloads
_MEDIAPIPE_CACHE = {}
_CACHE_LOCK = threading.Lock()

def get_cached_mediapipe_pose(model_complexity=2):
    """
    Get a cached MediaPipe pose model to avoid repeated downloads.
    Thread-safe singleton pattern for model caching.
    """
    cache_key = f"pose_complexity_{model_complexity}"
    
    with _CACHE_LOCK:
        if cache_key not in _MEDIAPIPE_CACHE:
            print(f"Creating and caching MediaPipe pose model (complexity {model_complexity})...")
            _MEDIAPIPE_CACHE[cache_key] = {
                'mp_pose': mp.solutions.pose,
                'mp_drawing': mp.solutions.drawing_utils,
                'pose': mp.solutions.pose.Pose(
                    static_image_mode=False, 
                    model_complexity=model_complexity,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            }
            print(f"MediaPipe pose model cached successfully!")
        else:
            print(f"Using cached MediaPipe pose model (complexity {model_complexity})")
    
    return _MEDIAPIPE_CACHE[cache_key]

# Robust Dask imports with fallback
try:
    from dask.distributed import Client, LocalCluster
    from dask import delayed
    import dask
    DASK_AVAILABLE = True
    logger_msg = "Dask distributed computing enabled"
except ImportError as e:
    print(f"Warning: Could not import Dask distributed: {e}")
    print("Falling back to non-distributed processing...")
    # Create dummy classes for fallback
    class Client:
        def __init__(self, *args, **kwargs):
            print("Warning: Using dummy Dask Client - no distributed processing")
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, func, *args, **kwargs):
            # Direct execution for fallback
            return func(*args, **kwargs)
        def close(self):
            # Dummy close method for fallback client
            pass
    
    class LocalCluster:
        def __init__(self, *args, **kwargs):
            print("Warning: Using dummy LocalCluster")
            pass
    
    def delayed(func):
        """Dummy delayed decorator that just returns the function"""
        return func
    
    DASK_AVAILABLE = False
    logger_msg = "Dask not available - using direct processing"

from typing import List, Dict, Tuple, Any, Optional
import json
import logging
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline")
logger.info(logger_msg)

# Load environment variables for API keys
load_dotenv()

# Conservative default settings to prevent system crashes
DEFAULT_MEMORY_LIMIT = 0.25  # 25% of memory (reduced from 40%)
DEFAULT_BATCH_SIZE = 5       # Smaller batches (reduced from 30)
DEFAULT_WORKERS = max(1, min(4, mp_lib.cpu_count() // 2))  # Conservative worker count

class MemoryMonitor:
    """Singleton class to monitor system memory and prevent crashes."""
    
    _instance = None
    
    def __init__(self, memory_limit_fraction=DEFAULT_MEMORY_LIMIT):
        if MemoryMonitor._instance is not None:
            raise Exception("MemoryMonitor is a singleton!")
        
        self.memory_limit_fraction = memory_limit_fraction
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit = int(self.total_memory * memory_limit_fraction)
        
        logger.info(f"Memory Monitor initialized: "
                   f"Total: {self.total_memory/(1024**3):.1f}GB, "
                   f"Limit: {self.memory_limit/(1024**3):.1f}GB ({memory_limit_fraction*100:.1f}%)")
        
        MemoryMonitor._instance = self
    
    @staticmethod
    def get_instance(memory_limit_fraction=DEFAULT_MEMORY_LIMIT):
        if MemoryMonitor._instance is None:
            MemoryMonitor(memory_limit_fraction)
        return MemoryMonitor._instance
    
    def get_available_memory(self):
        """Get available memory in bytes."""
        return psutil.virtual_memory().available
    
    def get_memory_usage_percent(self):
        """Get current memory usage as percentage."""
        return psutil.virtual_memory().percent
    
    def check_memory(self, required_memory=None):
        """
        Check if we have enough memory available.
        
        Args:
            required_memory: Required memory in bytes (optional)
            
        Returns:
            bool: True if we have enough memory, False otherwise
        """
        available = self.get_available_memory()
        usage_percent = self.get_memory_usage_percent()
        
        # If we're using more than 80% of system memory, consider it insufficient
        if usage_percent > 80:
            logger.warning(f"High memory usage detected: {usage_percent:.1f}%")
            return False
        
        # If specific memory requirement is given, check against that
        if required_memory:
            return available >= required_memory
        
        # Otherwise, check if we have at least 1GB available
        return available >= (1024 * 1024 * 1024)
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info("Forced garbage collection completed")

class MediapipeProcessor:
    """Handles MediaPipe pose estimation using cached models."""
    
    def __init__(self, model_complexity=2):
        """Initialize the MediaPipe pose estimator with cached model."""
        self.model_complexity = model_complexity
        self.cached_models = get_cached_mediapipe_pose(model_complexity)
        self.mp_pose = self.cached_models['mp_pose']
        self.mp_drawing = self.cached_models['mp_drawing']
        self.pose = self.cached_models['pose']
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        Process a single frame with MediaPipe.
        
        Returns:
            Tuple of (landmarks_np_array, pose_results)
        """
        if frame is None:
            return None, None
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        
        if not pose_results.pose_landmarks:
            return None, None
        
        # Convert to serializable numpy array
        landmarks_array = []
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks_array.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return np.array(landmarks_array), pose_results
    
    def draw_annotations(self, frame: np.ndarray, pose_results: Any, metrics: Dict = None) -> np.ndarray:
        """Draw pose annotations on frame."""
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Draw metrics if provided
        if metrics:
            y = 30
            for key, value in metrics.items():
                text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
        
        return frame

# Function to be distributed with Dask
def process_frame_batch(batch_data: Dict) -> List[Dict]:
    """
    Process a batch of frames with MediaPipe (outside of Ray).
    This function is distributed with Dask.
    
    Important: MediaPipe processing happens here, before Ray serialization.
    """
    frames = batch_data["frames"]
    start_index = batch_data["start_index"]
    processor = MediapipeProcessor(model_complexity=batch_data.get("model_complexity", 2))
    
    results = []
    for i, frame in enumerate(frames):
        if frame is None:
            continue
            
        frame_index = start_index + i
        
        try:
            # MediaPipe processing happens here (not in Ray)
            landmarks_array, pose_results = processor.process_frame(frame)
            
            # Store only serializable data - convert pose_results to a simple format if needed
            serializable_pose_data = None
            if pose_results and pose_results.pose_landmarks:
                # Convert landmarks to a serializable format (numpy array already handled)
                serializable_pose_data = {
                    "frame_index": frame_index,
                    "has_landmarks": True
                }
            
            # Store result
            result = {
                "frame_index": frame_index,
                "frame": frame,
                "landmarks_array": landmarks_array,
                "pose_data": serializable_pose_data
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing frame {frame_index}: {str(e)}")
    
    return results

# Ray function for pose analysis - runs after MediaPipe serialization issues are avoided
@ray.remote(max_calls=100)  # Limit Ray task reuse to prevent memory buildup
def analyze_pose(landmarks_array: np.ndarray, frame_index: int, timestamp: float = None) -> Dict:
    """
    Analyze pose landmarks with Ray (distributed).
    This function runs after MediaPipe processing to avoid serialization issues.
    """
    if landmarks_array is None or len(landmarks_array) == 0:
        return {"frame_index": frame_index, "valid": False}
    
    try:
        # Extract key points (example)
        # In a real implementation, this would do more complex analysis
        nose = landmarks_array[0]
        left_shoulder = landmarks_array[11]
        right_shoulder = landmarks_array[12]
        left_hip = landmarks_array[23]
        right_hip = landmarks_array[24]
        
        # Calculate metrics
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
        
        # Return analysis results
        return {
            "frame_index": frame_index,
            "timestamp": timestamp or frame_index / 30.0,  # Approximate if not provided
            "points": landmarks_array.tolist(),  # Convert to list for serialization
            "metrics": {
                "shoulder_width": float(shoulder_width),
                "hip_width": float(hip_width),
                "shoulder_hip_ratio": float(shoulder_width / hip_width) if hip_width > 0 else 0
            },
            "valid": True
        }
    except Exception as e:
        logger.error(f"Error analyzing pose for frame {frame_index}: {str(e)}")
        return {"frame_index": frame_index, "valid": False, "error": str(e)}

class VideoPipeline:
    """Main pipeline for distributed video processing and LLM training."""
    
    def __init__(self, 
                 memory_limit_fraction: float = DEFAULT_MEMORY_LIMIT,
                 n_workers: int = None, 
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 output_dir: str = "output",
                 model_dir: str = "models",
                 llm_training_dir: str = "llm_training_data",
                 llm_model_dir: str = "llm_models"):
        """
        Initialize the video processing pipeline with conservative resource usage.
        
        Args:
            memory_limit_fraction: Maximum fraction of system memory to use (default 25%)
            n_workers: Number of workers for parallel processing (conservative default)
            batch_size: Number of frames to process in a batch (small default)
            output_dir: Directory for processed videos
            model_dir: Directory for saving pose models
            llm_training_dir: Directory for LLM training data
            llm_model_dir: Directory for trained LLM models
        """
        # Setup resource limits with conservative defaults
        self.memory_monitor = MemoryMonitor.get_instance(memory_limit_fraction)
        
        # Conservative worker count calculation
        if n_workers is None:
            total_cpu_count = mp_lib.cpu_count()
            total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            # Use fewer workers based on memory constraints
            memory_based_workers = max(1, int(total_memory_gb / 4))  # 1 worker per 4GB
            cpu_based_workers = max(1, total_cpu_count // 2)  # Use half of CPU cores
            
            # Take the minimum to be conservative
            self.n_workers = min(memory_based_workers, cpu_based_workers, 4)  # Cap at 4 workers
        else:
            self.n_workers = min(n_workers, 6)  # Never exceed 6 workers regardless of user input
        
        self.batch_size = min(batch_size, 10)  # Cap batch size at 10 frames
        
        logger.info(f"Pipeline initialized with CONSERVATIVE settings:")
        logger.info(f"  - Workers: {self.n_workers}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Memory limit: {memory_limit_fraction*100:.1f}%")
        
        # Set up directories
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.llm_training_dir = Path(llm_training_dir)
        self.llm_model_dir = Path(llm_model_dir)
        
        for directory in [self.output_dir, self.model_dir, self.llm_training_dir, self.llm_model_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize distributed computing with conservative limits
        self.ray_initialized = False
        self.dask_client = None
    
    def _init_ray(self):
        """Initialize Ray with conservative memory limits."""
        if not self.ray_initialized:
            # Calculate very conservative memory limits
            total_memory = psutil.virtual_memory().total
            ray_memory_limit = int(total_memory * 0.15)  # Only 15% of total memory for Ray
            object_store_memory = int(ray_memory_limit * 0.6)  # 60% for object store
            
            try:
                ray.init(
                    ignore_reinit_error=True,
                    object_store_memory=object_store_memory,
                    _memory=ray_memory_limit - object_store_memory,  
                    num_cpus=self.n_workers,
                    num_gpus=0,  # Explicitly disable GPU to avoid conflicts
                    _temp_dir=f"/tmp/ray_pipeline_{int(time.time())}"  # Unique temp dir
                )
                self.ray_initialized = True
                logger.info(f"Ray initialized with CONSERVATIVE limits: "
                           f"{ray_memory_limit/(1024**3):.1f}GB total, "
                           f"{object_store_memory/(1024**3):.1f}GB object store, "
                           f"{self.n_workers} CPUs")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                self.ray_initialized = False
    
    def _init_dask(self):
        """Initialize Dask with conservative memory limits."""
        if self.dask_client is None:
            if not DASK_AVAILABLE:
                logger.info("Using fallback processing instead of Dask distributed")
                self.dask_client = Client()
                return
            
            try:
                # Very conservative memory limit per worker
                total_memory_gb = psutil.virtual_memory().total / (1024**3)
                worker_memory_gb = min(2.0, total_memory_gb / (self.n_workers * 2))  # Cap at 2GB per worker
                worker_memory_limit = f"{int(worker_memory_gb * 1024)}MB"
                
                # Start local Dask cluster with conservative settings
                cluster = LocalCluster(
                    n_workers=self.n_workers,
                    threads_per_worker=1,
                    memory_limit=worker_memory_limit,
                    processes=True,  # Use processes instead of threads
                    dashboard_address=None  # Disable dashboard to save memory
                )
                self.dask_client = Client(cluster)
                logger.info(f"Dask initialized with CONSERVATIVE limits: "
                           f"{self.n_workers} workers, {worker_memory_limit} per worker")
            except Exception as e:
                logger.error(f"Failed to initialize Dask: {e}")
                # Fall back to dummy client
                self.dask_client = Client()

    def process_video(self, video_path: str, output_annotations: bool = True) -> Dict:
        """
        Process a single video with distributed computing and conservative resource usage.
        
        Args:
            video_path: Path to the video file
            output_annotations: Whether to generate annotated video output
            
        Returns:
            Dictionary containing processing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path.name}")
        
        # Initialize distributed computing
        self._init_ray()
        self._init_dask()
        
        try:
            # Check memory before starting
            if not self.memory_monitor.check_memory():
                raise RuntimeError("Insufficient memory to process video")
            
            # Open video and get properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
            
            # Setup output video writer if annotations are enabled
            out = None
            if output_annotations:
                output_path = self.output_dir / f"annotated_{video_path.name}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                if not out.isOpened():
                    logger.warning(f"Could not create output video, disabling annotations")
                    output_annotations = False
            
            # Process frames in small, conservative batches
            frame_index = 0
            all_pose_data = []
            
            # Use a much smaller batch size and process more gradually
            actual_batch_size = min(self.batch_size, 3)  # Cap at 3 frames per batch
            
            logger.info(f"Processing with batch size: {actual_batch_size}")
            
            while cap.isOpened() and self.memory_monitor.check_memory():
                # Read a small batch of frames
                current_batch = []
                for _ in range(actual_batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    current_batch.append(frame)
                
                if not current_batch:
                    break
                
                batch_data = {
                    "frames": current_batch,
                    "start_index": frame_index,
                    "model_complexity": 1  # Use lower complexity to save memory
                }
                
                frame_index += len(current_batch)
                
                # Progress update every 30 frames
                if frame_index % 30 == 0:
                    progress = (frame_index / total_frames) * 100
                    memory_percent = self.memory_monitor.get_memory_usage_percent()
                    logger.info(f"Progress: {frame_index}/{total_frames} frames ({progress:.1f}%), "
                               f"Memory: {memory_percent:.1f}%")
                
                # Process batch with Dask (handles MediaPipe processing)
                try:
                    if DASK_AVAILABLE and hasattr(self.dask_client, 'submit'):
                        future = self.dask_client.submit(process_frame_batch, batch_data)
                        batch_results = future.result()
                    else:
                        # Direct processing when Dask is not available
                        batch_results = process_frame_batch(batch_data)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
                
                # Process results with Ray (but limit concurrent tasks)
                ray_futures = []
                max_concurrent_tasks = min(self.n_workers, 2)  # Limit concurrent Ray tasks
                
                for i, result in enumerate(batch_results):
                    if len(ray_futures) >= max_concurrent_tasks:
                        # Wait for some tasks to complete before submitting more
                        completed_futures = []
                        for rf in ray_futures:
                            try:
                                pose_analysis = ray.get(rf[0])
                                completed_futures.append((rf[1], pose_analysis))
                            except Exception as e:
                                logger.error(f"Ray analysis error: {e}")
                        
                        # Process completed results
                        for result_data, pose_analysis in completed_futures:
                            if pose_analysis.get("valid", False):
                                all_pose_data.append(pose_analysis)
                            
                            # Write frame if annotations are enabled
                            if output_annotations and out:
                                frame = result_data["frame"]
                                if frame is not None:
                                    # Add simple annotation
                                    if pose_analysis.get("valid", False):
                                        cv2.putText(frame, f"Frame: {result_data['frame_index']}", 
                                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    out.write(frame)
                        
                        ray_futures = []
                    
                    # Submit new Ray task
                    frame_index_in_batch = result["frame_index"]
                    landmarks_array = result["landmarks_array"]
                    
                    if landmarks_array is not None:
                        ray_future = analyze_pose.remote(
                            landmarks_array, 
                            frame_index_in_batch,
                            timestamp=frame_index_in_batch / fps
                        )
                        ray_futures.append((ray_future, result))
                
                # Process remaining Ray futures
                for rf in ray_futures:
                    try:
                        pose_analysis = ray.get(rf[0])
                        if pose_analysis.get("valid", False):
                            all_pose_data.append(pose_analysis)
                        
                        # Write frame if annotations are enabled
                        if output_annotations and out:
                            frame = rf[1]["frame"]
                            if frame is not None:
                                if pose_analysis.get("valid", False):
                                    cv2.putText(frame, f"Frame: {rf[1]['frame_index']}", 
                                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                out.write(frame)
                    except Exception as e:
                        logger.error(f"Ray analysis error: {e}")
                
                # Force garbage collection every 100 frames
                if frame_index % 100 == 0:
                    self.memory_monitor.force_garbage_collection()
            
            # Clean up resources
            cap.release()
            if out:
                out.release()
            
            logger.info(f"Processed {len(all_pose_data)} valid pose frames out of {frame_index} total frames")
            
            # Save pose model
            model_output_path = self.model_dir / f"{video_path.stem}_pose_model.json"
            with open(model_output_path, 'w') as f:
                json.dump({
                    "video_name": video_path.name,
                    "total_frames": frame_index,
                    "valid_poses": len(all_pose_data),
                    "fps": fps,
                    "pose_data": all_pose_data[:100]  # Limit saved data to prevent huge files
                }, f, indent=2)
            
            return {
                "success": True,
                "video_path": str(video_path),
                "output_video": str(output_path) if output_annotations else None,
                "pose_model": str(model_output_path),
                "total_frames": frame_index,
                "valid_poses": len(all_pose_data)
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_path.name}: {str(e)}")
            return {
                "success": False,
                "video_path": str(video_path),
                "error": str(e)
            }
        finally:
            # Always clean up resources
            self._cleanup()
    
    def _cleanup(self):
        """Clean up Ray and Dask resources."""
        try:
            if self.ray_initialized:
                ray.shutdown()
                self.ray_initialized = False
                logger.info("Ray shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down Ray: {e}")
        
        try:
            if self.dask_client:
                if hasattr(self.dask_client, 'close'):
                    self.dask_client.close()
                self.dask_client = None
                logger.info("Dask client closed successfully")
        except Exception as e:
            logger.error(f"Error closing Dask client: {e}")
        
        # Force garbage collection
        self.memory_monitor.force_garbage_collection()

def main():
    """Main function for the video processing pipeline."""
    parser = argparse.ArgumentParser(description='Distributed video processing and LLM training pipeline')
    
    # General options
    parser.add_argument('--video', type=str, help='Path to a specific video file to process')
    parser.add_argument('--input', type=str, default='public', 
                       help='Input folder containing videos (default: public)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output folder for processed videos (default: output)')
    parser.add_argument('--models', type=str, default='models',
                       help='Output folder for pose models (default: models)')
    parser.add_argument('--llm_data', type=str, default='llm_training_data',
                       help='Output folder for LLM training data (default: llm_training_data)')
    parser.add_argument('--llm_models', type=str, default='llm_models',
                       help='Output folder for trained LLM models (default: llm_models)')
    
    # Resource management options (CONSERVATIVE DEFAULTS TO PREVENT CRASHES)
    parser.add_argument('--memory_limit', type=float, default=DEFAULT_MEMORY_LIMIT,
                       help=f'Memory limit as a fraction of total system memory (default: {DEFAULT_MEMORY_LIMIT} = 25%% - CONSERVATIVE)')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Number of worker processes/threads (default: auto-detected conservative limit, max 4)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for frame processing (default: {DEFAULT_BATCH_SIZE} frames - SMALL for stability)')
    
    # Processing options
    parser.add_argument('--no_video', action='store_true',
                       help='Do not generate annotated videos, just pose models')
    parser.add_argument('--train_llm', action='store_true',
                       help='Train LLM on extracted pose data')
    parser.add_argument('--sport_type', type=str, default=None,
                       help='Type of sport in the video (for context in LLM training)')
    parser.add_argument('--use_openai', action='store_true',
                       help='Use OpenAI API for synthetic data generation')
    parser.add_argument('--use_claude', action='store_true',
                       help='Use Claude API for synthetic data generation')
    parser.add_argument('--both_llms', action='store_true',
                       help='Use both OpenAI and Claude for synthetic data generation')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VideoPipeline(
        memory_limit_fraction=args.memory_limit,
        n_workers=args.workers,
        batch_size=args.batch_size,
        output_dir=args.output,
        model_dir=args.models,
        llm_training_dir=args.llm_data,
        llm_model_dir=args.llm_models
    )
    
    # Process videos
    processed_models = []
    if args.video:
        # Process single video
        logger.info(f"Processing single video: {args.video}")
        result = pipeline.process_video(
            args.video,
            output_annotations=not args.no_video
        )
        if result.get("pose_model"):
            processed_models.append(result["pose_model"])
            logger.info(f"Processed {result['valid_poses']}/{result['total_frames']} frames in {result['processing_time']:.2f} seconds")
    else:
        # Process all videos in folder
        logger.info(f"Processing all videos in: {args.input}")
        results = pipeline.process_all_videos(args.input)
        processed_models = [r["pose_model"] for r in results if r.get("pose_model")]
        
        if processed_models:
            logger.info(f"Successfully processed {len(processed_models)} videos")
        else:
            logger.warning("No videos were successfully processed")
    
    # Generate LLM training data
    if processed_models and (args.train_llm or args.use_openai or args.use_claude or args.both_llms):
        logger.info(f"Generating training data from {len(processed_models)} pose models")
        training_data = pipeline.generate_llm_training_data(
            processed_models, 
            sport_type=args.sport_type
        )
        
        if not training_data:
            logger.error("Failed to generate training data")
            return
            
        logger.info(f"Training data generated: {training_data}")
        
        # Train LLM or generate synthetic data
        use_openai = args.use_openai or args.both_llms
        use_claude = args.use_claude or args.both_llms
        
        if use_openai or use_claude or args.train_llm:
            logger.info("Training LLM/generating synthetic data")
            training_results = pipeline.train_llm(
                training_data_path=training_data,
                use_openai=use_openai,
                use_claude=use_claude
            )
            
            if training_results["success"]:
                if training_results.get("openai_model"):
                    logger.info(f"OpenAI synthetic data generated: {training_results['openai_model']}")
                if training_results.get("claude_model"):
                    logger.info(f"Claude synthetic data generated: {training_results['claude_model']}")
                if training_results.get("local_model"):
                    logger.info(f"Local model trained: {training_results['local_model']}")
            else:
                logger.error(f"LLM training failed: {training_results.get('error', 'Unknown error')}")
    
    logger.info("Pipeline execution complete!")

if __name__ == "__main__":
    main() 