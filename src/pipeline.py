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
import torch  # For checking GPU availability
import queue
import threading

# Import internal moriarty modules
from .core.pose.pose_data_to_llm import PoseDataExtractor
from .distributed.rayprocessor import RayVideoProcessor
from .utils import file_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline")
logger.info(logger_msg)

# Load environment variables for API keys
load_dotenv()

# Default settings
DEFAULT_MEMORY_LIMIT = 0.4  # 40% of memory
DEFAULT_BATCH_SIZE = 30
DEFAULT_WORKERS = max(1, mp_lib.cpu_count() - 1)  # Leave one CPU free

class MemoryMonitor:
    """Monitors system memory usage and provides controls.
    
    This class implements the Singleton pattern to ensure only one instance
    exists throughout the application lifecycle.
    """
    _instance = None
    
    def __new__(cls, memory_limit_fraction=DEFAULT_MEMORY_LIMIT):
        """Create a singleton instance or return existing one."""
        if cls._instance is None:
            cls._instance = super(MemoryMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, memory_limit_fraction=DEFAULT_MEMORY_LIMIT):
        """Initialize memory monitor with a fraction limit."""
        # Only initialize once
        if getattr(self, '_initialized', False):
            return
            
        self.memory_limit_fraction = memory_limit_fraction
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit = self.total_memory * self.memory_limit_fraction
        logger.info(f"Memory monitor initialized with limit: {self.memory_limit / (1024**3):.2f} GB "
                   f"({self.memory_limit_fraction * 100:.0f}% of total {self.total_memory / (1024**3):.2f} GB)")
        
        # Add threshold warnings
        self.warning_threshold = 0.85 * self.memory_limit
        self.critical_threshold = 0.95 * self.memory_limit
        self._initialized = True
    
    @classmethod
    def get_instance(cls, memory_limit_fraction=DEFAULT_MEMORY_LIMIT):
        """Get or create the singleton instance."""
        return cls(memory_limit_fraction)
    
    def check_memory(self) -> bool:
        """Check if memory usage is below limit."""
        current_usage = psutil.virtual_memory().used
        is_ok = current_usage < self.memory_limit
        usage_percent = current_usage / self.total_memory * 100
        
        # Add warning levels for better monitoring
        if current_usage >= self.critical_threshold:
            logger.warning(f"CRITICAL memory usage: {usage_percent:.1f}% (limit: {self.memory_limit_fraction * 100:.0f}%)")
        elif current_usage >= self.warning_threshold:
            logger.warning(f"HIGH memory usage: {usage_percent:.1f}% (limit: {self.memory_limit_fraction * 100:.0f}%)")
        elif not is_ok:
            logger.warning(f"Memory usage exceeds limit: {usage_percent:.1f}% (limit: {self.memory_limit_fraction * 100:.0f}%)")
        
        return is_ok
    
    def get_current_usage_fraction(self) -> float:
        """Return current memory usage as a fraction of total."""
        return psutil.virtual_memory().used / self.total_memory
    
    def get_available_memory(self) -> int:
        """Return available memory within our limit in bytes."""
        current_usage = psutil.virtual_memory().used
        if current_usage >= self.memory_limit:
            return 0
        return int(self.memory_limit - current_usage)

class MediapipeProcessor:
    """Handles MediaPipe pose estimation outside of Ray/Dask distributed computing."""
    
    def __init__(self, model_complexity=2):
        """Initialize the MediaPipe pose estimator."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
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

    def draw_annotations(self, frame: np.ndarray, pose_results, metrics: Dict = None) -> np.ndarray:
        """Draw pose landmarks and metrics on the frame."""
        if frame is None or pose_results is None:
            return frame
        
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Draw metrics if available
        if metrics:
            y_pos = 30
            for metric, value in metrics.items():
                cv2.putText(frame, f"{metric}: {value:.2f}", (10, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
        
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
@ray.remote
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
        Initialize the video processing pipeline.
        
        Args:
            memory_limit_fraction: Maximum fraction of system memory to use (default 40%)
            n_workers: Number of workers for parallel processing
            batch_size: Number of frames to process in a batch
            output_dir: Directory for processed videos
            model_dir: Directory for saving pose models
            llm_training_dir: Directory for LLM training data
            llm_model_dir: Directory for trained LLM models
        """
        # Setup resource limits
        self.memory_monitor = MemoryMonitor.get_instance(memory_limit_fraction)
        self.n_workers = n_workers or DEFAULT_WORKERS
        self.batch_size = batch_size
        
        # Adjust workers based on available memory and batch size
        total_cpu_count = mp_lib.cpu_count()
        memory_per_worker = self.memory_monitor.memory_limit / self.n_workers
        
        # Ensure we have enough memory per worker for batch processing
        if memory_per_worker < 500 * 1024 * 1024:  # Less than 500MB per worker
            self.n_workers = max(1, int(self.memory_monitor.memory_limit / (500 * 1024 * 1024)))
            logger.warning(f"Reduced worker count to {self.n_workers} due to memory constraints")
        
        logger.info(f"Pipeline initialized with {self.n_workers} workers and batch size {self.batch_size}")
        
        # Set up directories
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.llm_training_dir = Path(llm_training_dir)
        self.llm_model_dir = Path(llm_model_dir)
        
        for directory in [self.output_dir, self.model_dir, self.llm_training_dir, self.llm_model_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize Ray and Dask but don't start clusters yet
        # We'll start them when needed with proper resource limits
        self.ray_initialized = False
        self.dask_client = None
    
    def _init_ray(self):
        """Initialize Ray with memory limits."""
        if not self.ray_initialized:
            # Calculate memory limit in bytes (within our 40% total limit)
            mem_limit_bytes = int(self.memory_monitor.get_available_memory() * 0.8)  # 80% of available memory
            
            # Initialize Ray
            ray.init(
                ignore_reinit_error=True,
                object_store_memory=int(mem_limit_bytes * 0.5),  # Half for object store
                _memory=int(mem_limit_bytes * 0.5),  # Half for Ray's processes
                num_cpus=self.n_workers
            )
            self.ray_initialized = True
            logger.info(f"Ray initialized with {self.n_workers} CPUs and "
                       f"{mem_limit_bytes/(1024**3):.1f} GB memory limit")
    
    def _init_dask(self):
        """Initialize Dask with memory limits."""
        if self.dask_client is None:
            if not DASK_AVAILABLE:
                # Use dummy client when Dask is not available
                logger.info("Using fallback processing instead of Dask distributed")
                self.dask_client = Client()
                return
            
            # Calculate memory limit per worker
            worker_memory_limit = f"{int(self.memory_monitor.memory_limit / self.n_workers / (1024**2))}MB"
            
            # Start local Dask cluster
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=1,
                memory_limit=worker_memory_limit
            )
            self.dask_client = Client(cluster)
            logger.info(f"Dask initialized with {self.n_workers} workers, "
                       f"each limited to {worker_memory_limit} memory")
    
    def _cleanup(self):
        """Clean up Ray and Dask resources."""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
        
        if self.dask_client:
            self.dask_client.close()
            self.dask_client = None
    
    def process_video(self, video_path: str, output_annotations: bool = True) -> Dict:
        """
        Process a video file using distributed computing with both Dask and Ray.
        
        Args:
            video_path: Path to the video file
            output_annotations: Whether to output an annotated video
            
        Returns:
            Dict with results including paths to output files
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        
        # Initialize Ray and Dask
        self._init_ray()
        self._init_dask()
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
            
            # Setup output video writer if needed
            out = None
            output_video_path = None
            if output_annotations:
                output_video_path = self.output_dir / f"annotated_{video_path.name}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    raise ValueError(f"Could not create output video file: {output_video_path}")
            
            # Prepare batches for processing with Dask - use adaptive batch system
            # Start with smaller batches that grow if memory allows
            current_batch_size = min(self.batch_size, 10)  # Start smaller
            
            # Process frames in chunks to manage memory
            frame_index = 0
            all_pose_data = []
            
            # Create a threadpool for Ray futures management
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Process video in chunks
                while cap.isOpened() and self.memory_monitor.check_memory():
                    # Read batch of frames
                    current_batch = []
                    batch_read_start = time.time()
                    
                    for _ in range(current_batch_size):
                        success, frame = cap.read()
                        if not success:
                            break
                        current_batch.append(frame)
                    
                    if not current_batch:
                        break
                    
                    batch_data = {
                        "frames": current_batch,
                        "start_index": frame_index,
                        "model_complexity": 2
                    }
                    
                    frame_index += len(current_batch)
                    batch_read_time = time.time() - batch_read_start
                    
                    # Progress update
                    if frame_index % 100 == 0 or len(current_batch) < current_batch_size:
                        logger.info(f"Read {frame_index}/{total_frames} frames ({frame_index/total_frames*100:.1f}%)")
                    
                    # Process batch with Dask (handles MediaPipe processing)
                    batch_process_start = time.time()
                    if DASK_AVAILABLE:
                        future = self.dask_client.submit(process_frame_batch, batch_data)
                        batch_results = future.result()
                    else:
                        # Direct processing when Dask is not available
                        batch_results = process_frame_batch(batch_data)
                    batch_process_time = time.time() - batch_process_start
                    
                    # Extract landmarks and send to Ray for analysis
                    ray_futures = []
                    ray_process_start = time.time()
                    
                    for result in batch_results:
                        frame_index_in_batch = result["frame_index"]
                        landmarks_array = result["landmarks_array"]
                        
                        # Submit to Ray for analysis (only if we have landmarks)
                        if landmarks_array is not None:
                            # Create Ray task for further analysis (after MediaPipe)
                            ray_future = analyze_pose.remote(
                                landmarks_array, 
                                frame_index_in_batch,
                                timestamp=frame_index_in_batch / fps
                            )
                            ray_futures.append((ray_future, result))
                    
                    # Process Ray results as they complete and write video frames
                    for ray_future, result in ray_futures:
                        pose_analysis = ray.get(ray_future)
                        
                        # Store pose data for model only if valid
                        if pose_analysis.get("valid", False):
                            all_pose_data.append(pose_analysis)
                        
                        # If we're creating an annotated video
                        if output_annotations and out:
                            frame = result["frame"]
                            if frame is None:
                                continue
                                
                            # Draw annotations if we have valid analysis
                            if pose_analysis.get("valid", False):
                                # Create processor for drawing
                                processor = MediapipeProcessor()
                                
                                # Extract metrics for annotation
                                metrics = pose_analysis.get("metrics", {})
                                
                                # Convert landmarks back to MediaPipe format for drawing
                                # This is a workaround for the serialization limitation
                                landmarks_np = result["landmarks_array"]
                                if landmarks_np is not None:
                                    # Create a dummy pose_results object that's drawable
                                    class DummyPoseLandmarks:
                                        def __init__(self, landmarks_array):
                                            self.landmark = []
                                            for lm in landmarks_array:
                                                landmark = type('', (), {})()
                                                landmark.x, landmark.y, landmark.z, landmark.visibility = lm
                                                self.landmark.append(landmark)
                                    
                                    class DummyResults:
                                        def __init__(self, landmarks):
                                            self.pose_landmarks = landmarks
                                            
                                    dummy_landmarks = DummyPoseLandmarks(landmarks_np)
                                    dummy_results = DummyResults(dummy_landmarks)
                                    
                                    # Draw with processor
                                    annotated_frame = processor.draw_annotations(frame, dummy_results, metrics)
                                    out.write(annotated_frame)
                            else:
                                # Write original frame if no valid analysis
                                out.write(frame)
                    
                    ray_process_time = time.time() - ray_process_start
                    
                    # Log timing for this batch
                    logger.debug(f"Batch timing: Read={batch_read_time:.2f}s, Dask={batch_process_time:.2f}s, Ray={ray_process_time:.2f}s")
                    
                    # Adaptive batch size based on memory usage
                    memory_usage = self.memory_monitor.get_current_usage_fraction()
                    if memory_usage < 0.7 * self.memory_monitor.memory_limit_fraction and current_batch_size < self.batch_size:
                        # Increase batch size if we have memory headroom
                        current_batch_size = min(current_batch_size + 5, self.batch_size)
                    elif memory_usage > 0.9 * self.memory_monitor.memory_limit_fraction and current_batch_size > 5:
                        # Decrease batch size if memory is getting tight
                        current_batch_size = max(5, current_batch_size - 5)
                        logger.info(f"Reduced batch size to {current_batch_size} due to memory usage")
            
            cap.release()
            
            # Close the output video
            if out:
                out.release()
            
            # Save pose data to a model file
            model_file = self.model_dir / f"pose_model_{video_path.stem}_{int(time.time())}.json"
            with open(model_file, 'w') as f:
                # Use a list comprehension to filter out non-serializable elements
                serializable_data = []
                for item in all_pose_data:
                    # Convert numpy arrays to lists for JSON serialization if needed
                    serializable_item = {}
                    for k, v in item.items():
                        if isinstance(v, np.ndarray):
                            serializable_item[k] = v.tolist()
                        elif isinstance(v, (np.float32, np.float64)):
                            serializable_item[k] = float(v)
                        elif isinstance(v, (np.int32, np.int64)):
                            serializable_item[k] = int(v)
                        else:
                            serializable_item[k] = v
                    serializable_data.append(serializable_item)
                
                json.dump(serializable_data, f)
            
            total_time = time.time() - start_time
            logger.info(f"Processing complete in {total_time:.2f} seconds")
            logger.info(f"Processed {len(all_pose_data)} valid frames out of {total_frames} total frames")
            logger.info(f"Saved pose model to {model_file}")
            if output_video_path:
                logger.info(f"Saved annotated video to {output_video_path}")
            
            return {
                "pose_model_path": str(model_file),
                "annotated_video_path": str(output_video_path) if output_video_path else None,
                "frame_count": len(all_pose_data),
                "total_frames": total_frames,
                "processing_time": total_time
            }
            
        finally:
            # Clean up resources
            self._cleanup()
    
    def process_all_videos(self, input_folder: str = "public") -> List[Dict]:
        """
        Process all videos in the input folder.
        
        Args:
            input_folder: Folder containing video files
            
        Returns:
            List of results dictionaries
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_path}")
        
        video_files = list(input_path.glob("*.mp4"))
        logger.info(f"Found {len(video_files)} videos to process in {input_folder}")
        
        results = []
        for video_file in video_files:
            logger.info(f"\nProcessing {video_file.name}...")
            try:
                result = self.process_video(video_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {str(e)}")
        
        return results
    
    def generate_llm_training_data(self, pose_model_paths: List[str], sport_type: str = None) -> str:
        """
        Generate training data for LLMs from pose models.
        
        Args:
            pose_model_paths: List of paths to pose model JSON files
            sport_type: Type of sport for context (optional)
            
        Returns:
            Path to generated training data file
        """
        # Use the existing PoseDataExtractor from moriarty.core.pose_data_to_llm
        extractor = PoseDataExtractor(
            model_dir=str(self.model_dir),
            output_dir=str(self.llm_training_dir)
        )
        
        logger.info(f"Generating training data from {len(pose_model_paths)} pose models")
        
        # Process specific models
        examples = []
        for model_path in pose_model_paths:
            model_examples = extractor.process_model(model_path, sport_type=sport_type)
            examples.extend(model_examples)
        
        # Save to a combined file
        timestamp = int(time.time())
        output_file = self.llm_training_dir / f"all_examples_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Saved {len(examples)} training examples to {output_file}")
        return str(output_file)
    
    def train_llm(self, training_data_path: str = None, model_name: str = None,
                use_openai: bool = False, use_claude: bool = False,
                epochs: int = 3, batch_size: int = 4) -> Dict:
        """
        Train an LLM on pose data or generate synthetic data using API models.
        
        Args:
            training_data_path: Path to training data file
            model_name: Name for trained model
            use_openai: Whether to use OpenAI API for synthetic data
            use_claude: Whether to use Claude API for synthetic data
            epochs: Number of training epochs for local model
            batch_size: Batch size for local model training
            
        Returns:
            Dictionary with paths to generated models/data
        """
        try:
            from .core.train_llm import LLMTrainer
        except ImportError:
            # Fall back to external train_llm module if necessary
            try:
                from src.models.train_llm import LLMTrainer
            except ImportError:
                logger.error("Could not import LLMTrainer. Make sure train_llm.py is available.")
                return {"success": False, "error": "Missing train_llm.py module"}
        
        if not training_data_path:
            logger.error("No training data provided")
            return {"success": False, "error": "No training data path provided"}
        
        # Initialize trainer
        trainer = LLMTrainer(
            data_dir=str(self.llm_training_dir),
            model_dir=str(self.llm_model_dir)
        )
        
        results = {"success": False, "openai_model": None, "claude_model": None, "local_model": None}
        
        # Generate synthetic data with both models if requested
        if use_openai:
            logger.info("Generating synthetic training data using OpenAI API...")
            trainer.use_api_model(openai=True, claude=False)
            openai_model = trainer.generate_synthetic_data(
                data_file=training_data_path,
                output_name=model_name + "_openai" if model_name else f"synthetic_openai_{int(time.time())}"
            )
            if openai_model:
                results["openai_model"] = openai_model
                results["success"] = True
                logger.info(f"OpenAI synthetic data generated at {openai_model}")
        
        if use_claude:
            logger.info("Generating synthetic training data using Claude API...")
            trainer.use_api_model(openai=False, claude=True)
            claude_model = trainer.generate_synthetic_data(
                data_file=training_data_path,
                output_name=model_name + "_claude" if model_name else f"synthetic_claude_{int(time.time())}"
            )
            if claude_model:
                results["claude_model"] = claude_model
                results["success"] = True
                logger.info(f"Claude synthetic data generated at {claude_model}")
        
        # If not using APIs, train a local model
        if not use_openai and not use_claude:
            logger.info(f"Preparing training data from {training_data_path}...")
            trainer.prepare_training_data(data_file=training_data_path)
            
            # Train the model
            logger.info(f"Training local model for {epochs} epochs with batch size {batch_size}...")
            local_model = trainer.train_model(
                epochs=epochs,
                batch_size=batch_size,
                model_name=model_name or f"pose_llm_{int(time.time())}"
            )
            
            if local_model:
                results["local_model"] = local_model
                results["success"] = True
                logger.info(f"Local model training complete. Model saved to {local_model}")
        
        return results


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
    
    # Resource management options
    parser.add_argument('--memory_limit', type=float, default=DEFAULT_MEMORY_LIMIT,
                       help=f'Memory limit as a fraction of total system memory (default: {DEFAULT_MEMORY_LIMIT})')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes/threads (default: auto)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for frame processing (default: {DEFAULT_BATCH_SIZE})')
    
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
        if result.get("pose_model_path"):
            processed_models.append(result["pose_model_path"])
            logger.info(f"Processed {result['frame_count']}/{result['total_frames']} frames in {result['processing_time']:.2f} seconds")
    else:
        # Process all videos in folder
        logger.info(f"Processing all videos in: {args.input}")
        results = pipeline.process_all_videos(args.input)
        processed_models = [r["pose_model_path"] for r in results if r.get("pose_model_path")]
        
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