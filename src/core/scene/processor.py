import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import ray
import multiprocessing as multiproc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Any
from .analyzer import PoseAnalyzer
from .metrics import MetricsCalculator
import os
import time
import json
import logging

logger = logging.getLogger(__name__)

@ray.remote
class DistributedPoseAnalyzer(PoseAnalyzer):
    """Distributed version of PoseAnalyzer using Ray."""
    pass

class VideoProcessor:
    """
    Handles video frame extraction, preprocessing, and saving for later analysis.
    Supports incremental processing to avoid reprocessing videos.
    """
    
    def __init__(self, model_path=None, n_workers=None, output_dir=None):
        """Initialize the video processor with MediaPipe pose model."""
        # Initialize Ray for distributed computing
        ray.init(ignore_reinit_error=True)
        
        self.n_workers = n_workers or multiproc.cpu_count()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create distributed analyzers
        self.analyzers = [DistributedPoseAnalyzer.remote() for _ in range(self.n_workers)]
        self.metrics = MetricsCalculator.remote()  # Initialize as a Ray actor
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir) if output_dir else Path("output/video_frames")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create metadata directory for tracking processed videos
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load custom model if provided
        self.custom_model = None
        if model_path:
            self.load_custom_model(model_path)

    def process_video(self, video_path: str, start_frame: int = 0,
                      max_frames: int = None, batch_size: int = 10,
                      resize_dim: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Process video and extract frames. Supports incremental processing.
        
        Args:
            video_path: Path to the video file
            start_frame: Frame to start processing from (for resuming)
            max_frames: Maximum number of frames to process
            batch_size: Number of frames to process in each batch
            resize_dim: Dimensions to resize frames to (width, height)
            
        Returns:
            Dict with processing metadata and output paths
        """
        video_path = Path(video_path)
        
        # Check if this video has been processed before
        metadata_file = self.metadata_dir / f"{video_path.stem}_metadata.json"
        previous_metadata = self._load_metadata(metadata_file)
        
        # Determine the video output directory
        video_output_dir = self.output_dir / video_path.stem
        video_output_dir.mkdir(exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If we have previous metadata, check if we need to resume
        if previous_metadata:
            # Check if the video properties match to ensure it's the same video
            if (previous_metadata.get('fps') == fps and 
                previous_metadata.get('frame_width') == frame_width and
                previous_metadata.get('frame_height') == frame_height):
                
                # Get the last processed frame
                processed_frames = previous_metadata.get('processed_frames', 0)
                
                # If start_frame is specified, use it, otherwise resume from last processed frame
                if start_frame <= 0:
                    start_frame = processed_frames
                
                logger.info(f"Resuming video processing from frame {start_frame} of {total_frames}")
            else:
                logger.warning("Video properties don't match previous processing. Starting from beginning.")
                start_frame = 0
        
        # Determine how many frames to process
        if max_frames:
            end_frame = min(start_frame + max_frames, total_frames)
        else:
            end_frame = total_frames
        
        # Set the frame position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames in batches
        current_frame = start_frame
        frames_processed = 0
        frame_paths = []
        
        logger.info(f"Processing video: {video_path.name}, frames {start_frame}-{end_frame}")
        
        while current_frame < end_frame:
            batch_end = min(current_frame + batch_size, end_frame)
            batch_frames = []
            batch_indices = []
            
            # Read a batch of frames
            for i in range(current_frame, batch_end):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if resize_dim:
                    frame = cv2.resize(frame, resize_dim)
                
                batch_frames.append(frame)
                batch_indices.append(i)
            
            # Process and save the batch
            if batch_frames:
                self._save_batch(batch_frames, batch_indices, video_output_dir)
                frame_paths.extend([str(video_output_dir / f"frame_{idx:06d}.jpg") for idx in batch_indices])
                frames_processed += len(batch_frames)
                
                # Log progress
                if frames_processed % 100 == 0:
                    logger.info(f"Processed {frames_processed} frames ({current_frame}/{end_frame})")
            
            # Update current frame position
            current_frame = batch_end
            
            # Update metadata periodically to allow for resuming if interrupted
            if frames_processed % 1000 == 0:
                self._save_metadata(metadata_file, {
                    'video_path': str(video_path),
                    'fps': fps,
                    'frame_width': frame_width,
                    'frame_height': frame_height,
                    'total_frames': total_frames,
                    'processed_frames': current_frame,
                    'last_updated': time.time()
                })
        
        # Release resources
        cap.release()
        
        # Create final metadata
        metadata = {
            'video_path': str(video_path),
            'output_dir': str(video_output_dir),
            'fps': fps,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'total_frames': total_frames,
            'processed_frames': current_frame,
            'frames_processed_this_run': frames_processed,
            'frame_paths': frame_paths,
            'last_updated': time.time()
        }
        
        # Save the metadata
        self._save_metadata(metadata_file, metadata)
        
        logger.info(f"Completed processing {frames_processed} frames from {video_path.name}")
        
        return metadata
    
    def _save_batch(self, frames, indices, output_dir):
        """Save a batch of frames as images."""
        for i, frame in zip(indices, frames):
            output_path = output_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
    
    def _save_metadata(self, metadata_file, metadata):
        """Save processing metadata to a JSON file."""
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, metadata_file):
        """Load processing metadata from a JSON file."""
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse metadata file: {metadata_file}")
                return None
        return None
    
    def get_processed_videos(self):
        """Get a list of videos that have been processed."""
        metadata_files = list(self.metadata_dir.glob("*_metadata.json"))
        processed_videos = []
        
        for metadata_file in metadata_files:
            metadata = self._load_metadata(metadata_file)
            if metadata:
                processed_videos.append({
                    'video_path': metadata.get('video_path'),
                    'frames_processed': metadata.get('processed_frames', 0),
                    'total_frames': metadata.get('total_frames', 0),
                    'last_updated': metadata.get('last_updated')
                })
        
        return processed_videos
    
    def _prepare_frame_batches(self, cap, total_frames) -> List[List[np.ndarray]]:
        """Prepare batches of frames for parallel processing."""
        batch_size = max(1, total_frames // (self.n_workers * 4))  # Adjust batch size based on workers
        frames = []
        current_batch = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            current_batch.append(frame)
            if len(current_batch) >= batch_size:
                frames.append(current_batch)
                current_batch = []
        
        if current_batch:
            frames.append(current_batch)
            
        return frames
    
    def _process_frame_batches(self, frame_batches: List[List[np.ndarray]]) -> List[Tuple[np.ndarray, Dict]]:
        """Process batches of frames in parallel using Ray."""
        # Distribute batches across workers
        future_results = []
        for batch_idx, batch in enumerate(frame_batches):
            analyzer = self.analyzers[batch_idx % len(self.analyzers)]
            future_results.append(self._process_batch.remote(self, analyzer, batch))
        
        # Gather results
        processed_frames = []
        for future_batch in ray.get(future_results):
            processed_frames.extend(future_batch)
        
        return processed_frames
    
    @ray.remote
    def _process_batch(self, analyzer, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, Dict]]:
        """Process a batch of frames using a distributed analyzer."""
        results = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            
            if pose_results.pose_landmarks:
                # Get pose analysis (distributed)
                pose_data = ray.get(analyzer.analyze_pose.remote(pose_results.pose_landmarks))
                
                # Calculate metrics (using distributed metrics calculator)
                metrics = ray.get(self.metrics.calculate_metrics.remote(pose_data))
                
                # Draw annotations
                annotated_frame = self._draw_annotations(frame, pose_results, metrics)
                
                results.append((annotated_frame, pose_data))
            else:
                results.append((frame, None))
        
        return results
    
    def _draw_annotations(self, frame, results, metrics):
        """Draw all annotations on the frame."""
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Draw metrics
        y_pos = 30
        for metric, value in metrics.items():
            cv2.putText(frame, f"{metric}: {value:.2f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 30
            
        return frame
    
    def _update_model(self, frame_data):
        """Update the custom model with new frame data using distributed training."""
        if self.custom_model:
            # Implement distributed model training using Ray
            pass
    
    def load_custom_model(self, model_path):
        """Load a custom model for additional analysis."""
        # Implement distributed model loading
        pass 