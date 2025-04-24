import cv2
import os
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class VideoDataExtractor:
    """
    Extracts pose data and metrics from processed videos for use in the RAG system.
    This class creates a structured data representation from the processed video files.
    """
    def __init__(self, output_dir="data_store"):
        """Initialize the data extractor with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe pose detector for re-extracting pose data if needed
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_video_data(self, video_path, sample_rate=5, reprocess=False):
        """
        Extract pose data from processed video at the given sample rate.
        
        Args:
            video_path (str): Path to the processed video
            sample_rate (int): Sample 1 frame every N frames
            reprocess (bool): Whether to reprocess with MediaPipe or use existing annotations
            
        Returns:
            dict: Dictionary containing video metadata and frame data
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # If data exists and we don't want to reprocess, load it
        data_file = self.output_dir / f"{video_name}_data.json"
        if data_file.exists() and not reprocess:
            print(f"Loading existing data for {video_name}")
            with open(data_file, 'r') as f:
                return json.load(f)
        
        print(f"Extracting data from {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize MediaPipe Pose if reprocessing
        pose = None
        if reprocess:
            pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Prepare data structure
        video_data = {
            "metadata": {
                "filename": video_path.name,
                "original_name": video_name.replace("annotated_", ""),
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "processed_date": datetime.now().isoformat(),
                "sampled_frames": 0
            },
            "frames": []
        }
        
        # Process frames
        frame_idx = 0
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    frame_data = self._process_frame(frame, frame_idx, pose)
                    if frame_data:
                        video_data["frames"].append(frame_data)
                        video_data["metadata"]["sampled_frames"] += 1
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Save data to file
        with open(data_file, 'w') as f:
            json.dump(video_data, f, indent=2)
        
        print(f"Extracted data from {video_data['metadata']['sampled_frames']} frames")
        return video_data
    
    def _process_frame(self, frame, frame_idx, pose=None):
        """
        Process a single frame to extract pose data and metrics.
        
        Args:
            frame (numpy.ndarray): Frame image
            frame_idx (int): Frame index
            pose (mediapipe.solutions.pose.Pose): MediaPipe pose detector
            
        Returns:
            dict: Extracted data from the frame
        """
        # If pose detector is provided, extract pose landmarks
        landmarks = None
        metrics = {}
        
        if pose:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })
        
        # Try to extract metrics from text on the frame
        metrics = self._extract_metrics_from_frame(frame)
        
        # Return frame data
        return {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / 30.0,  # Approximate timestamp based on 30fps
            "landmarks": landmarks,
            "metrics": metrics
        }
    
    def _extract_metrics_from_frame(self, frame):
        """
        Extract metrics from text annotations on the frame.
        This is a simple OCR-like function to get the metrics that were
        drawn on the processed video frames.
        
        Args:
            frame (numpy.ndarray): Frame image
            
        Returns:
            dict: Extracted metrics
        """
        # This is a placeholder for actual OCR
        # In a real implementation, you'd use OCR or other techniques 
        # to extract metrics from annotations on the frame
        
        # For now, we'll just try to detect light regions in the top left 
        # which is where metrics are typically drawn
        metrics = {}
        
        # Extract top-left region where metrics are usually displayed
        top_left = frame[10:200, 10:300]
        
        # Simple detection of white text regions 
        # (this is a very naive approach and would need to be replaced with proper OCR)
        gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # If we have white pixels in this region, assume there are metrics
        if np.sum(thresh) > 1000:
            metrics["has_annotations"] = True
        
        return metrics
    
    def extract_all_videos(self, video_dir="output", sample_rate=5):
        """
        Extract data from all processed videos in the given directory.
        
        Args:
            video_dir (str): Directory containing processed videos
            sample_rate (int): Sample 1 frame every N frames
            
        Returns:
            list: List of video data dictionaries
        """
        video_dir = Path(video_dir)
        videos = list(video_dir.glob("annotated_*.mp4"))
        
        all_data = []
        for video_path in videos:
            video_data = self.extract_video_data(video_path, sample_rate)
            all_data.append(video_data)
        
        # Save a catalog file with all video information
        catalog = {
            "updated_at": datetime.now().isoformat(),
            "total_videos": len(all_data),
            "videos": [data["metadata"] for data in all_data]
        }
        
        with open(self.output_dir / "video_catalog.json", 'w') as f:
            json.dump(catalog, f, indent=2)
        
        return all_data


if __name__ == "__main__":
    # Simple test
    extractor = VideoDataExtractor()
    data = extractor.extract_all_videos(sample_rate=10)
    print(f"Extracted data from {len(data)} videos") 