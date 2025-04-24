import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
import time
from scipy.optimize import linear_sum_assignment
import pickle
import os

from ..pose.human_detector import HumanDetector, DetectedPerson
from ..pose.pose_detector import PoseDetector
from ..pose.keypoints import PoseData, PoseKeypoint

logger = logging.getLogger(__name__)

@dataclass
class TrackedPerson:
    """Class for tracking a person across frames"""
    id: str
    bboxes: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)  # frame_idx -> bbox
    keypoints: Dict[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> keypoints
    confidences: Dict[int, float] = field(default_factory=dict)  # frame_idx -> detection confidence
    trajectory: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # frame_idx -> center position
    last_seen: int = -1
    first_seen: int = -1
    
    # Additional attributes for biomechanical analysis
    biomechanical_metrics: Dict[str, Dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    
    def update(self, frame_idx: int, detection: DetectedPerson, keypoints: Optional[np.ndarray] = None):
        """Update the tracked person with new detection"""
        if self.first_seen == -1:
            self.first_seen = frame_idx
            
        self.bboxes[frame_idx] = detection.bbox
        self.confidences[frame_idx] = detection.confidence
        self.trajectory[frame_idx] = detection.center
        self.last_seen = frame_idx
        
        if keypoints is not None:
            self.keypoints[frame_idx] = keypoints
    
    def add_biomechanical_metric(self, metric_name: str, frame_idx: int, value: float):
        """Add a biomechanical metric for a specific frame"""
        self.biomechanical_metrics[metric_name][frame_idx] = value
    
    def get_trajectory(self) -> np.ndarray:
        """Get person trajectory as numpy array"""
        frames = sorted(self.trajectory.keys())
        return np.array([self.trajectory[f] for f in frames])
    
    def missing_duration(self, current_frame: int) -> int:
        """Calculate for how many frames the person hasn't been seen"""
        return current_frame - self.last_seen if self.last_seen >= 0 else float('inf')
    
    def to_pose_data(self) -> PoseData:
        """Convert tracked keypoints to PoseData format for analysis"""
        pose_data = PoseData()
        
        for frame_idx, kpts in self.keypoints.items():
            person_id = int(self.id.split('-')[1]) if '-' in self.id else 0
            
            # Reshape keypoints to match expected format if necessary
            if len(kpts.shape) == 2 and kpts.shape[1] == 3:  # [num_keypoints, 3] format
                # Add z dimension with zeros
                kpts_with_z = np.zeros((kpts.shape[0], 4), dtype=np.float32)
                kpts_with_z[:, 0] = kpts[:, 0]  # x
                kpts_with_z[:, 1] = kpts[:, 1]  # y
                kpts_with_z[:, 3] = kpts[:, 2]  # confidence
                kpts = kpts_with_z
            
            # Add to pose data
            pose_data.add_pose(
                frame_idx=frame_idx,
                person_id=person_id,
                keypoints=kpts,
                timestamp=float(frame_idx)  # Using frame index as timestamp
            )
        
        return pose_data


class MultiPersonTracker:
    """
    Enhanced tracker that can detect and track multiple people across video frames.
    Maintains identity across frames and supports occlusion handling.
    """
    
    def __init__(self, 
                confidence_threshold: float = 0.5, 
                iou_threshold: float = 0.5,
                max_lost_frames: int = 30,
                use_gpu: bool = True):
        """
        Initialize the multi-person tracker.
        
        Args:
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for matching tracks
            max_lost_frames: Maximum number of frames a track can be lost before being removed
            use_gpu: Whether to use GPU for processing
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info(f"Using device: {self.device} for multi-person tracking")
        
        # Initialize detectors
        self.human_detector = HumanDetector(confidence_threshold=confidence_threshold, device=self.device)
        self.pose_detector = PoseDetector(confidence_threshold=confidence_threshold)
        
        # Initialize tracks
        self.tracks: Dict[str, TrackedPerson] = {}
        
        # Track ID counter
        self.next_id = 0
        
        # Cache directory for storing tracking results
        self.cache_dir = Path("cache/tracking")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_video(self, 
                     video_path: str, 
                     start_frame: int = 0,
                     max_frames: Optional[int] = None,
                     save_results: bool = True) -> Dict[str, TrackedPerson]:
        """
        Process a video file and track all people.
        
        Args:
            video_path: Path to the video file
            start_frame: Frame to start processing from
            max_frames: Maximum number of frames to process
            save_results: Whether to save tracking results to cache
            
        Returns:
            Dictionary mapping track IDs to TrackedPerson objects
        """
        video_path = Path(video_path)
        cache_file = self.cache_dir / f"{video_path.stem}_tracking.pkl"
        
        # Check if we have cached results
        if cache_file.exists():
            logger.info(f"Loading cached tracking results from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.tracks = pickle.load(f)
            return self.tracks
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set maximum frames if not specified
        if max_frames is None:
            max_frames = total_frames - start_frame
        
        # Set frame position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_idx = start_frame
        frames_processed = 0
        
        logger.info(f"Processing video: {video_path.name}, frames {start_frame}-{start_frame + max_frames}")
        
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            self.process_frame(frame, frame_idx)
            
            # Update counters
            frame_idx += 1
            frames_processed += 1
            
            # Log progress
            if frames_processed % 100 == 0:
                logger.info(f"Processed {frames_processed} frames")
        
        # Release resources
        cap.release()
        
        # Save results
        if save_results:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.tracks, f)
            logger.info(f"Saved tracking results to {cache_file}")
        
        return self.tracks
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, DetectedPerson]:
        """
        Process a single frame and update tracks.
        
        Args:
            frame: The frame to process
            frame_idx: The frame index
            
        Returns:
            Dictionary mapping track IDs to detected people
        """
        # Detect humans
        detections, distances = self.human_detector.detect_humans(frame)
        
        # Detect poses for each person
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            person_frame = frame[y1:y2, x1:x2]
            if person_frame.size == 0:
                continue
            
            # Detect pose
            keypoints = self.pose_detector.detect(person_frame)
            if keypoints is not None and len(keypoints) > 0:
                # Adjust keypoint coordinates to global frame
                keypoints[0, :, 0] += x1
                keypoints[0, :, 1] += y1
                detection.keypoints = keypoints[0]  # First person's keypoints
        
        # Track detected people
        self._update_tracks(detections, frame_idx)
        
        # Return the current detections with track IDs
        current_detections = {}
        for track_id, track in self.tracks.items():
            if track.last_seen == frame_idx:
                # Get the latest detection
                bbox = track.bboxes[frame_idx]
                confidence = track.confidences[frame_idx]
                center = track.trajectory[frame_idx]
                
                detection = DetectedPerson(
                    bbox=bbox,
                    confidence=confidence,
                    center=center,
                    keypoints=track.keypoints.get(frame_idx)
                )
                
                current_detections[track_id] = detection
        
        return current_detections
    
    def _update_tracks(self, detections: List[DetectedPerson], frame_idx: int):
        """
        Update existing tracks with new detections.
        
        Args:
            detections: List of detected people
            frame_idx: The current frame index
        """
        # If no tracks exist, create new ones for all detections
        if not self.tracks:
            for detection in detections:
                track_id = f"person-{self.next_id}"
                self.next_id += 1
                
                track = TrackedPerson(id=track_id)
                track.update(frame_idx, detection, detection.keypoints)
                self.tracks[track_id] = track
            return
        
        # Match detections to existing tracks
        if detections and self.tracks:
            matched_tracks, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
                detections, frame_idx
            )
            
            # Update matched tracks
            for track_idx, detection_idx in matched_tracks:
                track_id = list(self.tracks.keys())[track_idx]
                track = self.tracks[track_id]
                detection = detections[detection_idx]
                
                track.update(frame_idx, detection, detection.keypoints)
            
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                detection = detections[detection_idx]
                
                track_id = f"person-{self.next_id}"
                self.next_id += 1
                
                track = TrackedPerson(id=track_id)
                track.update(frame_idx, detection, detection.keypoints)
                self.tracks[track_id] = track
        
        # Remove tracks that have been lost for too long
        self._clean_lost_tracks(frame_idx)
    
    def _match_detections_to_tracks(self, 
                                   detections: List[DetectedPerson], 
                                   frame_idx: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Args:
            detections: List of detected people
            frame_idx: Current frame index
            
        Returns:
            Tuple of (matched_tracks, unmatched_detections, unmatched_tracks)
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, (_, track) in enumerate(self.tracks.items()):
            if track.last_seen < 0:
                continue
                
            last_bbox = track.bboxes.get(track.last_seen)
            if not last_bbox:
                continue
                
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(last_bbox, detection.bbox)
        
        # Use the Hungarian algorithm to find the best matches
        matched_indices = []
        
        # Only match detections with IoU above threshold
        valid_matches = iou_matrix > self.iou_threshold
        
        # Find optimal matching
        if valid_matches.any():
            # For valid matches, use the Hungarian algorithm
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix * valid_matches)
            
            # Filter out matches below threshold
            for track_idx, detection_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, detection_idx] >= self.iou_threshold:
                    matched_indices.append((track_idx, detection_idx))
        
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(len(detections)) 
                               if not any(j == i for _, j in matched_indices)]
        
        unmatched_tracks = [i for i in range(len(self.tracks)) 
                           if not any(j == i for j, _ in matched_indices)]
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _clean_lost_tracks(self, current_frame: int):
        """Remove tracks that have been lost for too long."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.missing_duration(current_frame) > self.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def visualize_tracks(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Visualize tracks on a frame.
        
        Args:
            frame: Frame to visualize on
            frame_idx: Current frame index
            
        Returns:
            Frame with visualized tracks
        """
        output = frame.copy()
        
        # Draw each track
        for track_id, track in self.tracks.items():
            # Only visualize tracks that are currently visible or recently lost
            if track.missing_duration(frame_idx) <= 5:
                # Draw bounding box if available for this frame
                if frame_idx in track.bboxes:
                    x1, y1, x2, y2 = map(int, track.bboxes[frame_idx])
                    
                    # Different color for each track (using hash of track_id)
                    color_hash = hash(track_id) % 0xFFFFFF
                    color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
                    
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output, track_id, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw keypoints if available
                if frame_idx in track.keypoints:
                    keypoints = track.keypoints[frame_idx]
                    for i, (x, y, conf) in enumerate(keypoints):
                        if conf > 0.5:  # Only draw high-confidence keypoints
                            cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)
                
                # Draw trajectory
                trajectory = track.get_trajectory()
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = tuple(map(int, trajectory[i-1]))
                        pt2 = tuple(map(int, trajectory[i]))
                        cv2.line(output, pt1, pt2, (255, 0, 0), 2)
        
        return output
    
    def save_to_file(self, output_path: str):
        """Save tracks to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.tracks, f)
        
        logger.info(f"Saved {len(self.tracks)} tracks to {output_path}")
    
    @classmethod
    def load_from_file(cls, input_path: str) -> 'MultiPersonTracker':
        """Load tracks from a file."""
        tracker = cls()
        
        with open(input_path, 'rb') as f:
            tracker.tracks = pickle.load(f)
        
        logger.info(f"Loaded {len(tracker.tracks)} tracks from {input_path}")
        return tracker
    
    def get_pose_data(self) -> Dict[str, PoseData]:
        """
        Convert all tracks to PoseData objects for analysis.
        
        Returns:
            Dictionary mapping track IDs to PoseData objects
        """
        pose_data_dict = {}
        
        for track_id, track in self.tracks.items():
            pose_data = track.to_pose_data()
            pose_data_dict[track_id] = pose_data
        
        return pose_data_dict 