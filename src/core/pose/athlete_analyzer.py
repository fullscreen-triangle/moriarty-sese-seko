import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Set
import logging
from dataclasses import dataclass, field
import json
import pickle
import os
import time
from collections import defaultdict

from .multi_person_tracker import MultiPersonTracker, TrackedPerson
from .keypoints import PoseData
from ..dynamics.ai_dynamics_analyzer import AIDynamicsAnalyzer
from ..dynamics.dynamics_analyzer import DynamicsAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class AthleteData:
    """Class for storing athlete data with biomechanical analysis"""
    id: str
    name: str = ""
    pose_data: Optional[PoseData] = None
    biomechanical_metrics: Dict[str, Any] = field(default_factory=dict)
    technical_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_performance(self, performance_data: Dict[str, Any]):
        """Add a performance entry to the history"""
        # Add current timestamp if not provided
        if "date" not in performance_data:
            performance_data["date"] = time.strftime("%Y-%m-%d")
            
        self.performance_history.append(performance_data)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert athlete data to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "biomechanical_metrics": self.biomechanical_metrics,
            "technical_analysis": self.technical_analysis,
            "performance_history": self.performance_history,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AthleteData':
        """Create AthleteData from dictionary"""
        athlete = cls(
            id=data["id"],
            name=data.get("name", ""),
        )
        
        athlete.biomechanical_metrics = data.get("biomechanical_metrics", {})
        athlete.technical_analysis = data.get("technical_analysis", {})
        athlete.performance_history = data.get("performance_history", [])
        athlete.metadata = data.get("metadata", {})
        
        return athlete


class AthleteAnalyzer:
    """
    Class for analyzing multiple athletes in videos, with support for 
    tracking, biomechanical analysis, and performance comparison.
    """
    
    def __init__(self, 
                use_gpu: bool = True,
                use_advanced_analysis: bool = True,
                cache_dir: str = "cache/athletes"):
        """
        Initialize the athlete analyzer.
        
        Args:
            use_gpu: Whether to use GPU for processing
            use_advanced_analysis: Whether to use AI-powered advanced analysis
            cache_dir: Directory for caching athlete data
        """
        self.use_gpu = use_gpu
        self.use_advanced_analysis = use_advanced_analysis
        
        # Initialize the multi-person tracker
        self.tracker = MultiPersonTracker(use_gpu=use_gpu)
        
        # Initialize dynamics analyzer
        if use_advanced_analysis:
            self.dynamics_analyzer = AIDynamicsAnalyzer(use_parallel=True)
        else:
            self.dynamics_analyzer = DynamicsAnalyzer(use_parallel=True)
        
        # Dictionary to store athlete data
        self.athletes: Dict[str, AthleteData] = {}
        
        # Cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_video(self, 
                     video_path: str, 
                     athlete_identities: Optional[Dict[str, str]] = None,
                     start_frame: int = 0,
                     max_frames: Optional[int] = None) -> Dict[str, AthleteData]:
        """
        Process a video to track and analyze multiple athletes.
        
        Args:
            video_path: Path to the video
            athlete_identities: Optional mapping of track IDs to athlete names
            start_frame: Frame to start from
            max_frames: Maximum frames to process
            
        Returns:
            Dictionary mapping athlete IDs to AthleteData objects
        """
        video_path = Path(video_path)
        cache_file = self.cache_dir / f"{video_path.stem}_athletes.pkl"
        
        # Check for cached results
        if cache_file.exists():
            logger.info(f"Loading cached athlete analysis from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.athletes = pickle.load(f)
            return self.athletes
        
        # Track people in the video
        logger.info(f"Tracking people in video: {video_path.name}")
        tracks = self.tracker.process_video(
            video_path=str(video_path),
            start_frame=start_frame,
            max_frames=max_frames,
            save_results=True
        )
        
        # Convert tracks to PoseData for biomechanical analysis
        pose_data_dict = self.tracker.get_pose_data()
        
        # Analyze each person
        logger.info(f"Analyzing {len(tracks)} people from the video")
        for track_id, pose_data in pose_data_dict.items():
            # Create athlete ID (use provided identity if available)
            athlete_id = track_id
            athlete_name = athlete_identities.get(track_id, f"Athlete {track_id}") if athlete_identities else f"Athlete {track_id}"
            
            # Create athlete data object
            athlete = AthleteData(
                id=athlete_id,
                name=athlete_name,
                pose_data=pose_data
            )
            
            # Analyze biomechanics
            self._analyze_athlete_biomechanics(athlete, tracks[track_id])
            
            # Store athlete data
            self.athletes[athlete_id] = athlete
        
        # Save results to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.athletes, f)
        
        return self.athletes
    
    def compare_athletes(self, 
                        athlete_ids: List[str], 
                        metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple athletes based on their biomechanical metrics.
        
        Args:
            athlete_ids: List of athlete IDs to compare
            metrics: Optional list of specific metrics to compare
            
        Returns:
            Comparison results
        """
        if not athlete_ids or len(athlete_ids) < 2:
            raise ValueError("At least two athletes are required for comparison")
        
        # Ensure all athletes exist
        for athlete_id in athlete_ids:
            if athlete_id not in self.athletes:
                raise ValueError(f"Athlete with ID {athlete_id} not found")
        
        # Get athletes to compare
        athletes_to_compare = [self.athletes[aid] for aid in athlete_ids]
        
        # Default metrics if not specified
        if metrics is None:
            metrics = [
                "stride.length", 
                "stride.cadence", 
                "stride.ground_contact_time",
                "kinematics.joint_angles.knee.extension", 
                "kinematics.joint_angles.hip.extension",
                "forces.peak.foot",
                "efficiency.coordination_index",
                "efficiency.force_application_efficiency"
            ]
        
        # Extract metrics for each athlete
        comparison_data = {}
        for athlete in athletes_to_compare:
            athlete_metrics = {}
            
            # Extract each requested metric from biomechanical_metrics
            for metric_path in metrics:
                parts = metric_path.split('.')
                value = athlete.biomechanical_metrics
                
                try:
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    
                    athlete_metrics[metric_path] = value
                except (KeyError, TypeError):
                    athlete_metrics[metric_path] = None
            
            comparison_data[athlete.id] = {
                "name": athlete.name,
                "metrics": athlete_metrics
            }
        
        # Use AI-powered comparison if available
        if self.use_advanced_analysis:
            comparison_results = self._advanced_athlete_comparison(comparison_data)
        else:
            # Basic comparison (just collect the data without analysis)
            comparison_results = {
                "data": comparison_data,
                "analysis": {
                    "summary": "Basic comparison data collected. Enable advanced analysis for detailed insights."
                }
            }
        
        return comparison_results
    
    def get_athlete_data(self, athlete_id: str) -> Optional[AthleteData]:
        """Get data for a specific athlete"""
        return self.athletes.get(athlete_id)
    
    def save_athlete_data(self, output_dir: str = "output/athletes"):
        """Save all athlete data to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for athlete_id, athlete in self.athletes.items():
            athlete_file = output_path / f"{athlete_id}.json"
            
            with open(athlete_file, 'w') as f:
                json.dump(athlete.to_dict(), f, indent=2)
            
            logger.info(f"Saved athlete data to {athlete_file}")
    
    def load_athlete_data(self, input_dir: str = "output/athletes"):
        """Load athlete data from JSON files"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.warning(f"Athlete data directory {input_path} does not exist")
            return
        
        athlete_files = list(input_path.glob("*.json"))
        
        for athlete_file in athlete_files:
            try:
                with open(athlete_file, 'r') as f:
                    athlete_dict = json.load(f)
                
                athlete = AthleteData.from_dict(athlete_dict)
                self.athletes[athlete.id] = athlete
                
                logger.info(f"Loaded athlete data from {athlete_file}")
            except Exception as e:
                logger.error(f"Error loading athlete data from {athlete_file}: {str(e)}")
    
    def add_performance_data(self, 
                            athlete_id: str, 
                            performance_data: Dict[str, Any],
                            analyze: bool = True) -> Optional[Dict[str, Any]]:
        """
        Add performance data for an athlete and optionally analyze improvement.
        
        Args:
            athlete_id: ID of the athlete
            performance_data: Performance data to add
            analyze: Whether to analyze improvement compared to previous performances
            
        Returns:
            Improvement analysis if analyze=True, None otherwise
        """
        if athlete_id not in self.athletes:
            raise ValueError(f"Athlete with ID {athlete_id} not found")
        
        athlete = self.athletes[athlete_id]
        
        # Add new performance data
        athlete.add_performance(performance_data)
        
        # Analyze improvement if requested and possible
        if analyze and len(athlete.performance_history) > 1 and self.use_advanced_analysis:
            current_performance = athlete.performance_history[-1]
            previous_performances = athlete.performance_history[:-1]
            
            improvement_analysis = self.dynamics_analyzer.compare_performances(
                current_data=current_performance,
                previous_data=previous_performances
            )
            
            return improvement_analysis
        
        return None
    
    def visualize_multi_athlete_tracking(self, 
                                        video_path: str, 
                                        output_path: str,
                                        start_frame: int = 0,
                                        max_frames: Optional[int] = None,
                                        show_metrics: bool = True):
        """
        Create a visualization of multi-athlete tracking with metrics.
        
        Args:
            video_path: Path to the input video
            output_path: Path for the output video
            start_frame: Frame to start from
            max_frames: Maximum frames to process
            show_metrics: Whether to show biomechanical metrics on the video
        """
        video_path = Path(video_path)
        
        # Ensure tracking has been done
        if not self.athletes:
            logger.info("No athletes tracked yet, processing video first")
            self.process_video(
                video_path=str(video_path),
                start_frame=start_frame,
                max_frames=max_frames
            )
        
        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set maximum frames if not specified
        if max_frames is None:
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set frame position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_idx = start_frame
        frames_processed = 0
        
        logger.info(f"Creating visualization for {video_path.name}")
        
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Visualize tracks
            vis_frame = self.tracker.visualize_tracks(frame, frame_idx)
            
            # Add metrics if requested
            if show_metrics:
                vis_frame = self._add_metrics_to_frame(vis_frame, frame_idx)
            
            # Write frame to output video
            out.write(vis_frame)
            
            # Update counters
            frame_idx += 1
            frames_processed += 1
            
            # Log progress
            if frames_processed % 100 == 0:
                logger.info(f"Processed {frames_processed} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def _analyze_athlete_biomechanics(self, athlete: AthleteData, tracked_person: TrackedPerson):
        """
        Perform biomechanical analysis for an athlete.
        
        Args:
            athlete: AthleteData object to update
            tracked_person: Tracked person data
        """
        try:
            # Extract keypoints from tracked person
            keypoints_frames = sorted(tracked_person.keypoints.keys())
            
            if not keypoints_frames:
                logger.warning(f"No keypoints found for athlete {athlete.id}")
                return
            
            # Prepare data for analysis
            positions_batch = []
            velocities_batch = []  # Will be calculated from positions
            accelerations_batch = []  # Will be calculated from velocities
            
            # Get positions for each frame
            for frame_idx in keypoints_frames:
                kpts = tracked_person.keypoints[frame_idx]
                
                # Convert keypoints to positions in the format expected by DynamicsAnalyzer
                positions = {
                    'foot': np.array([kpts[15][0], kpts[15][1]]) if kpts.shape[0] > 15 else np.zeros(2),
                    'shank': np.array([kpts[13][0], kpts[13][1]]) if kpts.shape[0] > 13 else np.zeros(2),
                    'thigh': np.array([kpts[11][0], kpts[11][1]]) if kpts.shape[0] > 11 else np.zeros(2)
                }
                
                positions_batch.append(positions)
            
            # Calculate velocities and accelerations (simplified)
            for i in range(len(positions_batch)):
                if i == 0:
                    velocities_batch.append({k: np.zeros_like(v) for k, v in positions_batch[i].items()})
                else:
                    velocities = {}
                    for k in positions_batch[i].keys():
                        velocities[k] = positions_batch[i][k] - positions_batch[i-1][k]
                    velocities_batch.append(velocities)
                
                if i < 2:
                    accelerations_batch.append({k: np.zeros_like(v) for k, v in positions_batch[i].items()})
                else:
                    accelerations = {}
                    for k in velocities_batch[i].keys():
                        accelerations[k] = velocities_batch[i][k] - velocities_batch[i-1][k]
                    accelerations_batch.append(accelerations)
            
            # Create athlete info
            athlete_info = {
                "name": athlete.name,
                "id": athlete.id,
                "sport": "sprint",  # Default assumption, could be configurable
                "level": "elite"  # Default assumption, could be configurable
            }
            
            # Run biomechanical analysis
            biomech_results = self.dynamics_analyzer.analyze_biomechanics(
                positions_batch=positions_batch,
                velocities_batch=velocities_batch,
                accelerations_batch=accelerations_batch,
                athlete_info=athlete_info
            )
            
            # Store results in athlete data
            athlete.biomechanical_metrics = biomech_results.get("metrics", {})
            
            # Generate technical analysis report if using advanced analysis
            if self.use_advanced_analysis:
                technical_report = self.dynamics_analyzer.generate_technical_report(
                    analysis_results=biomech_results,
                    athlete_profile=athlete_info
                )
                athlete.technical_analysis = technical_report
            
            # Create a performance data record for this analysis
            performance_data = {
                "date": time.strftime("%Y-%m-%d"),
                "sport": "sprint",
                "sprint_times": {},  # These would come from actual timing data
                "stride_metrics": athlete.biomechanical_metrics.get("stride", {}),
                "force_metrics": athlete.biomechanical_metrics.get("forces", {})
            }
            
            # Add to performance history
            athlete.add_performance(performance_data)
            
            logger.info(f"Completed biomechanical analysis for athlete {athlete.id}")
            
        except Exception as e:
            logger.error(f"Error in biomechanical analysis for athlete {athlete.id}: {str(e)}")
    
    def _advanced_athlete_comparison(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced AI-powered comparison between athletes.
        
        Args:
            comparison_data: Data to compare
            
        Returns:
            Comparison results with AI-powered analysis
        """
        try:
            # Format the data for the AI model
            athlete_metrics = {}
            
            for athlete_id, data in comparison_data.items():
                athlete_metrics[athlete_id] = {
                    "name": data["name"],
                    "metrics": data["metrics"]
                }
            
            # Use the AI client for comparison
            prompt = f"""
            Compare the biomechanical metrics of these athletes:
            
            {json.dumps(athlete_metrics, indent=2)}
            
            Provide a comprehensive analysis covering:
            1. Key strengths of each athlete
            2. Areas where each athlete could improve
            3. Technical differences between athletes
            4. Specific recommendations for each athlete
            
            Include numerical comparisons and context for why certain metrics are important.
            """
            
            # Use the AI client directly
            analysis = self.dynamics_analyzer.ai_client.analyze_sync(
                analysis_type="coaching_insights",
                data=athlete_metrics,
                coach_context={"comparison_mode": True}
            )
            
            # Combine the analysis with the raw data
            result = {
                "data": comparison_data,
                "analysis": analysis
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced athlete comparison: {str(e)}")
            return {
                "data": comparison_data,
                "error": str(e),
                "analysis": {
                    "summary": "Error performing advanced comparison."
                }
            }
    
    def _add_metrics_to_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Add biomechanical metrics to visualization frame.
        
        Args:
            frame: Frame to add metrics to
            frame_idx: Current frame index
            
        Returns:
            Frame with added metrics
        """
        output = frame.copy()
        
        # Add a semi-transparent background for better text visibility
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (300, 30 + 20 * len(self.athletes)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        
        # Add title
        cv2.putText(output, "Athlete Metrics", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add metrics for each athlete
        y_pos = 50
        for athlete_id, athlete in self.athletes.items():
            # Get tracked person
            tracked_person = self.tracker.tracks.get(athlete_id)
            if not tracked_person or tracked_person.missing_duration(frame_idx) > 5:
                continue
                
            # Get color associated with this track
            color_hash = hash(athlete_id) % 0xFFFFFF
            color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
            
            # Add athlete name
            cv2.putText(output, f"{athlete.name}", (15, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check if we have stride metrics
            stride_metrics = athlete.biomechanical_metrics.get("stride", {})
            if stride_metrics:
                stride_length = stride_metrics.get("length", 0)
                stride_cadence = stride_metrics.get("cadence", 0)
                y_pos += 20
                cv2.putText(output, f"Stride: {stride_length:.2f}m @ {stride_cadence:.1f}Hz", 
                          (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Add more metrics as needed
            y_pos += 30
        
        return output 