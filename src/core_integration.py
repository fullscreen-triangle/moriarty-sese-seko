"""
Core Module Integration for Moriarty Pipeline

This module demonstrates how to properly integrate the core modules
(pose, dynamics, motion, scene) into the pipeline workflow.
"""

import os
import sys
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import core modules
from src.core.pose.pose_detector import PoseDetector
from src.core.pose.human_detector import HumanDetector
from src.core.pose.skeleton import SkeletonDrawer
from src.core.pose.pose_visualizer import PoseVisualizer

from src.core.dynamics.kinematics_analyzer import KinematicsAnalyzer
from src.core.dynamics.stride_analyzer import StrideAnalyzer
from src.core.dynamics.sync_analyzer import SynchronizationAnalyzer
from src.core.dynamics.dynamics_analyzer import DynamicsAnalyzer
from src.core.dynamics.grf_analyzer import GRFAnalyzer

from src.core.motion.motion_classifier import MotionClassifier
from src.core.motion.movement_detector import MovementDetector
from src.core.motion.movement_tracker import MovementTracker
from src.core.motion.stabilography import StabilographyAnalyzer

from src.core.scene.scene_detector import SceneDetector
from src.core.scene.video_manager import VideoManager
from src.core.scene.processor import VideoProcessor
from src.core.scene.analyzer import SceneAnalyzer
from src.core.scene.metrics import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)

class CoreIntegrator:
    """
    Demonstrates how to properly integrate the core modules into the pipeline.
    This class should be used as a reference for updating the main pipeline.
    """
    
    def __init__(self, 
                fps: float = 30.0,
                output_dir: str = "output", 
                model_complexity: int = 2,
                device: str = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"):
        """
        Initialize all core module components.
        
        Args:
            fps: Frames per second of the video
            output_dir: Directory for output files
            model_complexity: Complexity of pose detection model
            device: Device to use for computation
        """
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.model_complexity = model_complexity
        self.device = device
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core modules
        self._init_pose_modules()
        self._init_dynamics_modules()
        self._init_motion_modules()
        self._init_scene_modules()
        
        logger.info("Core modules initialized successfully")
    
    def _init_pose_modules(self):
        """Initialize pose detection and processing modules."""
        # Human detection and pose estimation
        self.human_detector = HumanDetector(
            confidence_threshold=0.5,
            device=self.device
        )
        
        self.pose_detector = PoseDetector(
            model_path=None,  # Use default model
            confidence_threshold=0.5
        )
        
        # Skeleton visualization
        self.skeleton_drawer = SkeletonDrawer()
        
        # Visualization
        self.pose_visualizer = PoseVisualizer()
        
        logger.info("Pose modules initialized")
    
    def _init_dynamics_modules(self):
        """Initialize biomechanical dynamics modules."""
        # Kinematics analysis
        self.kinematics_analyzer = KinematicsAnalyzer(fps=self.fps)
        
        # Stride analysis
        self.stride_analyzer = StrideAnalyzer(fps=int(self.fps))
        
        # Synchronization analysis (for multiple athletes)
        self.sync_analyzer = SynchronizationAnalyzer(window_size=30)
        
        # General dynamics analyzer
        self.dynamics_analyzer = DynamicsAnalyzer()
        
        # Ground reaction force analyzer
        self.grf_analyzer = GRFAnalyzer(fps=self.fps)
        
        logger.info("Dynamics modules initialized")
    
    def _init_motion_modules(self):
        """Initialize motion tracking and analysis modules."""
        # Motion classification
        self.motion_classifier = MotionClassifier()
        
        # Movement detection and tracking
        self.movement_detector = MovementDetector()
        self.movement_tracker = MovementTracker()
        
        # Stabilography analysis
        self.stabilography_analyzer = StabilographyAnalyzer()
        
        logger.info("Motion modules initialized")
    
    def _init_scene_modules(self):
        """Initialize scene analysis and processing modules."""
        # Scene detection
        self.scene_detector = SceneDetector()
        
        # Video processing
        self.video_manager = VideoManager()
        self.video_processor = VideoProcessor()
        
        # Scene analysis
        self.scene_analyzer = SceneAnalyzer()
        
        # Metrics calculation
        self.metrics_calculator = MetricsCalculator()
        
        logger.info("Scene modules initialized")
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video using the proper integration of core modules.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with analysis results
        """
        # Use the video manager to prepare the video
        video_info = self.video_manager.load_video(video_path)
        self.fps = video_info["fps"]  # Update FPS from actual video
        
        logger.info(f"Processing video: {video_path} ({video_info['width']}x{video_info['height']} @ {self.fps} FPS)")
        
        # Process video frames
        results = self._process_frames(video_path)
        
        # Analyze results with advanced modules
        analysis_results = self._analyze_results(results)
        
        # Generate visualizations
        visualization_paths = self._generate_visualizations(video_path, results, analysis_results)
        
        return {
            "video_info": video_info,
            "analysis_results": analysis_results,
            "visualization_paths": visualization_paths
        }
    
    def _process_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process individual frames of the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of processed frame results
        """
        # Load the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        results = []
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # 1. Scene detection
                scene_info = self.scene_detector.detect_scene(frame)
                
                # 2. Human detection
                humans, distances = self.human_detector.detect_humans(frame)
                
                # Process each detected human
                frame_results = {
                    "frame_index": frame_index,
                    "timestamp": frame_index / self.fps,
                    "scene_info": scene_info,
                    "humans": []
                }
                
                for i, human in enumerate(humans):
                    # 3. Pose detection
                    keypoints = self.pose_detector.detect(human.cropped_image)
                    
                    if keypoints is not None:
                        # 4. Movement detection and tracking
                        movement = self.movement_detector.detect_movement(keypoints)
                        tracking_id = self.movement_tracker.track(keypoints, human.bbox)
                        
                        # 5. Store results for this human
                        human_results = {
                            "human_id": tracking_id,
                            "bbox": human.bbox,
                            "keypoints": keypoints,
                            "movement": movement
                        }
                        
                        frame_results["humans"].append(human_results)
                
                # Save the full frame results
                results.append(frame_results)
                frame_index += 1
                
                # Log progress occasionally
                if frame_index % 100 == 0:
                    logger.info(f"Processed {frame_index} frames")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_index}: {str(e)}")
        
        # Close the video
        cap.release()
        
        logger.info(f"Processed {frame_index} frames in total")
        return results
    
    def _analyze_results(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the processed frame results using dynamics and motion modules.
        
        Args:
            frame_results: List of processed frame results
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {
            "kinematics": {},
            "stride": {},
            "dynamics": {},
            "motion": {},
            "stabilography": {},
            "synchronization": {}
        }
        
        # Track unique humans
        humans = {}
        
        # Process each frame
        for frame_data in frame_results:
            # For each human in the frame
            for human_data in frame_data["humans"]:
                human_id = human_data["human_id"]
                
                # Create entry for new humans
                if human_id not in humans:
                    humans[human_id] = {
                        "frames": [],
                        "keypoints": [],
                        "timestamps": []
                    }
                
                # Store data for this human
                humans[human_id]["frames"].append(frame_data["frame_index"])
                humans[human_id]["keypoints"].append(human_data["keypoints"])
                humans[human_id]["timestamps"].append(frame_data["timestamp"])
        
        # Analyze each human's data
        for human_id, human_data in humans.items():
            # Skip humans with too few frames
            if len(human_data["frames"]) < 10:
                continue
                
            # 1. Kinematics analysis
            kinematics_results = []
            for i, keypoints in enumerate(human_data["keypoints"]):
                kinematic_data = self.kinematics_analyzer.calculate_kinematics(
                    human_id, keypoints
                )
                kinematics_results.append(kinematic_data)
            
            # 2. Stride analysis
            stride_results = []
            for i, keypoints in enumerate(human_data["keypoints"]):
                # Extract skeleton data in the format expected by stride analyzer
                skeleton_data = self._extract_skeleton_data(keypoints)
                stride_data = self.stride_analyzer.analyze_stride(
                    skeleton_data, human_id
                )
                stride_results.append(stride_data)
            
            # 3. Dynamics analysis
            dynamics_results = self.dynamics_analyzer.analyze_dynamics(
                kinematics_results, stride_results
            )
            
            # 4. Ground reaction force estimation
            grf_results = self.grf_analyzer.estimate_grf(
                kinematics_results, stride_results
            )
            
            # 5. Motion classification
            motion_classification = self.motion_classifier.classify(
                human_data["keypoints"]
            )
            
            # 6. Stabilography analysis (for balance)
            stability_results = self.stabilography_analyzer.analyze(
                human_data["keypoints"]
            )
            
            # Store all results for this human
            analysis_results["kinematics"][human_id] = kinematics_results
            analysis_results["stride"][human_id] = stride_results
            analysis_results["dynamics"][human_id] = dynamics_results
            analysis_results["grf"][human_id] = grf_results
            analysis_results["motion"][human_id] = motion_classification
            analysis_results["stabilography"][human_id] = stability_results
        
        # 7. Synchronization analysis (if multiple humans)
        if len(humans) > 1:
            sync_results = self.sync_analyzer.analyze_sync(
                {human_id: data["keypoints"] for human_id, data in humans.items()}
            )
            analysis_results["synchronization"] = sync_results
        
        # 8. Calculate aggregate metrics
        metrics = self.metrics_calculator.calculate_metrics(analysis_results)
        analysis_results["metrics"] = metrics
        
        return analysis_results
    
    def _generate_visualizations(self, 
                             video_path: str, 
                             frame_results: List[Dict[str, Any]], 
                             analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations of the analysis results.
        
        Args:
            video_path: Path to the original video
            frame_results: List of processed frame results
            analysis_results: Analysis results from _analyze_results
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        output_paths = {}
        
        # 1. Annotated video with pose landmarks and metrics
        annotated_path = str(self.output_dir / "annotated_video.mp4")
        self._create_annotated_video(video_path, frame_results, analysis_results, annotated_path)
        output_paths["annotated_video"] = annotated_path
        
        # 2. Use the pose visualizer to create additional visualizations
        pose_vis_path = str(self.output_dir / "pose_visualization.png")
        self.pose_visualizer.create_visualization(
            frame_results, analysis_results, pose_vis_path
        )
        output_paths["pose_visualization"] = pose_vis_path
        
        # 3. Create kinematics plots
        kinematics_path = str(self.output_dir / "kinematics.png")
        self._create_kinematics_plot(analysis_results["kinematics"], kinematics_path)
        output_paths["kinematics_plot"] = kinematics_path
        
        # 4. Create stride analysis plots
        stride_path = str(self.output_dir / "stride_analysis.png")
        self._create_stride_plot(analysis_results["stride"], stride_path)
        output_paths["stride_plot"] = stride_path
        
        return output_paths
    
    def _extract_skeleton_data(self, keypoints) -> Dict:
        """
        Convert keypoints to the skeleton data format expected by analyzers.
        
        Args:
            keypoints: Pose keypoints
            
        Returns:
            Dictionary with skeleton data
        """
        # This is a placeholder - the actual implementation would
        # convert from MediaPipe or another format to the format 
        # expected by the stride analyzer
        return {"keypoints": keypoints}
        
    def _create_annotated_video(self, 
                           video_path: str, 
                           frame_results: List[Dict[str, Any]], 
                           analysis_results: Dict[str, Any],
                           output_path: str) -> None:
        """
        Create annotated video with pose overlays and metrics.
        
        Args:
            video_path: Path to the original video
            frame_results: List of processed frame results
            analysis_results: Analysis results
            output_path: Path to save the annotated video
        """
        # This is a placeholder - the actual implementation would
        # create a video with annotations using the video processor
        # and skeleton drawer
        pass
    
    def _create_kinematics_plot(self, kinematics_data: Dict, output_path: str) -> None:
        """
        Create plots of kinematics data.
        
        Args:
            kinematics_data: Kinematics analysis results
            output_path: Path to save the plot
        """
        # This is a placeholder - the actual implementation would
        # create plots of joint angles, velocities, etc.
        pass
    
    def _create_stride_plot(self, stride_data: Dict, output_path: str) -> None:
        """
        Create plots of stride analysis data.
        
        Args:
            stride_data: Stride analysis results
            output_path: Path to save the plot
        """
        # This is a placeholder - the actual implementation would
        # create plots of stride length, frequency, etc.
        pass

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the core integrator
    integrator = CoreIntegrator(
        fps=30.0,
        output_dir="output/core_integration"
    )
    
    # Process a video
    results = integrator.process_video("public/sprint_100m.mp4")
    
    # Print results summary
    print(f"Processed video with {len(results['analysis_results']['metrics'])} metrics")
    print(f"Visualizations saved to: {results['visualization_paths']}") 