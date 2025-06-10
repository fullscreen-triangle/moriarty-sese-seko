#!/usr/bin/env python3
"""
Core Module Integrator for Moriarty Pipeline

This module integrates the core functionality modules:
- Pose analysis (detection, tracking, visualization)
- Dynamics analysis (kinematics, stride, synchronization)
- Motion analysis (classification, movement detection and tracking)
- Scene analysis (detection, segmentation, metrics)

It provides a unified interface for using these modules in the pipeline.
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import core modules - pose
from src.core.pose.pose_detector import PoseDetector
from src.core.pose.human_detector import HumanDetector
from src.core.pose.skeleton import SkeletonDrawer
from src.core.pose.pose_visualizer import PoseVisualizer
from src.core.pose.pose_data_to_llm import PoseDataExtractor

# Import core modules - dynamics
from src.core.dynamics.kinematics_analyzer import KinematicsAnalyzer
from src.core.dynamics.stride_analyzer import StrideAnalyzer
from src.core.dynamics.sync_analyzer import SynchronizationAnalyzer
from src.core.dynamics.dynamics_analyzer import DynamicsAnalyzer
from src.core.dynamics.grf_analyzer import GRFAnalyzer
from src.core.dynamics.aidynamics_analyzer import AIDynamicsAnalyzer

# Import core modules - motion
from src.core.motion.motion_classifier import MotionClassifier
from src.core.motion.movement_detector import MovementDetector
from src.core.motion.movement_tracker import MovementTracker
from src.core.motion.stabilography import StabilographyAnalyzer

# Import core modules - scene
from src.core.scene.scene_detector import SceneDetector
from src.core.scene.video_manager import VideoManager
from src.core.scene.processor import VideoProcessor
from src.core.scene.analyzer import SceneAnalyzer
from src.core.scene.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("core_integrator.log")
    ]
)
logger = logging.getLogger("core_integrator")

class CoreIntegrator:
    """
    Integrates core modules for comprehensive video analysis.
    
    This class combines the functionality of pose, dynamics, motion, and scene
    analysis modules to provide a unified interface for video processing.
    """
    
    def __init__(self, 
                fps: float = 30.0,
                output_dir: str = "output",
                device: str = None,
                model_complexity: int = 1):
        """
        Initialize the core integrator.
        
        Args:
            fps: Frames per second (will be updated per video)
            output_dir: Directory for output files
            device: Device to use (e.g., 'cpu', 'cuda')
            model_complexity: Complexity of pose models (0-2)
        """
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device if not specified
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model_complexity = model_complexity
        logger.info(f"Initializing CoreIntegrator (device: {self.device}, model_complexity: {model_complexity})")
        
        # Initialize all core modules
        self._init_pose_modules()
        self._init_dynamics_modules()
        self._init_motion_modules()
        self._init_scene_modules()
        
    def _init_pose_modules(self):
        """Initialize pose detection and analysis modules."""
        try:
            logger.info("Initializing pose modules...")
            self.human_detector = HumanDetector(device=self.device)
            self.pose_detector = PoseDetector(model_complexity=self.model_complexity)
            self.skeleton_drawer = SkeletonDrawer()
            self.pose_visualizer = PoseVisualizer()
            self.pose_data_extractor = PoseDataExtractor()
            logger.info("Pose modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pose modules: {e}")
            raise RuntimeError(f"Failed to initialize pose modules: {e}")
    
    def _init_dynamics_modules(self):
        """Initialize biomechanical dynamics modules."""
        # Kinematics analysis
        self.kinematics_analyzer = KinematicsAnalyzer(fps=self.fps)
        
        # Stride analysis
        self.stride_analyzer = StrideAnalyzer(fps=int(self.fps))
        
        # Synchronization analysis (for multiple athletes)
        self.sync_analyzer = SynchronizationAnalyzer(window_size=30)
        
        # General dynamics analyzer with GPU acceleration if available
        self.dynamics_analyzer = DynamicsAnalyzer(use_gpu=self.device == "cuda")
        
        # AI-powered dynamics analyzer with GPU acceleration if available
        self.ai_dynamics_analyzer = AIDynamicsAnalyzer(use_gpu=self.device == "cuda")
        
        # Ground reaction force analyzer
        self.grf_analyzer = GRFAnalyzer(fps=self.fps, use_gpu=self.device == "cuda")
        
        logger.info(f"Dynamics modules initialized with device: {self.device}")
    
    def _init_motion_modules(self):
        """Initialize motion analysis modules."""
        try:
            logger.info("Initializing motion modules...")
            self.motion_classifier = MotionClassifier()
            self.movement_detector = MovementDetector()
            self.movement_tracker = MovementTracker()
            self.stabilography_analyzer = StabilographyAnalyzer()
            logger.info("Motion modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing motion modules: {e}")
            raise RuntimeError(f"Failed to initialize motion modules: {e}")
    
    def _init_scene_modules(self):
        """Initialize scene analysis modules."""
        try:
            logger.info("Initializing scene modules...")
            # Scene detector configuration
            scene_config = {
                'scene_detection': {
                    'hist_threshold': 0.5,
                    'flow_threshold': 0.7,
                    'edge_threshold': 0.6,
                    'focus_threshold': 1000
                }
            }
            self.scene_detector = SceneDetector(scene_config)
            self.video_manager = VideoManager()
            self.video_processor = VideoProcessor()
            
            # Scene analyzer configuration
            analyzer_config = {
                'scene_detection': {
                    'hist_threshold': 0.5,
                    'flow_threshold': 0.7,
                    'edge_threshold': 0.6,
                    'focus_threshold': 1000
                },
                'output': {
                    'plots_directory': 'scene_analysis_plots'
                }
            }
            self.scene_analyzer = SceneAnalyzer(analyzer_config)
            self.metrics_calculator = MetricsCalculator()
            logger.info("Scene modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing scene modules: {e}")
            raise RuntimeError(f"Failed to initialize scene modules: {e}")
    
    def process_video(self, 
                   video_path: str, 
                   generate_visualizations: bool = True,
                   save_results: bool = True) -> Dict[str, Any]:
        """
        Process a video using the core modules.
        
        Args:
            video_path: Path to the video
            generate_visualizations: Whether to generate visualizations
            save_results: Whether to save the results to disk
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        logger.info(f"Processing video: {video_path}")
        
        try:
            # 1. Load video and update FPS
            video_info = self.video_manager.load_video(video_path)
            self.fps = video_info["fps"]
            
            # Update FPS for analyzers that use it
            self.kinematics_analyzer.fps = self.fps
            self.stride_analyzer.fps = int(self.fps)
            self.grf_analyzer.fps = self.fps
            
            logger.info(f"Video loaded: {video_info['width']}x{video_info['height']} @ {self.fps} FPS")
            
            # 2. Process frames
            logger.info("Processing video frames...")
            frame_results = self._process_frames(video_path)
            logger.info(f"Processed {len(frame_results)} frames")
            
            # 3. Analyze the results
            logger.info("Analyzing results...")
            analysis_results = self._analyze_results(frame_results)
            
            # 4. Generate visualizations if requested
            visualization_paths = {}
            if generate_visualizations:
                logger.info("Generating visualizations...")
                visualization_paths = self._generate_visualizations(video_path, frame_results, analysis_results)
            
            # 5. Save results if requested
            results_path = None
            if save_results:
                results_path = self.output_dir / f"{Path(video_path).stem}_analysis.json"
                
                # Prepare serializable results
                serializable_results = {
                    "video_info": video_info,
                    "analysis_summary": {
                        "num_athletes": len(analysis_results.get("kinematics", {})),
                        "num_frames": len(frame_results),
                        "metrics_count": self._count_metrics(analysis_results),
                    },
                    "kinematics": self._prepare_kinematics_for_json(analysis_results.get("kinematics", {})),
                    "stride_metrics": analysis_results.get("stride_metrics", {}),
                    "grf_data": analysis_results.get("grf_data", {}),
                    "scene_analysis": analysis_results.get("scene_analysis", {}),
                    "motion_classification": analysis_results.get("motion_classification", {}),
                }
                
                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                    
                logger.info(f"Results saved to: {results_path}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return the results
            return {
                "success": True,
                "video_path": video_path,
                "video_info": video_info,
                "processing_time": processing_time,
                "frame_count": len(frame_results),
                "analysis_results": analysis_results,
                "visualization_paths": visualization_paths,
                "results_path": str(results_path) if results_path else None
            }
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "video_path": video_path,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _process_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process individual frames from a video.
        
        Args:
            video_path: Path to the video
            
        Returns:
            List of dictionaries with frame processing results
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing {total_frames} frames ({width}x{height})")
        
        # Process frames
        frame_results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            try:
                frame_result = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / self.fps
                }
                
                # Scene detection
                scene_info = self.scene_detector.detect(frame)
                frame_result["scene"] = scene_info
                
                # Human detection
                humans = self.human_detector.detect(frame)
                frame_result["humans"] = humans
                
                # Pose detection
                poses = []
                for human in humans:
                    x1, y1, x2, y2 = human["bbox"]
                    human_crop = frame[y1:y2, x1:x2]
                    pose = self.pose_detector.detect(human_crop)
                    
                    # Convert pose coordinates to original frame coordinates
                    adjusted_pose = {}
                    for key, (x, y, confidence) in pose.items():
                        adjusted_pose[key] = (x + x1, y + y1, confidence)
                    
                    poses.append({
                        "human_id": human["id"],
                        "pose": adjusted_pose,
                        "bbox": human["bbox"]
                    })
                
                frame_result["poses"] = poses
                
                # Movement tracking
                track_info = self.movement_tracker.track(frame, poses)
                frame_result["tracks"] = track_info
                
                # Add the result to our list
                frame_results.append(frame_result)
                
                # Log progress periodically
                if frame_idx % 100 == 0:
                    logger.info(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                
                frame_idx += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
                # We'll continue processing other frames
        
        # Clean up
        cap.release()
        logger.info(f"Processed {frame_idx} frames")
        
        return frame_results
    
    def _analyze_results(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the processed frame results.
        
        Args:
            frame_results: List of processed frame results
            
        Returns:
            Dictionary with analysis results
        """
        # Extract pose data for each track
        track_poses = {}
        for frame in frame_results:
            for pose_data in frame["poses"]:
                track_id = pose_data["human_id"]
                if track_id not in track_poses:
                    track_poses[track_id] = []
                    
                track_poses[track_id].append({
                    "frame_idx": frame["frame_idx"],
                    "timestamp": frame["timestamp"],
                    "pose": pose_data["pose"],
                    "bbox": pose_data["bbox"]
                })
        
        # Analyze kinematics for each track
        kinematics_results = {}
        for track_id, poses in track_poses.items():
            # Extract pose sequences
            pose_sequence = [p["pose"] for p in poses]
            
            # Analyze kinematics
            kinematics = self.kinematics_analyzer.analyze(pose_sequence)
            
            # Analyze stride
            stride_metrics = self.stride_analyzer.analyze(pose_sequence)
            
            # Store the results
            kinematics_results[track_id] = {
                "kinematics": kinematics,
                "stride_metrics": stride_metrics
            }
        
        # Analyze synchronization between athletes
        sync_results = None
        if len(track_poses) > 1:
            # Extract pose sequences
            pose_sequences = {track_id: [p["pose"] for p in poses] 
                             for track_id, poses in track_poses.items()}
            
            # Analyze synchronization
            sync_results = self.sync_analyzer.analyze(pose_sequences)
        
        # Analyze GRF (Ground Reaction Force)
        grf_results = {}
        for track_id, poses in track_poses.items():
            # Extract pose sequence
            pose_sequence = [p["pose"] for p in poses]
            
            # Analyze GRF
            grf_data = self.grf_analyzer.analyze(pose_sequence)
            grf_results[track_id] = grf_data
        
        # Classify movements
        motion_results = {}
        for track_id, poses in track_poses.items():
            # Extract pose sequence
            pose_sequence = [p["pose"] for p in poses]
            
            # Classify motion
            classification = self.motion_classifier.classify(pose_sequence)
            motion_results[track_id] = classification
        
        # Scene analysis
        scene_features = [frame["scene"] for frame in frame_results]
        scene_analysis = self.scene_analyzer.analyze(scene_features)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate(
            kinematics_results, 
            scene_analysis,
            motion_results
        )
        
        # Combine all results
        return {
            "kinematics": {track_id: results["kinematics"] for track_id, results in kinematics_results.items()},
            "stride_metrics": {track_id: results["stride_metrics"] for track_id, results in kinematics_results.items()},
            "synchronization": sync_results,
            "grf_data": grf_results,
            "motion_classification": motion_results,
            "scene_analysis": scene_analysis,
            "metrics": metrics
        }
    
    def _generate_visualizations(self, 
                             video_path: str,
                             frame_results: List[Dict[str, Any]],
                             analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations from the analysis results.
        
        Args:
            video_path: Path to the original video
            frame_results: List of processed frame results
            analysis_results: Analysis results
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        visualization_paths = {}
        
        # Ensure visualization directory exists
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Generate pose visualization video
        pose_video_path = vis_dir / f"{Path(video_path).stem}_pose.mp4"
        self.pose_visualizer.create_visualization_video(
            video_path, 
            frame_results, 
            str(pose_video_path)
        )
        visualization_paths["pose_video"] = str(pose_video_path)
        
        # 2. Generate skeleton visualization
        skeleton_path = vis_dir / f"{Path(video_path).stem}_skeleton.mp4"
        self.skeleton_drawer.create_skeleton_video(
            video_path,
            frame_results,
            str(skeleton_path)
        )
        visualization_paths["skeleton_video"] = str(skeleton_path)
        
        # 3. Generate kinematics plots
        kinematics_plot_path = vis_dir / f"{Path(video_path).stem}_kinematics.png"
        self.kinematics_analyzer.plot_metrics(
            analysis_results["kinematics"],
            str(kinematics_plot_path)
        )
        visualization_paths["kinematics_plot"] = str(kinematics_plot_path)
        
        # 4. Generate stride analysis plots
        stride_plot_path = vis_dir / f"{Path(video_path).stem}_stride.png"
        self.stride_analyzer.plot_stride_metrics(
            analysis_results["stride_metrics"],
            str(stride_plot_path)
        )
        visualization_paths["stride_plot"] = str(stride_plot_path)
        
        # 5. Generate GRF visualization if available
        if analysis_results["grf_data"]:
            grf_plot_path = vis_dir / f"{Path(video_path).stem}_grf.png"
            self.grf_analyzer.plot_grf(
                analysis_results["grf_data"],
                str(grf_plot_path)
            )
            visualization_paths["grf_plot"] = str(grf_plot_path)
        
        # 6. Generate synchronization visualization if available
        if analysis_results["synchronization"]:
            sync_plot_path = vis_dir / f"{Path(video_path).stem}_sync.png"
            self.sync_analyzer.plot_synchronization(
                analysis_results["synchronization"],
                str(sync_plot_path)
            )
            visualization_paths["sync_plot"] = str(sync_plot_path)
        
        logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def _count_metrics(self, analysis_results: Dict[str, Any]) -> int:
        """
        Count the number of metrics in the analysis results.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Total number of metrics
        """
        count = 0
        
        # Count kinematics metrics
        for athlete_id, kinematics in analysis_results.get("kinematics", {}).items():
            count += len(kinematics)
        
        # Count stride metrics
        for athlete_id, stride_metrics in analysis_results.get("stride_metrics", {}).items():
            count += len(stride_metrics)
        
        # Count GRF metrics
        for athlete_id, grf_data in analysis_results.get("grf_data", {}).items():
            count += len(grf_data)
        
        # Count other metrics
        count += len(analysis_results.get("metrics", {}))
        
        return count
    
    def _prepare_kinematics_for_json(self, kinematics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare kinematics data for JSON serialization.
        
        Args:
            kinematics: Kinematics data
            
        Returns:
            JSON-serializable kinematics data
        """
        serializable = {}
        
        for athlete_id, data in kinematics.items():
            # Convert numpy arrays to lists
            serializable[str(athlete_id)] = {}
            for key, value in data.items():
                if hasattr(value, 'tolist'):
                    serializable[str(athlete_id)][key] = value.tolist()
                else:
                    serializable[str(athlete_id)][key] = value
        
        return serializable

    def convert_to_llm_data(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert analysis results to LLM training data.
        
        Args:
            analysis_results: Analysis results from process_video
            
        Returns:
            List of training examples for the LLM
        """
        # Use the PoseDataConverter to convert the data
        training_examples = self.pose_data_extractor.convert_to_text_descriptions(analysis_results)
        return training_examples

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create core integrator
    integrator = CoreIntegrator(
        output_dir="output/core_integration",
        model_complexity=1
    )
    
    # Process a video
    results = integrator.process_video(
        video_path="public/sprint_100m.mp4",
        generate_visualizations=True,
        save_results=True
    )
    
    # Print results summary
    if results["success"]:
        print(f"Successfully processed video: {results['video_path']}")
        print(f"Processed {results['frame_count']} frames in {results['processing_time']:.2f} seconds")
        print(f"Metrics processed: {integrator._count_metrics(results['analysis_results'])}")
        print(f"Visualizations saved to: {', '.join(results['visualization_paths'].values())}")
    else:
        print(f"Failed to process video: {results['error']}") 