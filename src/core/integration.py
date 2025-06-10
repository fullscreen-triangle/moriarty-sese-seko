#!/usr/bin/env python3
"""
Core Module Integration for Moriarty Pipeline

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


from src.core.motion.movement_tracker import MovementTracker
from src.core.motion.stabilography import StabilographyAnalyzer

# Import core modules - scene
from src.core.scene.scene_detector import SceneDetector
from src.core.scene.processor import VideoProcessor
from src.core.scene.metrics import MetricsCalculator

# Import configuration and caching
from src.config import config
from src.utils.cache import cached

# Configure logging
logger = logging.getLogger(__name__)

class CoreIntegrator:
    """
    Integrates core modules for comprehensive video analysis.
    
    This class combines the functionality of pose, dynamics, motion, and scene
    analysis modules to provide a unified interface for video processing.
    """
    
    def __init__(self, 
                fps: float = None,
                output_dir: str = None,
                device: str = None,
                model_complexity: int = None):
        """
        Initialize the core integrator.
        
        Args:
            fps: Frames per second (will be updated per video)
            output_dir: Directory for output files
            device: Device to use (e.g., 'cpu', 'cuda')
            model_complexity: Complexity of pose models (0-2)
        """
        # Use configuration values with fallbacks to parameters
        self.fps = fps or config.get("video.extract_fps", 30.0)
        self.output_dir = Path(output_dir or config.get("paths.output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device if not specified
        if device is None:
            self.device = config.get("system.device", "cpu")
        else:
            self.device = device
            
        self.model_complexity = model_complexity or config.get("pose.model_complexity", 1)
        logger.info(f"Initializing CoreIntegrator (device: {self.device}, model_complexity: {self.model_complexity})")
        
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
            self.pose_detector = PoseDetector(
                model_complexity=self.model_complexity,
                confidence_threshold=config.get("pose.min_detection_confidence", 0.5)
            )
            self.skeleton_drawer = SkeletonDrawer()
            self.pose_visualizer = PoseVisualizer()
            self.pose_data_extractor = PoseDataExtractor()
            logger.info("Pose modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pose modules: {e}")
            raise RuntimeError(f"Failed to initialize pose modules: {e}")
    
    def _init_dynamics_modules(self):
        """Initialize dynamics analysis modules."""
        try:
            logger.info("Initializing dynamics modules...")
            self.kinematics_analyzer = KinematicsAnalyzer(fps=self.fps)
            self.stride_analyzer = StrideAnalyzer(fps=int(self.fps))
            self.sync_analyzer = SynchronizationAnalyzer(window_size=int(self.fps))
            self.dynamics_analyzer = DynamicsAnalyzer()
            self.grf_analyzer = GRFAnalyzer(fps=self.fps)
            logger.info("Dynamics modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing dynamics modules: {e}")
            raise RuntimeError(f"Failed to initialize dynamics modules: {e}")
    
    def _init_motion_modules(self):
        """Initialize motion analysis modules."""
        try:
            logger.info("Initializing motion modules...")
            # Initialize movement tracker (exists)
            self.movement_tracker = MovementTracker()
            # Initialize stabilography analyzer (exists)
            self.stabilography_analyzer = StabilographyAnalyzer()
            
            # Initialize optional modules with graceful fallback
            try:
                from src.core.motion.motion_classifier import MotionClassifier
                self.motion_classifier = MotionClassifier()
            except ImportError:
                logger.warning("MotionClassifier not available - using basic implementation")
                self.motion_classifier = None
                
            try:
                from src.core.motion.movement_detector import MovementDetector
                self.movement_detector = MovementDetector()
            except ImportError:
                logger.warning("MovementDetector not available - using basic implementation")
                self.movement_detector = None
                
            logger.info("Motion modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing motion modules: {e}")
            raise RuntimeError(f"Failed to initialize motion modules: {e}")
    
    def _init_scene_modules(self):
        """Initialize scene analysis modules."""
        try:
            logger.info("Initializing scene modules...")
            # Initialize core scene modules (exist)
            scene_config = {
                'scene_detection': {
                    'hist_threshold': 0.5,
                    'flow_threshold': 0.7,
                    'edge_threshold': 0.6,
                    'focus_threshold': 1000
                }
            }
            self.scene_detector = SceneDetector(scene_config)
            self.video_processor = VideoProcessor()
            self.metrics_calculator = MetricsCalculator()
            
            # Initialize optional modules with graceful fallback
            try:
                from src.core.scene.video_manager import VideoManager
                self.video_manager = VideoManager()
            except ImportError:
                logger.warning("VideoManager not available - using basic video handling")
                # Create a simple video manager substitute
                self.video_manager = self._create_basic_video_manager()
                
            try:
                from src.core.scene.scene_analyzer import SceneAnalyzer
                analyzer_config = {
                    'scene_detection': {
                        'hist_threshold': 0.5,
                        'flow_threshold': 0.7,
                        'edge_threshold': 0.6,
                        'focus_threshold': 1000
                    },
                    'output': {
                        'plots_directory': str(self.output_dir / 'scene_analysis_plots')
                    }
                }
                self.scene_analyzer = SceneAnalyzer(analyzer_config)
            except ImportError:
                logger.warning("SceneAnalyzer not available - using basic implementation")
                self.scene_analyzer = None
                
            logger.info("Scene modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing scene modules: {e}")
            raise RuntimeError(f"Failed to initialize scene modules: {e}")
    
    def _create_basic_video_manager(self):
        """Create a basic video manager for handling video operations."""
        class BasicVideoManager:
            def load_video(self, video_path):
                """Basic video loading functionality."""
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video: {video_path}")
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                cap.release()
                
                return {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "duration": frame_count / fps if fps > 0 else 0
                }
        
        return BasicVideoManager()
    
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
                visualization_paths = self._generate_visualizations(
                    video_path, frame_results, analysis_results
                )
            
            # 5. Save results if requested
            if save_results:
                logger.info("Saving results...")
                results_path = self._save_results(video_path, analysis_results)
                logger.info(f"Results saved to {results_path}")
            
            # 6. Calculate metrics
            metrics_count = self._count_metrics(analysis_results)
            logger.info(f"Analysis complete with {metrics_count} metrics calculated")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Video processing completed in {elapsed_time:.2f} seconds")
            
            return {
                "video_info": video_info,
                "frame_count": len(frame_results),
                "metrics_count": metrics_count,
                "analysis_results": analysis_results,
                "visualization_paths": visualization_paths,
                "processing_time": elapsed_time
            }
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    @cached(expiration=3600)  # Cache frame results for 1 hour
    def _process_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process individual frames of the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of processed frame results
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Human detection
            human_boxes = self.human_detector.detect(frame)
            
            # If no humans detected, continue to next frame
            if not human_boxes:
                frame_results.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / self.fps,
                    "humans_detected": 0
                })
                frame_idx += 1
                continue
            
            # Pose detection for each human
            pose_results = []
            for box in human_boxes:
                # Extract region of interest
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = box[4]
                
                # Ensure box is within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Get pose keypoints
                keypoints = self.pose_detector.detect_pose(frame, box=box)
                
                if keypoints is not None:
                    pose_results.append({
                        "box": box,
                        "keypoints": keypoints,
                        "confidence": confidence
                    })
            
            # Store frame results
            frame_results.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / self.fps,
                "humans_detected": len(human_boxes),
                "pose_results": pose_results
            })
            
            frame_idx += 1
        
        cap.release()
        return frame_results
    
    def _analyze_results(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the processed frame results.
        
        Args:
            frame_results: List of results from frame processing
            
        Returns:
            Dictionary with analysis results
        """
        # Extract keypoints for all frames
        all_keypoints = []
        for frame in frame_results:
            if "pose_results" in frame and frame["pose_results"]:
                # Use the first person's keypoints for now
                # (multi-person tracking would be more complex)
                all_keypoints.append(frame["pose_results"][0]["keypoints"])
            else:
                # No keypoints for this frame
                all_keypoints.append(None)
        
        # Skip analysis if no valid keypoints
        if not any(all_keypoints):
            logger.warning("No valid keypoints found for analysis")
            return {"valid": False}
        
        # Perform kinematics analysis
        logger.info("Performing kinematics analysis...")
        kinematics = self.kinematics_analyzer.analyze(all_keypoints)
        
        # Perform stride analysis
        logger.info("Performing stride analysis...")
        stride_data = self.stride_analyzer.analyze(all_keypoints)
        
        # Perform movement classification
        logger.info("Performing movement classification...")
        movement_class = self.motion_classifier.classify(all_keypoints)
        
        # Return combined analysis results
        return {
            "valid": True,
            "kinematics": self._prepare_kinematics_for_json(kinematics),
            "stride": stride_data,
            "movement_class": movement_class,
            "frame_count": len(frame_results),
            "keypoint_count": sum(1 for kp in all_keypoints if kp is not None)
        }
    
    def _generate_visualizations(self, 
                             video_path: str,
                             frame_results: List[Dict[str, Any]],
                             analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations for the analysis results.
        
        Args:
            video_path: Path to the original video
            frame_results: List of processed frame results
            analysis_results: Dictionary with analysis results
            
        Returns:
            Dictionary with paths to visualization files
        """
        base_name = Path(video_path).stem
        vis_paths = {}
        
        # Generate annotated video
        output_video = self.output_dir / f"{base_name}_annotated.mp4"
        self._create_annotated_video(video_path, frame_results, analysis_results, output_video)
        vis_paths["annotated_video"] = str(output_video)
        
        # Generate kinematics plot if available
        if analysis_results["valid"] and "kinematics" in analysis_results:
            kinematics_plot = self.output_dir / f"{base_name}_kinematics.png"
            self._create_kinematics_plot(analysis_results["kinematics"], kinematics_plot)
            vis_paths["kinematics_plot"] = str(kinematics_plot)
        
        # Generate stride plot if available
        if analysis_results["valid"] and "stride" in analysis_results:
            stride_plot = self.output_dir / f"{base_name}_stride.png"
            self._create_stride_plot(analysis_results["stride"], stride_plot)
            vis_paths["stride_plot"] = str(stride_plot)
        
        return vis_paths
    
    def _create_annotated_video(self, 
                           video_path: str, 
                           frame_results: List[Dict[str, Any]], 
                           analysis_results: Dict[str, Any],
                           output_path: str) -> None:
        """
        Create an annotated video with pose visualization and metrics.
        
        Args:
            video_path: Path to the original video
            frame_results: List of processed frame results
            analysis_results: Dictionary with analysis results
            output_path: Path to save the output video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Find the corresponding frame result
            if frame_idx < len(frame_results):
                frame_result = frame_results[frame_idx]
                
                # Draw pose annotations if available
                if "pose_results" in frame_result and frame_result["pose_results"]:
                    for pose in frame_result["pose_results"]:
                        # Draw skeleton
                        frame = self.skeleton_drawer.draw(frame, pose["keypoints"])
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, pose["box"][:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add metrics text
                if analysis_results["valid"]:
                    y_pos = 30
                    
                    # Add movement class
                    if "movement_class" in analysis_results:
                        cv2.putText(frame, f"Movement: {analysis_results['movement_class']}", 
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_pos += 30
                    
                    # Add kinematics for current frame if available
                    if "kinematics" in analysis_results:
                        kinematics = analysis_results["kinematics"]
                        if "joint_velocities" in kinematics and frame_idx < len(kinematics["joint_velocities"]):
                            vel = kinematics["joint_velocities"][frame_idx]
                            if vel and "hip" in vel:
                                cv2.putText(frame, f"Hip Velocity: {vel['hip']:.2f} m/s", 
                                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                y_pos += 30
            
            # Write the frame
            out.write(frame)
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
    
    def _create_kinematics_plot(self, kinematics_data: Dict, output_path: str) -> None:
        """
        Create a plot of kinematics data.
        
        Args:
            kinematics_data: Dictionary with kinematics analysis results
            output_path: Path to save the output plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Plot joint velocities if available
            if "joint_velocities" in kinematics_data:
                velocities = kinematics_data["joint_velocities"]
                frames = range(len(velocities))
                
                # Extract hip velocities
                hip_velocities = [v["hip"] if v and "hip" in v else 0 for v in velocities]
                
                plt.subplot(2, 1, 1)
                plt.plot(frames, hip_velocities, label="Hip Velocity")
                plt.title("Joint Velocities")
                plt.xlabel("Frame")
                plt.ylabel("Velocity (m/s)")
                plt.legend()
            
            # Plot joint angles if available
            if "joint_angles" in kinematics_data:
                angles = kinematics_data["joint_angles"]
                frames = range(len(angles))
                
                # Extract knee angles
                knee_angles = [a["knee"] if a and "knee" in a else 0 for a in angles]
                
                plt.subplot(2, 1, 2)
                plt.plot(frames, knee_angles, label="Knee Angle")
                plt.title("Joint Angles")
                plt.xlabel("Frame")
                plt.ylabel("Angle (degrees)")
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(str(output_path))
            plt.close()
        
        except ImportError:
            logger.error("Matplotlib not available for creating plots")
        except Exception as e:
            logger.error(f"Error creating kinematics plot: {e}")
    
    def _create_stride_plot(self, stride_data: Dict, output_path: str) -> None:
        """
        Create a plot of stride data.
        
        Args:
            stride_data: Dictionary with stride analysis results
            output_path: Path to save the output plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            # Plot stride length if available
            if "stride_lengths" in stride_data:
                stride_lengths = stride_data["stride_lengths"]
                strides = range(len(stride_lengths))
                
                plt.plot(strides, stride_lengths, 'o-', label="Stride Length")
                plt.title("Stride Analysis")
                plt.xlabel("Stride Number")
                plt.ylabel("Length (m)")
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(str(output_path))
            plt.close()
        
        except ImportError:
            logger.error("Matplotlib not available for creating plots")
        except Exception as e:
            logger.error(f"Error creating stride plot: {e}")
    
    def _save_results(self, video_path: str, analysis_results: Dict[str, Any]) -> str:
        """
        Save analysis results to a JSON file.
        
        Args:
            video_path: Path to the original video
            analysis_results: Dictionary with analysis results
            
        Returns:
            Path to the saved results file
        """
        base_name = Path(video_path).stem
        results_path = self.output_dir / f"{base_name}_results.json"
        
        try:
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            return str(results_path)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def _count_metrics(self, analysis_results: Dict[str, Any]) -> int:
        """
        Count the number of metrics in the analysis results.
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            Number of metrics
        """
        if not analysis_results.get("valid", False):
            return 0
        
        count = 0
        
        # Count metrics in kinematics
        if "kinematics" in analysis_results:
            kinematics = analysis_results["kinematics"]
            if "joint_velocities" in kinematics:
                count += len(kinematics["joint_velocities"])
            if "joint_angles" in kinematics:
                count += len(kinematics["joint_angles"])
        
        # Count stride metrics
        if "stride" in analysis_results:
            stride = analysis_results["stride"]
            if "stride_lengths" in stride:
                count += len(stride["stride_lengths"])
            if "stride_times" in stride:
                count += len(stride["stride_times"])
        
        # Count other metrics
        if "movement_class" in analysis_results:
            count += 1
        
        return count
    
    def _prepare_kinematics_for_json(self, kinematics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare kinematics data for JSON serialization.
        
        Args:
            kinematics: Dictionary with kinematics data
            
        Returns:
            JSON-serializable kinematics data
        """
        result = {}
        
        for key, value in kinematics.items():
            if isinstance(value, list):
                # Convert numpy arrays to lists
                result[key] = [
                    {k: float(v) if isinstance(v, (np.number, np.ndarray)) else v 
                     for k, v in item.items()} if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                # Convert dict values
                result[key] = {
                    k: float(v) if isinstance(v, (np.number, np.ndarray)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, (np.number, np.ndarray)):
                # Convert numpy scalar
                result[key] = float(value)
            else:
                result[key] = value
        
        return result
    
    def convert_to_llm_data(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert analysis results to format suitable for LLM training.
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            List of LLM training data entries
        """
        if not analysis_results.get("valid", False):
            return []
        
        llm_data = []
        
        # Convert kinematics
        if "kinematics" in analysis_results:
            # Use pose data extractor to convert data
            kinematics_data = self.pose_data_extractor.convert_to_text_descriptions(
                analysis_results["kinematics"]
            )
            llm_data.extend(kinematics_data if isinstance(kinematics_data, list) else [kinematics_data])
        
        # Convert stride data
        if "stride" in analysis_results:
            # Use pose data extractor to convert data
            stride_data = self.pose_data_extractor.convert_to_text_descriptions(
                analysis_results["stride"]
            )
            llm_data.extend(stride_data if isinstance(stride_data, list) else [stride_data])
        
        return llm_data 