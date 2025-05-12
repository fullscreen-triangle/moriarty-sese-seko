#!/usr/bin/env python3
"""
Moriarty Pipeline Orchestrator

This script orchestrates the complete Moriarty pipeline workflows:
1. Sprint Running Video Analysis Pipeline
2. Domain Expert LLM Training Pipeline

It serves as the main entry point for running the entire system.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json
import multiprocessing
from typing import Dict, List, Any, Optional, Union

# Import internal components
from src.pipeline import VideoPipeline
from src.models.train_llm import LLMTrainer
from src.distributed.memory_monitor import MemoryMonitor # this function does not exist
from src.utils.file_utils import ensure_dir_exists
from src.core_integrator import CoreIntegrator  # Import our new CoreIntegrator
from src.utils.logging_manager import configure_logging, get_logger

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

from src.core.motion.motion_classifier import MotionClassifier # none existent function
from src.core.motion.movement_detector import MovementDetector # none existent function
from src.core.motion.movement_tracker import MovementTracker
from src.core.motion.stabilography import StabilographyAnalyzer

from src.core.scene.scene_detector import SceneDetector
from src.core.scene.video_manager import VideoManager # none existent function
from src.core.scene.processor import VideoProcessor
from src.core.scene.analyzer import SceneAnalyzer # none existent function
from src.core.scene.metrics import MetricsCalculator

# Configure unified logging system
logger = configure_logging(
    app_name="moriarty_pipeline",
    verbosity="info",
    log_file="moriarty_pipeline.log",
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    module_levels={
        "src.core": "info",
        "src.utils": "info",
        "src.models": "info"
    }
)

class MoriartyPipeline:
    """
    Master orchestrator for Moriarty pipelines.
    Handles both video analysis and LLM training pipelines.
    """

    def __init__(self, 
                output_dir: str = "output",
                memory_limit: float = 0.4,
                n_workers: int = None,
                batch_size: int = 5):
        """
        Initialize the Moriarty pipeline orchestrator.

        Args:
            output_dir: Directory for all pipeline outputs
            memory_limit: Memory limit as a fraction of total system memory
            n_workers: Number of workers for distributed processing
            batch_size: Batch size for frame processing
        """
        self.output_dir = Path(output_dir)
        self.memory_limit = memory_limit

        # Auto-determine number of workers if not specified
        if n_workers is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_workers = n_workers

        self.batch_size = batch_size

        # Create output directories
        self.video_output_dir = self.output_dir / "video_analysis"
        self.llm_training_dir = self.output_dir / "llm_training_data"
        self.llm_models_dir = self.output_dir / "llm_models"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.output_dir, self.video_output_dir, 
                         self.llm_training_dir, self.llm_models_dir, 
                         self.logs_dir]:
            ensure_dir_exists(dir_path)

        # Initialize the video pipeline (legacy fallback)
        self.video_pipeline = VideoPipeline(
            memory_limit_fraction=memory_limit,
            n_workers=self.n_workers,
            batch_size=batch_size,
            output_dir=str(self.video_output_dir),
            llm_training_dir=str(self.llm_training_dir),
            llm_model_dir=str(self.llm_models_dir)
        )

        # Initialize the LLM trainer
        self.llm_trainer = LLMTrainer(
            data_dir=str(self.llm_training_dir),
            model_dir=str(self.llm_models_dir)
        )

        # Check and initialize core modules
        self.core_modules_available = False
        self._check_core_modules()

        # Initialize core integrator
        try:
            self.core_integrator = CoreIntegrator(
                output_dir=str(self.video_output_dir)
            )
            self.core_modules_available = True
            logger.info("CoreIntegrator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CoreIntegrator: {e}")
            self.core_modules_available = False

        logger.info(f"Initialized Moriarty Pipeline with {self.n_workers} workers")

    def _check_core_modules(self):
        """
        Check the availability of different core module categories
        and set appropriate flags.
        """
        # Check pose modules
        try:
            # Test initializing PoseDetector
            _ = PoseDetector()
            _ = HumanDetector()
            self.pose_modules_available = True
            logger.info("Pose modules are available")
        except Exception as e:
            logger.warning(f"Pose modules not available: {e}")
            self.pose_modules_available = False

        # Check dynamics modules
        try:
            # Test initializing KinematicsAnalyzer
            _ = KinematicsAnalyzer()
            _ = StrideAnalyzer()
            self.dynamics_modules_available = True
            logger.info("Dynamics modules are available")
        except Exception as e:
            logger.warning(f"Dynamics modules not available: {e}")
            self.dynamics_modules_available = False

        # Check motion modules
        try:
            # Test initializing MotionClassifier
            _ = MotionClassifier()
            _ = MovementTracker()
            self.motion_modules_available = True
            logger.info("Motion modules are available")
        except Exception as e:
            logger.warning(f"Motion modules not available: {e}")
            self.motion_modules_available = False

        # Check scene modules
        try:
            # Test initializing SceneDetector
            _ = SceneDetector()
            _ = VideoProcessor(None)  # Just testing if class is available
            self.scene_modules_available = True
            logger.info("Scene modules are available")
        except Exception as e:
            logger.warning(f"Scene modules not available: {e}")
            self.scene_modules_available = False

        # Overall core module availability
        self.core_modules_available = all([
            self.pose_modules_available,
            self.dynamics_modules_available,
            self.motion_modules_available,
            self.scene_modules_available
        ])

    def run_sprint_video_analysis(self, 
                               video_path: str, 
                               create_visualizations: bool = True,
                               generate_report: bool = True,
                               prepare_llm_data: bool = True) -> Dict[str, Any]:
        """
        Run the Sprint Running Video Analysis Pipeline.

        Args:
            video_path: Path to sprint running video
            create_visualizations: Whether to create annotated videos
            generate_report: Whether to generate a detailed analysis report
            prepare_llm_data: Whether to prepare data for LLM training

        Returns:
            Dictionary with analysis results and status
        """
        start_time = time.time()

        logger.info(f"Starting sprint video analysis for: {video_path}")

        # Check if core modules are available via CoreIntegrator
        if self.core_modules_available:
            # Use core integrator for processing
            logger.info("Using CoreIntegrator for video analysis")
            result = self.core_integrator.process_video(
                video_path=video_path,
                generate_visualizations=create_visualizations,
                save_results=True
            )
        else:
            # Instead of falling back to legacy pipeline, directly use core modules
            logger.info("CoreIntegrator not available, using direct core module integration")
            try:
                result = self._run_core_module_analysis(
                    video_path=video_path,
                    create_visualizations=create_visualizations
                )
            except Exception as e:
                logger.error(f"Core module analysis failed, falling back to legacy pipeline: {e}")
                # Fall back to the legacy pipeline as last resort
                result = self.video_pipeline.process_video(
                    video_path=video_path,
                    sport_type="sprint",
                    output_annotations=create_visualizations
                )

        if not result["success"]:
            logger.error(f"Video processing failed: {result.get('error', 'Unknown error')}")
            return result

        # Generate detailed report if requested
        if generate_report and result["success"]:
            try:
                report_path = os.path.join(self.video_output_dir, "sprint_report.html")
                # This would use the report generator from utils
                from src.utils.report_generator import generate_sprint_report # none existent function
                generate_sprint_report(result["analysis_results"], report_path)
                result["report_path"] = report_path
                logger.info(f"Generated sprint report: {report_path}")
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                result["report_error"] = str(e)

        # Prepare LLM training data if requested
        if prepare_llm_data and result["success"]:
            try:
                # Generate LLM training data from the analysis results
                if self.core_modules_available:
                    # Use the CoreIntegrator's method to convert to LLM data
                    llm_data = self.core_integrator.convert_to_llm_data(result["analysis_results"])
                    # Save the LLM training data
                    llm_data_path = os.path.join(
                        self.llm_training_dir, 
                        f"{os.path.basename(video_path)}_training_data.json"
                    )
                    with open(llm_data_path, 'w') as f:
                        json.dump(llm_data, f, indent=2)
                    result["llm_data_path"] = llm_data_path
                else:
                    # Use the legacy method for generating LLM data
                    llm_data_path = self._generate_llm_data(result["analysis_results"])
                    result["llm_data_path"] = llm_data_path

                logger.info(f"Generated LLM training data: {result.get('llm_data_path')}")
            except Exception as e:
                logger.error(f"Error generating LLM training data: {str(e)}")
                result["llm_data_error"] = str(e)

        # Calculate processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        logger.info(f"Completed sprint video analysis in {processing_time:.2f} seconds")
        return result

    def _run_core_module_analysis(self, video_path: str, create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run analysis using the individual core modules directly.
        This method directly uses all imported core modules for comprehensive analysis.

        Args:
            video_path: Path to the video file
            create_visualizations: Whether to create annotated visualizations

        Returns:
            Dictionary with analysis results and status
        """
        # Initialize result dictionary
        result = {
            "success": False,
            "video_path": video_path,
            "analysis_results": {},
            "visualizations": {}
        }

        try:
            # 1. Initialize video processing
            logger.info("Initializing video manager and scene detection")
            video_manager = VideoManager(video_path)
            video_processor = VideoProcessor(video_manager)
            scene_detector = SceneDetector()
            scene_analyzer = SceneAnalyzer()

            # Get video metadata
            video_info = video_manager.get_video_info()
            result["video_info"] = video_info

            # 2. Human detection and pose estimation
            logger.info("Running human detection and pose estimation")
            human_detector = HumanDetector()
            pose_detector = PoseDetector()

            # Process video frames
            frames = video_processor.extract_frames()
            human_detections = []
            pose_estimations = []

            for frame in frames:
                # Detect humans
                humans = human_detector.detect(frame)
                human_detections.append(humans)

                # Estimate poses for each detected human
                poses = []
                for human in humans:
                    pose = pose_detector.detect_pose(frame, human)
                    poses.append(pose)
                pose_estimations.append(poses)

            # 3. Movement tracking and motion analysis
            logger.info("Running movement tracking and motion analysis")
            movement_tracker = MovementTracker()
            movement_detector = MovementDetector()
            motion_classifier = MotionClassifier()
            stabilography = StabilographyAnalyzer()

            # Track movement across frames
            tracked_movements = movement_tracker.track_movements(frames, pose_estimations)
            motion_data = movement_detector.detect_movements(tracked_movements)
            motion_classifications = motion_classifier.classify_motions(motion_data)
            stability_metrics = stabilography.analyze(tracked_movements)

            # 4. Biomechanical dynamics analysis
            logger.info("Running biomechanical dynamics analysis")
            kinematics = KinematicsAnalyzer()
            stride_analyzer = StrideAnalyzer()
            sync_analyzer = SynchronizationAnalyzer()
            dynamics_analyzer = DynamicsAnalyzer()
            grf_analyzer = GRFAnalyzer()

            # Analyze kinematics
            kinematic_data = kinematics.analyze(tracked_movements)
            stride_data = stride_analyzer.analyze_strides(tracked_movements)
            sync_data = sync_analyzer.analyze_synchronization(tracked_movements)
            dynamics_data = dynamics_analyzer.analyze_dynamics(tracked_movements)
            grf_data = grf_analyzer.estimate_grf(tracked_movements, kinematic_data)

            # 5. Performance metrics calculation
            logger.info("Calculating performance metrics")
            metrics_calculator = MetricsCalculator()
            performance_metrics = metrics_calculator.calculate(
                kinematic_data, 
                stride_data,
                motion_classifications
            )

            # 6. Create visualizations if requested
            if create_visualizations:
                logger.info("Creating visualizations")
                skeleton_drawer = SkeletonDrawer()
                pose_visualizer = PoseVisualizer()

                # Create annotated video
                annotated_frames = []
                for i, frame in enumerate(frames):
                    # Draw skeleton
                    skeleton_frame = skeleton_drawer.draw(
                        frame.copy(), 
                        pose_estimations[i]
                    )

                    # Add pose visualizations
                    annotated_frame = pose_visualizer.visualize(
                        skeleton_frame,
                        pose_estimations[i],
                        kinematic_data.get(i, {}),
                        stride_data.get(i, {})
                    )

                    annotated_frames.append(annotated_frame)

                # Save annotated video
                output_video_path = os.path.join(
                    self.video_output_dir,
                    f"{os.path.splitext(os.path.basename(video_path))[0]}_annotated.mp4"
                )
                video_processor.save_frames_as_video(
                    annotated_frames,
                    output_video_path,
                    fps=video_info["fps"]
                )

                result["visualizations"]["annotated_video"] = output_video_path

            # 7. Compile all analysis results
            analysis_results = {
                "video_info": video_info,
                "kinematics": kinematic_data,
                "stride_analysis": stride_data,
                "synchronization": sync_data,
                "dynamics": dynamics_data,
                "ground_reaction_forces": grf_data,
                "motion_classification": motion_classifications,
                "stability": stability_metrics,
                "performance_metrics": performance_metrics
            }

            # Save analysis results to file
            results_file = os.path.join(
                self.video_output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_analysis.json"
            )
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)

            result["analysis_results"] = analysis_results
            result["analysis_file"] = results_file
            result["success"] = True

            logger.info(f"Core module analysis completed successfully")

        except Exception as e:
            logger.error(f"Error in core module analysis: {str(e)}")
            result["error"] = str(e)
            result["success"] = False

        return result

    def _generate_llm_data(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate LLM training data from analysis results (legacy method).

        Args:
            analysis_results: Analysis results

        Returns:
            Path to the generated LLM training data file
        """
        # Create a unique filename
        timestamp = int(time.time())
        llm_data_path = os.path.join(
            self.llm_training_dir, 
            f"sprint_analysis_{timestamp}.json"
        )

        # Create training data (simplified for demonstration)
        training_data = {
            "examples": [
                {
                    "input": "Describe the runner's stride pattern.",
                    "output": f"The runner exhibits a stride length of approximately {analysis_results.get('stride_length', 'unknown')} meters with a cadence of {analysis_results.get('cadence', 'unknown')} strides per minute."
                },
                {
                    "input": "What was the average velocity?",
                    "output": f"The average velocity was {analysis_results.get('velocity', 'unknown')} m/s."
                }
            ]
        }

        # Save the data
        with open(llm_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        return llm_data_path

    def run_batch_video_analysis(self, 
                              input_folder: str = "public",
                              file_pattern: str = "*.mp4") -> List[Dict[str, Any]]:
        """
        Run batch analysis on multiple videos in a folder.

        Args:
            input_folder: Folder containing videos to analyze
            file_pattern: File pattern for videos to process

        Returns:
            List of results for each processed video
        """
        from glob import glob
        import os

        # Find all matching video files
        video_paths = glob(os.path.join(input_folder, file_pattern))
        if not video_paths:
            logger.warning(f"No videos found matching {file_pattern} in {input_folder}")
            return []

        logger.info(f"Found {len(video_paths)} videos to process")

        # Process each video
        results = []
        for video_path in video_paths:
            try:
                result = self.run_sprint_video_analysis(video_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results.append({
                    "success": False,
                    "video_path": video_path,
                    "error": str(e)
                })

        return results

    def setup_llm_training(self, 
                        base_model: str = "facebook/opt-1.3b",
                        datasets: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Set up the Domain Expert LLM Training Pipeline.

        Args:
            base_model: Base model to use for fine-tuning
            datasets: Dictionary mapping dataset names to paths

        Returns:
            Status dictionary
        """
        logger.info(f"Setting up LLM training with base model: {base_model}")

        # Initialize result dictionary
        result = {
            "success": False,
            "base_model": base_model,
            "timestamp": time.time()
        }

        try:
            # Load and merge datasets if provided
            if datasets:
                logger.info(f"Loading datasets: {', '.join(datasets.keys())}")

                # Import dataset loaders
                # was this meant to be the c3d dataset ? the amc dataset is not being used
                from src.datasets.cud_loader import load_cud_dataset
                from src.datasets.maxplanck_dataset import load_maxplanck_dataset
                from src.datasets.nomo_dataset import load_nomo_dataset
                from src.datasets.unified_dataset import merge_datasets

                # Load individual datasets
                loaded_datasets = []

                for dataset_name, dataset_path in datasets.items():
                    if dataset_name.lower() == "cud":
                        dataset = load_cud_dataset(dataset_path)
                    elif dataset_name.lower() == "maxplanck":
                        dataset = load_maxplanck_dataset(dataset_path)
                    elif dataset_name.lower() == "nomo":
                        dataset = load_nomo_dataset(dataset_path)
                    else:
                        logger.warning(f"Unknown dataset type: {dataset_name}, skipping")
                        continue

                    if dataset:
                        loaded_datasets.append(dataset)
                        logger.info(f"Loaded {dataset_name} dataset: {len(dataset)} examples")

                # Merge the datasets
                if loaded_datasets:
                    merged_dataset = merge_datasets(loaded_datasets)

                    # Process and split the dataset
                    from src.datasets.data_processor import process_biomechanics_data
                    from src.utils.data_splitter import split_dataset

                    processed_dataset = process_biomechanics_data(merged_dataset)
                    train_dataset, val_dataset, test_dataset = split_dataset(processed_dataset)

                    # Save the datasets
                    train_path = self.llm_training_dir / "train_dataset.json"
                    val_path = self.llm_training_dir / "val_dataset.json"
                    test_path = self.llm_training_dir / "test_dataset.json"

                    with open(train_path, 'w') as f:
                        json.dump(train_dataset, f)

                    with open(val_path, 'w') as f:
                        json.dump(val_dataset, f)

                    with open(test_path, 'w') as f:
                        json.dump(test_dataset, f)

                    result["train_dataset_path"] = str(train_path)
                    result["val_dataset_path"] = str(val_path)
                    result["test_dataset_path"] = str(test_path)
                    result["train_size"] = len(train_dataset)
                    result["val_size"] = len(val_dataset)
                    result["test_size"] = len(test_dataset)

                    logger.info(f"Datasets prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")

            # Create checkpoint directory
            checkpoint_dir = self.llm_models_dir / "checkpoints"
            ensure_dir_exists(checkpoint_dir)
            result["checkpoint_dir"] = str(checkpoint_dir)

            # Set up GPU memory management
            try:
                from src.utils.gpu_manager import setup_gpu_memory_management
                setup_gpu_memory_management()
                logger.info("GPU memory management configured")
            except ImportError:
                logger.warning("GPU memory management module not found, skipping")

            # Setup training configuration
            from transformers import TrainingArguments

            training_args = TrainingArguments(
                output_dir=str(self.llm_models_dir / "biomechanics_llm"),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=1e-5,
                weight_decay=0.01,
                warmup_steps=500,
                logging_steps=100,
                eval_steps=500,
                save_steps=500,
                fp16=True,
                optim="adamw_torch",
                report_to="tensorboard"
            )

            result["training_args"] = {
                "num_train_epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                "output_dir": training_args.output_dir
            }

            # Save the configuration for future reference
            config_path = self.llm_models_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(result, f, indent=2)

            result["config_path"] = str(config_path)
            result["success"] = True

            logger.info(f"LLM training setup complete: {config_path}")

        except Exception as e:
            logger.error(f"Error setting up LLM training: {str(e)}")
            result["error"] = str(e)

        return result

    def start_llm_training(self, 
                        config_path: str = None,
                        background: bool = True,
                        resume_from_checkpoint: str = None) -> Dict[str, Any]:
        """
        Start the LLM training process.

        Args:
            config_path: Path to training configuration
            background: Whether to run in background
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Status dictionary with process information
        """
        # Load configuration if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "base_model": "facebook/opt-1.3b",
                "train_dataset_path": str(self.llm_training_dir / "train_dataset.json"),
                "val_dataset_path": str(self.llm_training_dir / "val_dataset.json"),
            }

        logger.info(f"Starting LLM training with model: {config.get('base_model')}")

        if background:
            # Import process management utilities
            from src.utils.process_manager import detach_process
            from src.utils.logging_manager import setup_training_logs

            # Setup logging for long-running process
            log_dir = str(self.logs_dir / "llm_training")
            ensure_dir_exists(log_dir)
            setup_training_logs(log_dir)

            # Start training in background process
            pid = detach_process()

            if pid == 0:  # Child process
                # This runs in the background
                logger.info(f"Starting background training process (PID: {os.getpid()})")

                # Prepare the training data
                self.llm_trainer.prepare_training_data(config.get("train_dataset_path"))

                # Start the training
                model_path = self.llm_trainer.train_model(
                    epochs=config.get("num_train_epochs", 3),
                    batch_size=config.get("per_device_train_batch_size", 4),
                    learning_rate=config.get("learning_rate", 1e-5),
                    model_name="biomechanics_expert",
                    save_steps=config.get("save_steps", 500)
                )

                # Save completion status
                completion_path = os.path.join(log_dir, "training_complete.json")
                with open(completion_path, 'w') as f:
                    json.dump({
                        "status": "complete",
                        "timestamp": time.time(),
                        "model_path": model_path
                    }, f)

                logger.info(f"LLM training completed: {model_path}")
                sys.exit(0)
            else:  # Parent process
                # Return status about the background process
                return {
                    "status": "started",
                    "background": True,
                    "pid": pid,
                    "log_dir": log_dir,
                    "tensorboard": f"tensorboard --logdir={config.get('output_dir', str(self.llm_models_dir))}"
                }
        else:
            # Run in foreground
            try:
                # Prepare the training data
                self.llm_trainer.prepare_training_data(config.get("train_dataset_path"))

                # Start the training
                model_path = self.llm_trainer.train_model(
                    epochs=config.get("num_train_epochs", 3),
                    batch_size=config.get("per_device_train_batch_size", 4),
                    learning_rate=config.get("learning_rate", 1e-5),
                    model_name="biomechanics_expert_foreground",
                    save_steps=config.get("save_steps", 500)
                )

                return {
                    "status": "complete",
                    "background": False,
                    "model_path": model_path
                }
            except Exception as e:
                logger.error(f"Error in LLM training: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }

    def check_training_progress(self, log_dir: str = None) -> Dict[str, Any]:
        """
        Check the progress of a background training process.

        Args:
            log_dir: Directory containing training logs

        Returns:
            Status dictionary with progress information
        """
        if log_dir is None:
            log_dir = str(self.logs_dir / "llm_training")

        logger.info(f"Checking training progress in: {log_dir}")

        # Check if the completion file exists
        completion_path = os.path.join(log_dir, "training_complete.json")
        if os.path.exists(completion_path):
            try:
                with open(completion_path, 'r') as f:
                    completion_data = json.load(f)
                return {
                    "status": "complete",
                    "completion_data": completion_data
                }
            except Exception as e:
                logger.error(f"Error reading completion file: {str(e)}")

        # If the training is still running, check the checkpoint dir
        from src.utils.checkpoint_validator import validate_checkpoints
        from src.models.model_evaluator import evaluate_latest_checkpoint

        checkpoint_dir = str(self.llm_models_dir / "checkpoints")
        valid_checkpoints = validate_checkpoints(checkpoint_dir)

        if valid_checkpoints:
            latest_checkpoint = valid_checkpoints[-1]

            # Try to get metrics
            try:
                # This would use the model evaluator
                metrics = evaluate_latest_checkpoint(latest_checkpoint, None)
                return {
                    "status": "in_progress",
                    "latest_checkpoint": latest_checkpoint,
                    "metrics": metrics,
                    "checkpoint_count": len(valid_checkpoints)
                }
            except Exception as e:
                logger.error(f"Error evaluating checkpoint: {str(e)}")
                return {
                    "status": "in_progress",
                    "latest_checkpoint": latest_checkpoint,
                    "checkpoint_count": len(valid_checkpoints),
                    "eval_error": str(e)
                }
        else:
            return {
                "status": "unknown",
                "message": "No valid checkpoints found yet"
            }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of both pipelines.

        Returns:
            Status dictionary with information about both pipelines
        """
        # Check video analysis status
        video_analyses = [f.stem for f in self.video_output_dir.glob("*.json")]

        # Check LLM training status
        llm_training_files = list(self.llm_training_dir.glob("*.json"))
        llm_models = list(self.llm_models_dir.glob("*"))

        # Check for ongoing training
        training_log_dir = self.logs_dir / "llm_training"
        training_status = "not_started"

        if training_log_dir.exists():
            completion_path = training_log_dir / "training_complete.json"
            if completion_path.exists():
                training_status = "completed"
            else:
                training_status = "in_progress"

        # Check core module availability
        core_modules = {
            "pose": self.pose_modules_available if hasattr(self, 'pose_modules_available') else False,
            "dynamics": self.dynamics_modules_available if hasattr(self, 'dynamics_modules_available') else False,
            "motion": self.motion_modules_available if hasattr(self, 'motion_modules_available') else False,
            "scene": self.scene_modules_available if hasattr(self, 'scene_modules_available') else False
        }

        return {
            "video_pipeline": {
                "analyzed_videos": len(video_analyses),
                "video_names": video_analyses,
                "core_modules": core_modules
            },
            "llm_pipeline": {
                "training_files": len(llm_training_files),
                "models": len(llm_models),
                "training_status": training_status
            }
        }

def main():
    """Main entry point for running the Moriarty pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Moriarty Pipeline")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Video analysis command
    video_parser = subparsers.add_parser("analyze_video", help="Run video analysis")
    video_parser.add_argument("--video", required=True, help="Path to video")
    video_parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    video_parser.add_argument("--report", action="store_true", help="Generate report")
    video_parser.add_argument("--llm_data", action="store_true", help="Prepare LLM data")

    # Batch analysis command
    batch_parser = subparsers.add_parser("batch_analyze", help="Run batch video analysis")
    batch_parser.add_argument("--folder", required=True, help="Input folder")
    batch_parser.add_argument("--pattern", default="*.mp4", help="File pattern")

    # LLM training command
    llm_parser = subparsers.add_parser("train_llm", help="Train LLM")
    llm_parser.add_argument("--config", help="Training config path")
    llm_parser.add_argument("--background", action="store_true", help="Run in background")
    llm_parser.add_argument("--resume", help="Resume from checkpoint")

    # Parse arguments
    args = parser.parse_args()

    # Initialize the pipeline
    pipeline = MoriartyPipeline(output_dir="output")

    # Execute the command
    if args.command == "analyze_video":
        # Run video analysis
        result = pipeline.run_sprint_video_analysis(
            video_path=args.video,
            create_visualizations=args.visualize,
            generate_report=args.report,
            prepare_llm_data=args.llm_data
        )

        # Print results
        if result["success"]:
            print(f"✅ Successfully analyzed video: {args.video}")
            print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
            if "report_path" in result:
                print(f"Report saved to: {result['report_path']}")
            if "llm_data_path" in result:
                print(f"LLM training data saved to: {result['llm_data_path']}")
        else:
            print(f"❌ Failed to analyze video: {result.get('error', 'Unknown error')}")

    elif args.command == "batch_analyze":
        # Run batch analysis
        results = pipeline.run_batch_video_analysis(
            input_folder=args.folder,
            file_pattern=args.pattern
        )

        # Print summary
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"Batch Analysis: {success_count}/{len(results)} videos successfully processed")

    elif args.command == "train_llm":
        # Start LLM training
        result = pipeline.start_llm_training(
            config_path=args.config,
            background=args.background,
            resume_from_checkpoint=args.resume
        )

        if result["success"]:
            print(f"✅ LLM training started: {result.get('job_id')}")
            print(f"Log file: {result.get('log_file')}")
        else:
            print(f"❌ Failed to start LLM training: {result.get('error')}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
