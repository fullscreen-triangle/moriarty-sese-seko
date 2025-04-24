#!/usr/bin/env python3
"""
Moriarty Pipeline Command Line Interface

This script provides a simple command-line interface to run the Moriarty Pipeline.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add the parent directory to the path to find the src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the pipeline orchestrator
from src.moriarty_pipeline import MoriartyPipeline

def main():
    """Main entry point for the Moriarty Pipeline CLI."""
    parser = argparse.ArgumentParser(description="Moriarty Pipeline Command Line Interface")
    
    # Define top-level commands
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to run")
    
    # Video Analysis command
    video_parser = subparsers.add_parser("analyze-video", help="Run sprint video analysis")
    video_parser.add_argument("video_path", help="Path to the video file")
    video_parser.add_argument("--output-dir", default="output", help="Output directory")
    video_parser.add_argument("--no-visualize", action="store_true", help="Disable visualizations")
    video_parser.add_argument("--no-report", action="store_true", help="Disable report generation")
    video_parser.add_argument("--no-llm-data", action="store_true", help="Disable LLM data preparation")
    
    # Batch Video Analysis command
    batch_parser = subparsers.add_parser("analyze-batch", help="Run batch video analysis")
    batch_parser.add_argument("input_folder", help="Folder containing videos")
    batch_parser.add_argument("--pattern", default="*.mp4", help="File pattern to match")
    batch_parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # LLM Training Setup command
    setup_parser = subparsers.add_parser("setup-llm", help="Set up LLM training")
    setup_parser.add_argument("--base-model", default="facebook/opt-1.3b", help="Base model for fine-tuning")
    setup_parser.add_argument("--cud-dataset", help="Path to CUD dataset")
    setup_parser.add_argument("--maxplanck-dataset", help="Path to MAXPLANCK dataset")
    setup_parser.add_argument("--nomo-dataset", help="Path to NOMO dataset")
    setup_parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # LLM Training Start command
    train_parser = subparsers.add_parser("train-llm", help="Start LLM training")
    train_parser.add_argument("--config", help="Path to training configuration")
    train_parser.add_argument("--foreground", action="store_true", help="Run in foreground instead of background")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    train_parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # LLM Training Status command
    status_parser = subparsers.add_parser("training-status", help="Check LLM training status")
    status_parser.add_argument("--log-dir", help="Path to log directory")
    status_parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # Pipeline status command
    pipeline_status_parser = subparsers.add_parser("status", help="Get overall pipeline status")
    pipeline_status_parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize the pipeline
    pipeline = MoriartyPipeline(output_dir=args.output_dir)
    
    # Execute the requested command
    if args.command == "analyze-video":
        # Run video analysis
        result = pipeline.run_sprint_video_analysis(
            video_path=args.video_path,
            create_visualizations=not args.no_visualize,
            generate_report=not args.no_report,
            prepare_llm_data=not args.no_llm_data
        )
        print(json.dumps(result, indent=2))
        
    elif args.command == "analyze-batch":
        # Run batch analysis
        results = pipeline.run_batch_video_analysis(
            input_folder=args.input_folder,
            file_pattern=args.pattern
        )
        print(f"Processed {len(results)} videos")
        print(json.dumps(results, indent=2))
        
    elif args.command == "setup-llm":
        # Set up LLM training
        datasets = {}
        if args.cud_dataset:
            datasets["cud"] = args.cud_dataset
        if args.maxplanck_dataset:
            datasets["maxplanck"] = args.maxplanck_dataset
        if args.nomo_dataset:
            datasets["nomo"] = args.nomo_dataset
            
        result = pipeline.setup_llm_training(
            base_model=args.base_model,
            datasets=datasets if datasets else None
        )
        print(json.dumps(result, indent=2))
        
    elif args.command == "train-llm":
        # Start LLM training
        result = pipeline.start_llm_training(
            config_path=args.config,
            background=not args.foreground,
            resume_from_checkpoint=args.resume
        )
        print(json.dumps(result, indent=2))
        
    elif args.command == "training-status":
        # Check training status
        result = pipeline.check_training_progress(args.log_dir)
        print(json.dumps(result, indent=2))
        
    elif args.command == "status":
        # Get overall pipeline status
        result = pipeline.get_pipeline_status()
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main() 