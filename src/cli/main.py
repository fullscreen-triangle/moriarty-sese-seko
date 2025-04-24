#!/usr/bin/env python3
"""
Moriarty CLI - Command Line Interface for the Moriarty biomechanics video analysis framework.

This module provides the main entry point for the CLI, handling command-line arguments
and dispatching to the appropriate command handlers.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, List, Optional

# Ensure the src directory is in the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Moriarty modules
from cli.utils import print_status, StatusReporter
from cli.config import load_config, merge_cli_args, validate_config, generate_default_config
from cli.commands import analyze_command, batch_command, distill_command, report_command, visualize_command

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    # Set up logger for this module
    logger = logging.getLogger("moriarty-cli")
    logger.setLevel(numeric_level)
    
    logger.debug("Logging initialized")

def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Moriarty CLI - Command line interface for the Moriarty biomechanics video analysis framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="Moriarty CLI v0.1.0"
    )
    
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="count", 
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str, 
        help="Path to log file"
    )
    
    parser.add_argument(
        "--generate-config", 
        action="store_true",
        help="Generate a default configuration file and exit"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="Analyze a video for biomechanical metrics"
    )
    analyze_parser.add_argument(
        "--input", 
        "-i", 
        type=str, 
        help="Input video file path"
    )
    analyze_parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        help="Output directory for analysis results"
    )
    analyze_parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations during analysis"
    )
    analyze_parser.add_argument(
        "--visualization-types", 
        type=str, 
        nargs="+",
        choices=["pose", "trajectory", "forces", "all"],
        help="Types of visualizations to generate"
    )
    analyze_parser.add_argument(
        "--model", 
        type=str, 
        help="Model to use for analysis (e.g., 'default', 'high-precision')"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", 
        help="Process multiple videos in batch mode"
    )
    batch_parser.add_argument(
        "--input-dir", 
        type=str, 
        help="Input directory containing videos"
    )
    batch_parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for batch results"
    )
    batch_parser.add_argument(
        "--pattern", 
        type=str, 
        default="*.mp4",
        help="File pattern for videos (default: *.mp4)"
    )
    batch_parser.add_argument(
        "--parallel", 
        type=int, 
        default=1,
        help="Number of parallel processes to use"
    )
    
    # Knowledge distillation command
    distill_parser = subparsers.add_parser(
        "distill", 
        help="Run knowledge distillation to train analysis models"
    )
    distill_parser.add_argument(
        "--training-data", 
        type=str, 
        help="Path to training data directory"
    )
    distill_parser.add_argument(
        "--model-output", 
        type=str, 
        help="Path to save trained model"
    )
    distill_parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    distill_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Training batch size"
    )
    
    # Report generation command
    report_parser = subparsers.add_parser(
        "report", 
        help="Generate reports from analysis results"
    )
    report_parser.add_argument(
        "--input", 
        type=str, 
        help="Input directory with analysis results"
    )
    report_parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory for reports"
    )
    report_parser.add_argument(
        "--format", 
        type=str, 
        choices=["pdf", "html", "json"],
        default="pdf",
        help="Report format (default: pdf)"
    )
    report_parser.add_argument(
        "--template", 
        type=str, 
        help="Custom report template path"
    )
    
    # Visualization command
    visualize_parser = subparsers.add_parser(
        "visualize", 
        help="Generate visualizations from analysis results"
    )
    visualize_parser.add_argument(
        "--input", 
        type=str, 
        help="Input directory with analysis results"
    )
    visualize_parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory for visualizations"
    )
    visualize_parser.add_argument(
        "--types", 
        type=str, 
        nargs="+",
        choices=["pose", "trajectory", "forces", "all"],
        default=["all"],
        help="Types of visualizations to generate"
    )
    
    return parser

def handle_generate_config(args: argparse.Namespace) -> int:
    """
    Handle the --generate-config flag.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    output_path = args.config or "moriarty_config.yaml"
    
    try:
        config = generate_default_config()
        
        with open(output_path, "w") as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print_status(f"Default configuration generated at: {output_path}", "success")
        return 0
        
    except Exception as e:
        print_status(f"Failed to generate configuration: {str(e)}", "error")
        logging.error(f"Failed to generate configuration: {str(e)}")
        return 1

def determine_log_level(verbosity: int) -> str:
    """
    Determine the log level based on verbosity.
    
    Args:
        verbosity: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        
    Returns:
        Log level string
    """
    if verbosity == 0:
        return "WARNING"
    elif verbosity == 1:
        return "INFO"
    else:
        return "DEBUG"

def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging based on verbosity
    log_level = determine_log_level(args.verbose)
    setup_logging(log_level, args.log_file)
    
    # Handle --generate-config flag
    if args.generate_config:
        return handle_generate_config(args)
    
    # If no command specified, show help and exit
    if not args.command:
        parser.print_help()
        return 1
    
    # Load configuration file if specified
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print_status(f"Error loading configuration file: {str(e)}", "error")
            logging.error(f"Error loading configuration file: {str(e)}")
            return 1
    
    # Merge command line arguments into configuration
    config = merge_cli_args(config, args)
    
    # Validate configuration
    try:
        validate_config(config, args.command)
    except ValueError as e:
        print_status(f"Configuration error: {str(e)}", "error")
        logging.error(f"Configuration error: {str(e)}")
        return 1
    
    # Create status reporter for the command
    reporter = StatusReporter()
    
    # Execute the appropriate command
    try:
        if args.command == "analyze":
            return analyze_command(config, reporter)
        elif args.command == "batch":
            return batch_command(config, reporter)
        elif args.command == "distill":
            return distill_command(config, reporter)
        elif args.command == "report":
            return report_command(config, reporter)
        elif args.command == "visualize":
            return visualize_command(config, reporter)
        else:
            print_status(f"Unknown command: {args.command}", "error")
            return 1
    except Exception as e:
        print_status(f"Error executing command: {str(e)}", "error")
        logging.exception(f"Error executing command: {str(e)}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main()) 