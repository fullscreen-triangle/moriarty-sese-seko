#!/usr/bin/env python
import os
import argparse
import subprocess
from pathlib import Path
import time
import sys

def process_video(video_path, output_dir="output", use_ray=True):
    """
    Process a video using the visualkinetics package.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Output directory for processed videos
        use_ray (bool): Whether to use Ray for distributed processing
        
    Returns:
        str: Path to the processed video
    """
    start_time = time.time()
    print(f"Processing video: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose the processing script based on whether to use Ray
    if use_ray:
        script = "process_one_video.py"
    else:
        script = "main.py"
    
    # Run the video processing
    try:
        cmd = ["python", script, video_path, "--output", output_dir]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Extract the output file path from the output
        for line in result.stdout.split("\n"):
            if "Output saved to" in line:
                output_file = line.split("Output saved to")[-1].strip()
                break
        else:
            # If we can't find the output file path, assume it's in the output directory
            video_name = Path(video_path).name
            output_file = os.path.join(output_dir, f"annotated_{video_name}")
        
        elapsed = time.time() - start_time
        print(f"Video processing completed in {elapsed:.2f} seconds")
        return output_file
    
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

def find_latest_model(models_dir="models"):
    """Find the most recently created pose model file."""
    model_files = list(Path(models_dir).glob("*_model.pth"))
    if not model_files:
        return None
    
    # Sort by modification time, newest first
    latest_model = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return latest_model

def analyze_model(model_path, api="openai", sport_type=None, output_dir="pose_analysis_results"):
    """
    Analyze a pose model using AI APIs.
    
    Args:
        model_path (str): Path to the pose model file
        api (str): API provider to use ('openai' or 'anthropic')
        sport_type (str, optional): Type of sport for context
        output_dir (str): Directory to save analysis results
        
    Returns:
        bool: Whether the analysis was successful
    """
    start_time = time.time()
    print(f"Analyzing model: {model_path}")
    
    try:
        cmd = [
            "python", "pose_analysis_api.py",
            "--api", api,
            "--single_model", Path(model_path).name,
            "--model_dir", str(Path(model_path).parent),
            "--output_dir", output_dir
        ]
        
        if sport_type:
            cmd.extend(["--sport_type", sport_type])
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        elapsed = time.time() - start_time
        print(f"Model analysis completed in {elapsed:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing model: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return False

def run_pipeline(video_path, api="openai", sport_type=None, use_ray=True, output_dir="pose_analysis_results"):
    """
    Run the complete pipeline from video processing to model analysis.
    
    Args:
        video_path (str): Path to the video file
        api (str): API provider to use ('openai' or 'anthropic')
        sport_type (str, optional): Type of sport for context
        use_ray (bool): Whether to use Ray for distributed processing
        output_dir (str): Directory to save analysis results
        
    Returns:
        bool: Whether the pipeline was successful
    """
    print(f"Starting analysis pipeline for video: {video_path}")
    print(f"API provider: {api}")
    print(f"Sport type: {sport_type or 'Not specified'}")
    
    # Step 1: Process the video
    processed_video = process_video(video_path, use_ray=use_ray)
    if processed_video is None:
        print("Video processing failed. Stopping pipeline.")
        return False
    
    # Step 2: Find the latest model file (created from processing the video)
    latest_model = find_latest_model()
    if latest_model is None:
        print("No model file found. Stopping pipeline.")
        return False
    
    print(f"Found latest model: {latest_model}")
    
    # Step 3: Analyze the model
    if not analyze_model(latest_model, api, sport_type, output_dir):
        print("Model analysis failed. Stopping pipeline.")
        return False
    
    print("Complete pipeline executed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process a video and analyze it with AI APIs")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--api", choices=["openai", "anthropic"], default="openai", help="API provider to use")
    parser.add_argument("--sport_type", help="Type of sport in the video (e.g., running, jumping, martial_arts)")
    parser.add_argument("--no_ray", action="store_true", help="Disable Ray for distributed processing")
    parser.add_argument("--output_dir", default="pose_analysis_results", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Check if the video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Run the pipeline
    success = run_pipeline(
        args.video_path,
        api=args.api,
        sport_type=args.sport_type,
        use_ray=not args.no_ray,
        output_dir=args.output_dir
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 