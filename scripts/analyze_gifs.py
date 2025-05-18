#!/usr/bin/env python3
"""
Analyze GIFs Script for Moriarty

This script processes GIF files from the public/results folder to extract frames
and analyze biomechanical data that might be visible in the animations.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def extract_frames_from_gif(gif_path, output_dir=None):
    """
    Extract individual frames from a GIF file.
    
    Args:
        gif_path: Path to the GIF file
        output_dir: Directory to save extracted frames (default: <gif_name>_frames)
        
    Returns:
        List of paths to extracted frames
    """
    gif_name = os.path.splitext(os.path.basename(gif_path))[0]
    
    if not output_dir:
        output_dir = f"{gif_name}_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting frames from {gif_path}")
    
    # Open the GIF
    gif = Image.open(gif_path)
    
    frame_paths = []
    
    # Loop through the frames
    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        frame = frame.convert('RGB')  # Convert to RGB mode
        frame.save(frame_path, format='PNG')
        frame_paths.append(frame_path)
        
        if i % 10 == 0:
            print(f"Extracted {i} frames...")
            
    print(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths


def analyze_frame_for_biomechanics(frame_path):
    """
    Analyze a frame for biomechanical data.
    
    This extracts information from the frame as if it contains overlaid
    biomechanical data.
    
    Args:
        frame_path: Path to the frame image
        
    Returns:
        Dictionary with biomechanical data
    """
    # Load the frame
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Could not read frame: {frame_path}")
        return {}
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Example analysis: This is a placeholder for actual CV analysis
    # In a real implementation, you would use computer vision techniques
    # to detect skeletal points, joint angles, velocities, etc.
    
    # For demonstration, let's calculate some basic image statistics
    # that could correlate with biomechanical features
    
    # Image regions: top = header info, middle = athlete, bottom = footer info
    h, w = gray.shape
    top_region = gray[0:int(h*0.2), :]
    middle_region = gray[int(h*0.2):int(h*0.8), :]
    bottom_region = gray[int(h*0.8):, :]
    
    # Check for text in top and bottom regions (could contain metrics)
    # This is a very simplified approach - real OCR would be better
    top_text_detected = np.std(top_region) > 50  # Higher std dev suggests text
    bottom_text_detected = np.std(bottom_region) > 50
    
    # Look for potential pose keypoints in middle region
    # Here we're just using a simple threshold and contour detection
    # Real pose estimation would use a specialized model
    _, thresh = cv2.threshold(middle_region, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find potential keypoints (small, circular)
    keypoints = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 100:  # Reasonable size for a keypoint
            x, y, w, h = cv2.boundingRect(contour)
            if 0.5 < w/h < 2.0:  # Roughly square or circular
                center_x = x + w//2
                center_y = y + h//2 + int(gray.shape[0]*0.2)  # Adjust for region offset
                keypoints.append((center_x, center_y))
    
    # Calculate biomechanical proxies from keypoint distribution
    bio_data = {}
    
    if keypoints:
        # Convert to numpy array for calculations
        kp_array = np.array(keypoints)
        
        # Basic metrics from keypoint distribution
        kp_centroid = np.mean(kp_array, axis=0)
        kp_spread = np.std(kp_array, axis=0)
        
        # Count keypoints
        bio_data['keypoints_detected'] = len(keypoints)
        bio_data['centroid_x'] = float(kp_centroid[0])
        bio_data['centroid_y'] = float(kp_centroid[1])
        bio_data['spread_x'] = float(kp_spread[0])
        bio_data['spread_y'] = float(kp_spread[1])
        
        # Calculate height (proxy for athlete height)
        if len(keypoints) > 1:
            heights = [y for _, y in keypoints]
            bio_data['keypoint_height_range'] = float(max(heights) - min(heights))
    
    # Add text-region data
    bio_data['metrics_in_header'] = top_text_detected
    bio_data['metrics_in_footer'] = bottom_text_detected
    
    # Add frame path info
    bio_data['frame_path'] = frame_path
    bio_data['frame_number'] = int(os.path.splitext(os.path.basename(frame_path))[0].split('_')[1])
    
    return bio_data


def process_gif_sequence(gif_path, output_dir=None):
    """
    Process a GIF file to extract and analyze biomechanical data from the sequence.
    
    Args:
        gif_path: Path to the GIF file
        output_dir: Directory to save analysis results
        
    Returns:
        Path to the analysis results file
    """
    gif_name = os.path.splitext(os.path.basename(gif_path))[0]
    
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(gif_path), f"{gif_name}_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frames_dir = os.path.join(output_dir, "frames")
    frame_paths = extract_frames_from_gif(gif_path, frames_dir)
    
    print(f"Analyzing {len(frame_paths)} frames for biomechanical data...")
    
    # Analyze each frame
    frame_data = []
    for i, frame_path in enumerate(frame_paths):
        if i % 10 == 0:
            print(f"Analyzing frame {i}/{len(frame_paths)}...")
        
        bio_data = analyze_frame_for_biomechanics(frame_path)
        if bio_data:
            frame_data.append(bio_data)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(frame_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{gif_name}_frame_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{gif_name}_frame_data.json")
    df.to_json(json_path, orient='records')
    
    # Create basic visualizations
    if not df.empty:
        # Plot keypoint positions over time if available
        if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
            plt.figure(figsize=(14, 8))
            plt.plot(df['frame_number'], df['centroid_y'], 'o-', label='Vertical Position')
            plt.xlabel('Frame Number')
            plt.ylabel('Vertical Position (pixels)')
            plt.title('Body Position Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{gif_name}_position.png"), dpi=300)
            plt.close()
        
        # Plot keypoint count over time
        if 'keypoints_detected' in df.columns:
            plt.figure(figsize=(14, 8))
            plt.plot(df['frame_number'], df['keypoints_detected'], 'o-')
            plt.xlabel('Frame Number')
            plt.ylabel('Keypoints Detected')
            plt.title('Pose Detection Quality Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{gif_name}_keypoints.png"), dpi=300)
            plt.close()
        
        # Plot height range over time if available
        if 'keypoint_height_range' in df.columns:
            plt.figure(figsize=(14, 8))
            plt.plot(df['frame_number'], df['keypoint_height_range'], 'o-')
            plt.xlabel('Frame Number')
            plt.ylabel('Height Range (pixels)')
            plt.title('Estimated Body Height Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{gif_name}_height.png"), dpi=300)
            plt.close()
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return json_path


def process_multiple_gifs(gif_dir, pattern="*.gif", output_dir=None):
    """
    Process multiple GIF files in a directory.
    
    Args:
        gif_dir: Directory containing GIF files
        pattern: Glob pattern to match GIF files
        output_dir: Directory to save analysis results
        
    Returns:
        List of analysis result paths
    """
    import glob
    
    # Find all GIFs
    gif_paths = glob.glob(os.path.join(gif_dir, pattern))
    
    if not gif_paths:
        print(f"No GIF files found in {gif_dir} matching pattern {pattern}")
        return []
    
    print(f"Found {len(gif_paths)} GIF files to process")
    
    # Create output directory
    if not output_dir:
        output_dir = os.path.join(gif_dir, "gif_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each GIF
    result_paths = []
    for gif_path in gif_paths:
        gif_name = os.path.splitext(os.path.basename(gif_path))[0]
        gif_output_dir = os.path.join(output_dir, gif_name)
        result_path = process_gif_sequence(gif_path, gif_output_dir)
        result_paths.append(result_path)
    
    # Create a summary of all GIFs
    summary = {
        "processed_gifs": len(gif_paths),
        "gif_paths": gif_paths,
        "result_paths": result_paths
    }
    
    summary_path = os.path.join(output_dir, "gif_analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processed {len(gif_paths)} GIF files. Summary saved to {summary_path}")
    return result_paths


def main():
    parser = argparse.ArgumentParser(description='Analyze GIFs for biomechanical data.')
    
    parser.add_argument('--gif-dir', '-d', type=str, default='public/results/gif',
                        help='Directory containing GIF files (default: public/results/gif)')
    
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Directory to save analysis results')
    
    parser.add_argument('--gif-file', '-f', type=str,
                        help='Process a specific GIF file instead of a directory')
    
    parser.add_argument('--pattern', '-p', type=str, default='*.gif',
                        help='Glob pattern to match GIF files (default: *.gif)')
    
    args = parser.parse_args()
    
    if args.gif_file:
        # Process a single GIF file
        process_gif_sequence(args.gif_file, args.output_dir)
    else:
        # Process multiple GIF files
        process_multiple_gifs(args.gif_dir, args.pattern, args.output_dir)


if __name__ == "__main__":
    main() 