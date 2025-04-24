#!/usr/bin/env python3

import os
import json
import numpy as np
import logging
from pathlib import Path
import argparse
from dotenv import load_dotenv
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_biomechanics_example")

# Add parent directory to path to allow imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.core.dynamics.ai_dynamics_analyzer import AIDynamicsAnalyzer

def generate_sample_data(frames=100):
    """Generate sample biomechanical data for testing."""
    positions_batch = []
    velocities_batch = []
    accelerations_batch = []
    
    for i in range(frames):
        # Generate sample positions for one frame
        positions = {
            'foot': np.array([0.2 * np.sin(i/10), 0.1 + 0.05 * np.sin(i/8)]),
            'shank': np.array([0.1 * np.sin(i/12), 0.5 + 0.1 * np.sin(i/10)]),
            'thigh': np.array([0.05 * np.sin(i/15), 1.0 + 0.15 * np.sin(i/12)])
        }
        
        # Generate sample velocities
        velocities = {
            'foot': np.array([0.2 * np.cos(i/10) / 10, 0.05 * np.cos(i/8) / 8]),
            'shank': np.array([0.1 * np.cos(i/12) / 12, 0.1 * np.cos(i/10) / 10]),
            'thigh': np.array([0.05 * np.cos(i/15) / 15, 0.15 * np.cos(i/12) / 12])
        }
        
        # Generate sample accelerations
        accelerations = {
            'foot': np.array([-0.2 * np.sin(i/10) / 100, -0.05 * np.sin(i/8) / 64]),
            'shank': np.array([-0.1 * np.sin(i/12) / 144, -0.1 * np.sin(i/10) / 100]),
            'thigh': np.array([-0.05 * np.sin(i/15) / 225, -0.15 * np.sin(i/12) / 144])
        }
        
        positions_batch.append(positions)
        velocities_batch.append(velocities)
        accelerations_batch.append(accelerations)
    
    return positions_batch, velocities_batch, accelerations_batch

def generate_sample_pose_sequence(frames=20):
    """Generate sample pose sequence data for movement pattern analysis."""
    pose_sequence = []
    
    # Sample keypoints for pose data
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    for i in range(frames):
        # Generate positions with some variation to simulate movement
        keypoint_data = {}
        for kp in keypoints:
            # Add some variation based on frame number to simulate movement
            x = 0.5 + 0.1 * np.sin(i/5) + np.random.normal(0, 0.02)
            y = 0.5 + 0.2 * np.sin(i/8) + np.random.normal(0, 0.02)
            
            # Adjust positions based on keypoint type
            if "shoulder" in kp:
                y -= 0.2
            elif "hip" in kp:
                y += 0.1
            elif "knee" in kp:
                y += 0.3
            elif "ankle" in kp:
                y += 0.5
            
            # Add left/right variation
            if "left" in kp:
                x -= 0.1
            elif "right" in kp:
                x += 0.1
                
            keypoint_data[kp] = {"x": x, "y": y, "confidence": 0.9 + np.random.normal(0, 0.05)}
        
        # Add timestamp and frame info
        frame_data = {
            "frame": i,
            "timestamp": i / 30.0,  # Assuming 30fps
            "keypoints": keypoint_data
        }
        
        pose_sequence.append(frame_data)
    
    return pose_sequence

def generate_sample_sprint_data():
    """Generate sample sprint-specific data."""
    return {
        "blocks": {
            "reaction_time": 0.154,
            "block_forces": [650, 720, 850],
            "block_angles": {"hip": 45, "knee": 100}
        },
        "splits": {
            "10m": 1.83,
            "20m": 2.96,
            "30m": 3.92,
            "40m": 4.85,
            "50m": 5.76,
            "60m": 6.68
        },
        "stride_data": [
            {"distance": 5, "length": 1.65, "frequency": 4.2, "contact_time": 0.145},
            {"distance": 15, "length": 1.95, "frequency": 4.5, "contact_time": 0.125},
            {"distance": 25, "length": 2.15, "frequency": 4.7, "contact_time": 0.115},
            {"distance": 35, "length": 2.25, "frequency": 4.8, "contact_time": 0.11},
            {"distance": 45, "length": 2.28, "frequency": 4.7, "contact_time": 0.112},
            {"distance": 55, "length": 2.25, "frequency": 4.6, "contact_time": 0.115}
        ],
        "force_plate_data": {
            "horizontal_forces": [450, 650, 820, 850, 830, 810],
            "vertical_forces": [1500, 2200, 2500, 2550, 2500, 2450]
        }
    }

def generate_sample_performance_data():
    """Generate sample performance data."""
    current_data = {
        "date": "2023-07-15",
        "sprint_times": {
            "10m": 1.81,
            "30m": 3.89,
            "60m": 6.65
        },
        "stride_metrics": {
            "max_length": 2.28,
            "max_frequency": 4.8,
            "min_contact_time": 0.11
        },
        "force_metrics": {
            "max_horizontal": 850,
            "max_vertical": 2550
        },
        "technical_scores": {
            "acceleration": 8.5,
            "maximum_velocity": 8.7,
            "overall": 8.6
        }
    }
    
    previous_data = [
        {
            "date": "2023-06-01",
            "sprint_times": {
                "10m": 1.85,
                "30m": 3.94,
                "60m": 6.72
            },
            "stride_metrics": {
                "max_length": 2.25,
                "max_frequency": 4.7,
                "min_contact_time": 0.113
            },
            "force_metrics": {
                "max_horizontal": 830,
                "max_vertical": 2520
            },
            "technical_scores": {
                "acceleration": 8.2,
                "maximum_velocity": 8.5,
                "overall": 8.3
            }
        },
        {
            "date": "2023-04-15",
            "sprint_times": {
                "10m": 1.88,
                "30m": 3.98,
                "60m": 6.78
            },
            "stride_metrics": {
                "max_length": 2.22,
                "max_frequency": 4.6,
                "min_contact_time": 0.115
            },
            "force_metrics": {
                "max_horizontal": 810,
                "max_vertical": 2480
            },
            "technical_scores": {
                "acceleration": 8.0,
                "maximum_velocity": 8.3,
                "overall": 8.1
            }
        }
    ]
    
    return current_data, previous_data

def save_results(results, filename):
    """Save results to a JSON file."""
    path = Path(filename)
    path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to {path}")

def main():
    """Main function to demonstrate AI biomechanical analysis."""
    parser = argparse.ArgumentParser(description="AI-powered biomechanical analysis example")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to generate")
    parser.add_argument("--output", type=str, default="output/ai_results", help="Output directory for results")
    parser.add_argument("--api-token", type=str, help="Hugging Face API token (overrides env variable)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the AI dynamics analyzer
    analyzer = AIDynamicsAnalyzer(api_token=args.api_token)
    
    # Generate sample data
    logger.info("Generating sample biomechanical data...")
    positions, velocities, accelerations = generate_sample_data(args.frames)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Example 1: Basic biomechanical analysis
    logger.info("Performing AI-powered biomechanical analysis...")
    athlete_info = {
        "name": "John Doe",
        "age": 25,
        "height": 185,
        "weight": 75,
        "sport": "sprint",
        "level": "elite"
    }
    
    try:
        biomech_results = analyzer.analyze_biomechanics(
            positions_batch=positions,
            velocities_batch=velocities,
            accelerations_batch=accelerations,
            athlete_info=athlete_info
        )
        save_results(biomech_results, f"{args.output}/biomechanical_analysis.json")
    except Exception as e:
        logger.error(f"Error in biomechanical analysis: {str(e)}")
    
    # Example 2: Movement pattern analysis
    logger.info("Analyzing movement patterns...")
    pose_sequence = generate_sample_pose_sequence()
    
    try:
        movement_results = analyzer.analyze_movement_patterns(pose_sequence)
        save_results(movement_results, f"{args.output}/movement_pattern_analysis.json")
    except Exception as e:
        logger.error(f"Error in movement pattern analysis: {str(e)}")
    
    # Example 3: Sprint technique analysis
    logger.info("Analyzing sprint technique...")
    sprint_data = generate_sample_sprint_data()
    
    try:
        sprint_results = analyzer.analyze_sprint_technique(sprint_data)
        save_results(sprint_results, f"{args.output}/sprint_technique_analysis.json")
    except Exception as e:
        logger.error(f"Error in sprint technique analysis: {str(e)}")
    
    # Example 4: Performance comparison
    logger.info("Comparing performances...")
    current_data, previous_data = generate_sample_performance_data()
    
    try:
        comparison_results = analyzer.compare_performances(current_data, previous_data)
        save_results(comparison_results, f"{args.output}/performance_comparison.json")
    except Exception as e:
        logger.error(f"Error in performance comparison: {str(e)}")
    
    # Example 5: Generate technical report
    logger.info("Generating technical report...")
    try:
        if 'biomech_results' in locals():
            report = analyzer.generate_technical_report(
                biomech_results,
                athlete_profile=athlete_info
            )
            save_results(report, f"{args.output}/technical_report.json")
    except Exception as e:
        logger.error(f"Error generating technical report: {str(e)}")
    
    # Example 6: Quick analysis
    logger.info("Performing quick analysis...")
    try:
        quick_results = analyzer.quick_analysis(sprint_data)
        save_results(quick_results, f"{args.output}/quick_analysis.json")
    except Exception as e:
        logger.error(f"Error in quick analysis: {str(e)}")
    
    logger.info("AI biomechanical analysis examples completed.")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 