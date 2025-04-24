#!/usr/bin/env python3

import argparse
import logging
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.core.pose.athlete_analyzer import AthleteAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_athlete_example")

def main():
    """Example script for multi-athlete tracking and comparison."""
    parser = argparse.ArgumentParser(description="Multi-athlete tracking and analysis example")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="output/athletes", help="Output directory for results")
    parser.add_argument("--visualization", type=str, default=None, help="Output path for visualization video")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame for processing")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for processing if available")
    parser.add_argument("--disable_advanced", action="store_true", help="Disable advanced AI-powered analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize athlete analyzer
    logger.info(f"Initializing athlete analyzer (GPU: {args.use_gpu}, Advanced: {not args.disable_advanced})")
    analyzer = AthleteAnalyzer(
        use_gpu=args.use_gpu,
        use_advanced_analysis=not args.disable_advanced,
        cache_dir=str(output_dir / "cache")
    )
    
    # Process video to track and analyze athletes
    logger.info(f"Processing video: {args.video}")
    
    # Optional athlete identities
    # In a real application, this could come from a database or user input
    athlete_identities = {
        "person-0": "Athlete A",
        "person-1": "Athlete B",
        "person-2": "Athlete C"
    }
    
    athletes = analyzer.process_video(
        video_path=args.video,
        athlete_identities=athlete_identities,
        start_frame=args.start_frame,
        max_frames=args.max_frames
    )
    
    logger.info(f"Found {len(athletes)} athletes in the video")
    
    # Save athlete data
    analyzer.save_athlete_data(output_dir=str(output_dir))
    
    # Compare athletes if we have multiple
    if len(athletes) >= 2:
        logger.info("Comparing athletes")
        athlete_ids = list(athletes.keys())
        comparison_results = analyzer.compare_athletes(athlete_ids)
        
        # Save comparison results
        comparison_file = output_dir / "athlete_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Print some comparison highlights
        print("\n===== ATHLETE COMPARISON =====")
        
        # Display the athletes being compared
        print("\nAthletes being compared:")
        for athlete_id, athlete in athletes.items():
            print(f"- {athlete.name} (ID: {athlete_id})")
        
        # Print basic metrics comparison
        print("\nBasic metrics comparison:")
        for athlete_id, athlete_data in comparison_results["data"].items():
            athlete_name = athlete_data["name"]
            metrics = athlete_data["metrics"]
            
            print(f"\n{athlete_name}:")
            for metric_name, value in metrics.items():
                if value is not None:
                    if isinstance(value, (int, float)):
                        print(f"  - {metric_name}: {value:.2f}")
                    else:
                        print(f"  - {metric_name}: {value}")
        
        # Print comparison summary from advanced analysis (if available)
        if "analysis" in comparison_results and "summary" in comparison_results["analysis"]:
            print("\nAnalysis Summary:")
            print(comparison_results["analysis"]["summary"])
        
        print("\nDetailed comparison saved to:", comparison_file)
    
    # Create visualization if requested
    if args.visualization:
        logger.info(f"Creating visualization: {args.visualization}")
        analyzer.visualize_multi_athlete_tracking(
            video_path=args.video,
            output_path=args.visualization,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            show_metrics=True
        )
        
        print(f"\nVisualization saved to: {args.visualization}")


if __name__ == "__main__":
    main() 