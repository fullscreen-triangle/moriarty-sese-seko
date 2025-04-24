"""
Moriarty - Video Processing and LLM Training Pipeline

Main entry point for the moriarty package.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("moriarty")

# Import the pipeline module
from .pipeline import VideoPipeline, DEFAULT_MEMORY_LIMIT, DEFAULT_BATCH_SIZE

def main():
    """Main entry point for the moriarty package."""
    parser = argparse.ArgumentParser(description='Moriarty Video Processing and LLM Training Pipeline')
    
    # General options
    parser.add_argument('--video', type=str, help='Path to a specific video file to process')
    parser.add_argument('--input', type=str, default='public', 
                       help='Input folder containing videos (default: public)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output folder for processed videos (default: output)')
    parser.add_argument('--models', type=str, default='models',
                       help='Output folder for pose models (default: models)')
    parser.add_argument('--llm_data', type=str, default='llm_training_data',
                       help='Output folder for LLM training data (default: llm_training_data)')
    parser.add_argument('--llm_models', type=str, default='llm_models',
                       help='Output folder for trained LLM models (default: llm_models)')
    
    # Resource management options
    parser.add_argument('--memory_limit', type=float, default=DEFAULT_MEMORY_LIMIT,
                       help=f'Memory limit as a fraction of total system memory (default: {DEFAULT_MEMORY_LIMIT})')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes/threads (default: auto)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for frame processing (default: {DEFAULT_BATCH_SIZE})')
    
    # Processing options
    parser.add_argument('--no_video', action='store_true',
                       help='Do not generate annotated videos, just pose models')
    parser.add_argument('--train_llm', action='store_true',
                       help='Train LLM on extracted pose data')
    parser.add_argument('--sport_type', type=str, default=None,
                       help='Type of sport in the video (for context in LLM training)')
    parser.add_argument('--use_openai', action='store_true',
                       help='Use OpenAI API for synthetic data generation')
    parser.add_argument('--use_claude', action='store_true',
                       help='Use Claude API for synthetic data generation')
    parser.add_argument('--both_llms', action='store_true',
                       help='Use both OpenAI and Claude for synthetic data generation')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VideoPipeline(
        memory_limit_fraction=args.memory_limit,
        n_workers=args.workers,
        batch_size=args.batch_size,
        output_dir=args.output,
        model_dir=args.models,
        llm_training_dir=args.llm_data,
        llm_model_dir=args.llm_models
    )
    
    # Process videos
    processed_models = []
    if args.video:
        # Process single video
        logger.info(f"Processing single video: {args.video}")
        result = pipeline.process_video(
            args.video,
            output_annotations=not args.no_video
        )
        if result.get("pose_model_path"):
            processed_models.append(result["pose_model_path"])
            logger.info(f"Processed {result['frame_count']}/{result['total_frames']} frames in {result['processing_time']:.2f} seconds")
    else:
        # Process all videos in folder
        logger.info(f"Processing all videos in: {args.input}")
        results = pipeline.process_all_videos(args.input)
        processed_models = [r["pose_model_path"] for r in results if r.get("pose_model_path")]
        
        if processed_models:
            logger.info(f"Successfully processed {len(processed_models)} videos")
        else:
            logger.warning("No videos were successfully processed")
    
    # Generate LLM training data
    if processed_models and (args.train_llm or args.use_openai or args.use_claude or args.both_llms):
        logger.info(f"Generating training data from {len(processed_models)} pose models")
        training_data = pipeline.generate_llm_training_data(
            processed_models, 
            sport_type=args.sport_type
        )
        
        if not training_data:
            logger.error("Failed to generate training data")
            return
            
        logger.info(f"Training data generated: {training_data}")
        
        # Train LLM or generate synthetic data
        use_openai = args.use_openai or args.both_llms
        use_claude = args.use_claude or args.both_llms
        
        if use_openai or use_claude or args.train_llm:
            logger.info("Training LLM/generating synthetic data")
            training_results = pipeline.train_llm(
                training_data_path=training_data,
                use_openai=use_openai,
                use_claude=use_claude
            )
            
            if training_results["success"]:
                if training_results.get("openai_model"):
                    logger.info(f"OpenAI synthetic data generated: {training_results['openai_model']}")
                if training_results.get("claude_model"):
                    logger.info(f"Claude synthetic data generated: {training_results['claude_model']}")
                if training_results.get("local_model"):
                    logger.info(f"Local model trained: {training_results['local_model']}")
            else:
                logger.error(f"LLM training failed: {training_results.get('error', 'Unknown error')}")
    
    logger.info("Pipeline execution complete!")

if __name__ == "__main__":
    main() 