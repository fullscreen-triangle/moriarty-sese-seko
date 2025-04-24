#!/usr/bin/env python
import argparse
import os
import json
from pathlib import Path
import sys

def main():
    """
    Command-line interface for extracting pose data from models and converting to LLM training data.
    """
    parser = argparse.ArgumentParser(
        description="Extract pose data from models and convert to LLM training data"
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models",
        help="Directory containing pose model files"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="llm_training_data",
        help="Directory to save LLM training data"
    )
    
    parser.add_argument(
        "--sport_type", 
        type=str, 
        choices=["running", "jumping", "swimming", "martial_arts", "basketball", "soccer", "tennis", "golf", "general"],
        default="general",
        help="Type of sport for specialized prompts"
    )
    
    parser.add_argument(
        "--single_model", 
        type=str, 
        help="Process only a single model file (provide the filename)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed information during processing"
    )
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' does not exist")
        return 1
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Import the extractor
        from pose_data_to_llm import PoseDataExtractor
        
        # Create an extractor instance
        extractor = PoseDataExtractor(
            model_dir=args.model_dir,
            output_dir=args.output_dir
        )
        
        # Process models
        if args.single_model:
            model_path = os.path.join(args.model_dir, args.single_model)
            if not os.path.exists(model_path):
                print(f"Error: Model file '{model_path}' does not exist")
                return 1
                
            print(f"Processing single model: {model_path}")
            raw_data = extractor.extract_from_model(model_path)
            descriptions = extractor.convert_to_text_descriptions(raw_data, args.sport_type)
            examples = extractor.prepare_training_examples(descriptions, args.sport_type)
            
            # Save examples to a file
            model_name = Path(args.single_model).stem
            output_file = os.path.join(args.output_dir, f"{model_name}_examples.json")
            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)
                
            print(f"Generated {len(examples)} training examples")
            print(f"Saved to: {output_file}")
            
        else:
            print(f"Processing all models in: {args.model_dir}")
            examples = extractor.process_all_models(args.sport_type, verbose=args.verbose)
            print(f"Generated {len(examples)} training examples from all models")
            
        return 0
            
    except ImportError:
        print("Error: Could not import PoseDataExtractor. Make sure pose_data_to_llm.py is available.")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 