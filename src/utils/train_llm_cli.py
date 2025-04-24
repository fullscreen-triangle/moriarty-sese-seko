#!/usr/bin/env python
import argparse
import os
import json
import sys
import torch
from pathlib import Path

def main():
    """
    Command-line interface for training or updating a language model on pose data.
    """
    parser = argparse.ArgumentParser(
        description="Train or update a language model on pose data"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="llm_training_data",
        help="Directory containing LLM training data"
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="llm_models",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--base_model", 
        type=str, 
        default=None,
        help="Base model to use (e.g., facebook/opt-125m, facebook/opt-350m)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        help="Specific data file to use for training (otherwise uses all data in data_dir)"
    )
    
    parser.add_argument(
        "--progressive", 
        action="store_true",
        help="Use progressive training (update existing model)"
    )
    
    parser.add_argument(
        "--quantize", 
        action="store_true",
        help="Use 4-bit quantization (requires bitsandbytes)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test the model after training with sample prompts"
    )
    
    parser.add_argument(
        "--test_prompt", 
        type=str,
        default="Describe the motion pattern in this sequence:",
        help="Prompt to use for testing the model"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for training (cpu, cuda, or auto)"
    )
    
    parser.add_argument(
        "--save_config",
        action="store_true",
        help="Save the configuration used for training"
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available when device is set to cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    # If auto, determine the device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print device information
    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for training (this will be slow)")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return 1
    
    # If data_file is specified, check if it exists
    if args.data_file and not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' does not exist")
        return 1
    
    try:
        # Import the trainer
        from src.models.train_llm import LLMTrainer
        
        # Determine the base model if not specified
        if args.base_model is None:
            if args.device == "cuda":
                args.base_model = "facebook/opt-1.3b"  # Medium-sized model for GPU
            else:
                args.base_model = "facebook/opt-125m"  # Tiny model for CPU
        
        # Create a trainer instance
        trainer = LLMTrainer(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            base_model=args.base_model,
            device=args.device
        )
        
        # Save configuration if requested
        if args.save_config:
            config = {
                "data_dir": args.data_dir,
                "model_dir": args.model_dir,
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "device": args.device,
                "quantize": args.quantize,
                "progressive": args.progressive
            }
            trainer.save_config(config)
        
        # Train the model
        if args.progressive:
            if args.data_file:
                trainer.progressive_training(
                    new_data_file=args.data_file,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate
                )
            else:
                print("Error: Progressive training requires a specific data file (--data_file)")
                return 1
        else:
            # Prepare the training data
            trainer.prepare_training_data(input_file=args.data_file)
            
            # Set up the model with quantization if requested
            trainer.setup_model_and_tokenizer(use_4bit=args.quantize)
            
            # Train the model
            trainer.train_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        
        # Test the model if requested
        if args.test:
            response = trainer.generate_response(args.test_prompt)
            print("\nModel test:")
            print(f"Prompt: {args.test_prompt}")
            print(f"Response: {response}\n")
        
        print(f"Model training completed successfully. Model saved to {args.model_dir}")
        return 0
        
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure train_llm.py is available and all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 