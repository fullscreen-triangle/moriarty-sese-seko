#!/usr/bin/env python3
"""
Example usage of the PostureAnalysisSolver for production environments.

This script demonstrates how to use the PostureAnalysisSolver to:
1. Analyze posture data
2. Generate distillation trios
3. Create a distillation dataset

For production use, you should replace the MockLLMClient with your actual
commercial LLM API client (e.g., OpenAI, Anthropic, etc.)
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("posture_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("example_usage")

# Import the posture solver (with error handling)
try:
    from posture_solver import (
        PostureAnalysisSolver, 
        create_distillation_trio, 
        generate_distillation_dataset,
        ModelLoadError,
        AnalysisError
    )
except ImportError as e:
    logger.error(f"Failed to import posture_solver: {str(e)}")
    logger.error("Please make sure the posture_solver.py file is in the same directory or in PYTHONPATH")
    raise


class MockLLMClient:
    """
    Mock implementation of a commercial LLM client for demonstration purposes.
    In production, replace with an actual client for your preferred LLM API.
    """
    def __init__(self, delay: float = 0.5, max_retries: int = 3):
        """
        Initialize the mock LLM client
        
        Args:
            delay: Simulated API delay in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.delay = delay
        self.max_retries = max_retries
        logger.info("Initialized Mock LLM Client (replace with actual API client in production)")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the mock LLM
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The generated response
            
        Raises:
            Exception: If LLM generation fails after max retries
        """
        # Log prompt length to avoid logging full prompt content
        logger.info(f"Sending prompt to LLM (length: {len(prompt)} characters)")
        
        # Simulate API delay
        time.sleep(self.delay)
        
        # In production, this would call the actual API with retries
        for attempt in range(self.max_retries):
            try:
                # Mock response - in production this would be the actual API call
                response = "Based on the biomechanical analysis, I can see that your posture shows a moderate spine curvature issue. The spine alignment model detected an average angle of 147.5 degrees between vertebral segments, which indicates excessive curvature. This is commonly associated with a slouching posture. Additionally, your shoulders show a slight imbalance with the right shoulder being 1.2cm higher than the left. I would recommend exercises to strengthen your core and upper back muscles, particularly focusing on the left side to help balance your shoulders. Regular stretching of chest muscles can also help improve your spine alignment."
                
                logger.info(f"Received response from LLM (length: {len(response)} characters)")
                return response
            except Exception as e:
                logger.warning(f"LLM API call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    time.sleep((attempt + 1) * 2)
                else:
                    logger.error("Max retries exceeded")
                    raise Exception(f"Failed to generate LLM response after {self.max_retries} attempts")


def load_example_posture_data() -> Dict[str, Any]:
    """
    Load example posture data
    
    In production, this would load from your capture system or database
    
    Returns:
        Dictionary containing posture keypoints and metadata
    """
    logger.info("Loading example posture data")
    
    return {
        "keypoints": {
            "spine": [
                [300, 100], 
                [305, 150], 
                [315, 200], 
                [330, 250], 
                [350, 300]
            ],
            "shoulder_left": [250, 152],
            "shoulder_right": [360, 148],
            "hip_center": [320, 400]
        },
        "metadata": {
            "image_dimensions": [640, 480],
            "capture_conditions": "standing",
            "timestamp": time.time(),
            "source": "example_data"
        }
    }


def create_model_directories() -> None:
    """Create directory structure for models if it doesn't exist"""
    logger.info("Creating model directories")
    os.makedirs("models/posture/spine_alignment", exist_ok=True)
    os.makedirs("models/posture/shoulder_balance", exist_ok=True)


def run_basic_analysis(solver: PostureAnalysisSolver, posture_data: Dict[str, Any]) -> None:
    """
    Run a basic posture analysis and print the results
    
    Args:
        solver: Initialized PostureAnalysisSolver
        posture_data: Posture data to analyze
    """
    query = "Is my back posture correct or am I slouching?"
    
    try:
        logger.info(f"Running analysis for query: '{query}'")
        result = solver.solve(query, posture_data)
        
        print("\n===== BASIC SOLVER USAGE =====")
        print(f"Query: {query}")
        print("\nSolution Method:")
        print(result["solution_method"])
        print("\nAnalysis Results:")
        print(json.dumps(result["analysis_result"], indent=2))
        print(f"\nOverall confidence: {result['confidence']:.2f}")
        
        logger.info(f"Analysis complete with confidence {result['confidence']:.2f}")
    except AnalysisError as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\nERROR: Analysis failed: {str(e)}")


def generate_single_trio(
    solver: PostureAnalysisSolver, 
    llm_client: MockLLMClient, 
    posture_data: Dict[str, Any]
) -> None:
    """
    Generate a single distillation trio and print/save it
    
    Args:
        solver: Initialized PostureAnalysisSolver
        llm_client: LLM client for text generation
        posture_data: Posture data to analyze
    """
    query = "Is my back posture correct or am I slouching?"
    
    try:
        logger.info(f"Generating distillation trio for query: '{query}'")
        trio = create_distillation_trio(solver, query, posture_data, llm_client)
        
        print("\n\n===== GENERATING A DISTILLATION TRIO =====")
        print("Trio generated:")
        print(json.dumps({
            "query": trio["query"],
            "solution_method_excerpt": trio["solution_method"].split("\n")[:5],
            "answer_excerpt": trio["answer"][:100] + "...",
            "metadata": trio["metadata"]
        }, indent=2))
        
        # Save trio to file
        output_file = "example_posture_trio.json"
        with open(output_file, "w") as f:
            json.dump(trio, f, indent=2)
            
        print(f"\nSaved complete trio to {output_file}")
        logger.info(f"Saved trio to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate trio: {str(e)}")
        print(f"\nERROR: Failed to generate trio: {str(e)}")


def generate_dataset(
    solver: PostureAnalysisSolver, 
    llm_client: MockLLMClient, 
    posture_data: Dict[str, Any]
) -> None:
    """
    Generate a small dataset of distillation trios
    
    Args:
        solver: Initialized PostureAnalysisSolver
        llm_client: LLM client for text generation
        posture_data: Posture data to analyze
    """
    # Example queries for dataset generation
    sample_queries = [
        "Is my back posture correct or am I slouching?",
        "Are my shoulders balanced?",
        "How is my overall posture?",
        "What posture issues do I have?",
        "Is my spine alignment healthy?"
    ]
    
    # In production, this would use multiple diverse posture samples
    posture_samples = [posture_data]
    
    output_file = "posture_distillation_data.jsonl"
    
    print("\n\n===== GENERATING MULTIPLE TRIOS =====")
    print(f"Creating trios for {len(sample_queries)} different queries...")
    
    try:
        generate_distillation_dataset(
            solver, 
            posture_samples, 
            sample_queries, 
            llm_client, 
            output_file
        )
        
        print(f"\nSaved distillation dataset to {output_file}")
        print("\nThis collection of trios would be used for knowledge distillation to train a domain-expert LLM.")
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        print(f"\nERROR: Failed to generate dataset: {str(e)}")


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="PostureAnalysisSolver example script")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/posture",
        help="Path to model registry"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["basic", "trio", "dataset", "all"], 
        default="all",
        help="Which example to run"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create model directories
    create_model_directories()
    
    # Initialize solver
    try:
        logger.info(f"Initializing solver with model path: {args.model_path}")
        solver = PostureAnalysisSolver(args.model_path)
    except ModelLoadError as e:
        logger.error(f"Failed to initialize solver: {str(e)}")
        print(f"ERROR: Failed to initialize solver: {str(e)}")
        return
    
    # Initialize LLM client
    llm_client = MockLLMClient()
    
    # Load example data
    posture_data = load_example_posture_data()
    
    # Run examples based on mode
    if args.mode in ["basic", "all"]:
        run_basic_analysis(solver, posture_data)
    
    if args.mode in ["trio", "all"]:
        generate_single_trio(solver, llm_client, posture_data)
    
    if args.mode in ["dataset", "all"]:
        generate_dataset(solver, llm_client, posture_data)
    
    logger.info("Example script completed successfully")


if __name__ == "__main__":
    main() 