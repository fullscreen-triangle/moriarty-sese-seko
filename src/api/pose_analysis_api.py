#!/usr/bin/env python
import os
import json
import argparse
import requests
from pathlib import Path
import time
from dotenv import load_dotenv
import anthropic
import openai
import sys
from src.core.pose.pose_data_to_llm import PoseDataExtractor

# Load environment variables
load_dotenv()

class PoseAnalysisAPI:
    """
    Use OpenAI and Claude APIs to analyze pose data extracted from sports videos.
    This approach leverages powerful pre-trained LLMs instead of training custom models.
    """
    
    def __init__(self, api_provider="openai"):
        """
        Initialize the pose analysis API client.
        
        Args:
            api_provider (str): The API provider to use ('openai' or 'anthropic')
        """
        self.api_provider = api_provider.lower()
        
        # Check and set up API keys
        if self.api_provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            openai.api_key = self.api_key
            print("Using OpenAI API")
            
        elif self.api_provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("Using Anthropic Claude API")
            
        else:
            raise ValueError("Invalid API provider. Use 'openai' or 'anthropic'")
    
    def analyze_pose_data(self, description, prompt=None, max_tokens=1000):
        """
        Send pose data description to the API for analysis.
        
        Args:
            description (str): The pose data description text
            prompt (str, optional): Custom prompt to use
            max_tokens (int): Maximum tokens in the response
            
        Returns:
            str: The API's analysis response
        """
        if not prompt:
            prompt = "Analyze this sports video pose data and provide insights about the movement patterns, technique, and body mechanics:"
        
        full_prompt = f"{prompt}\n\n{description}"
        
        try:
            if self.api_provider == "openai":
                return self._query_openai(full_prompt, max_tokens)
            elif self.api_provider == "anthropic":
                return self._query_claude(full_prompt, max_tokens)
        except Exception as e:
            print(f"Error querying {self.api_provider} API: {e}")
            return None
    
    def _query_openai(self, prompt, max_tokens=1000):
        """Query the OpenAI API."""
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or gpt-3.5-turbo for lower cost
            messages=[
                {"role": "system", "content": "You are a sports biomechanics expert analyzing pose data extracted from sports videos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def _query_claude(self, prompt, max_tokens=1000):
        """Query the Anthropic Claude API."""
        response = self.client.messages.create(
            model="claude-3-opus-20240229",  # or claude-3-sonnet-20240229 for balance of quality and cost
            max_tokens=max_tokens,
            system="You are a sports biomechanics expert analyzing pose data extracted from sports videos.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def batch_analyze_from_file(self, data_file, output_file=None):
        """
        Analyze all pose descriptions in a data file.
        
        Args:
            data_file (str): Path to JSON file with pose descriptions
            output_file (str, optional): Path to save results
            
        Returns:
            list: All analysis results
        """
        try:
            with open(data_file, 'r') as f:
                descriptions = json.load(f)
            
            results = []
            for i, desc in enumerate(descriptions):
                print(f"Analyzing description {i+1}/{len(descriptions)}...")
                analysis = self.analyze_pose_data(desc)
                results.append({
                    "description": desc,
                    "analysis": analysis
                })
                # Sleep to avoid hitting API rate limits
                if i < len(descriptions) - 1:
                    time.sleep(1)
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")
            
            return results
            
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return None
    
    def analyze_from_model(self, model_path, sport_type=None, output_dir="pose_analysis_results"):
        """
        Extract data from a pose model file and analyze it directly.
        
        Args:
            model_path (str): Path to the pose model file
            sport_type (str, optional): Type of sport for context
            output_dir (str): Directory to save analysis results
            
        Returns:
            dict: Analysis results
        """
        try:
            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract pose data
            extractor = PoseDataExtractor()
            raw_data = extractor.extract_from_model(model_path)
            
            if not raw_data:
                print(f"Failed to extract data from {model_path}")
                return None
            
            # Convert to text descriptions
            raw_data_path = Path(output_dir) / f"{Path(model_path).stem}_data.json"
            with open(raw_data_path, 'w') as f:
                json.dump(raw_data, f, indent=2)
                
            descriptions = extractor.convert_to_text_descriptions(raw_data_path)
            
            if not descriptions:
                print(f"Failed to convert data to descriptions")
                return None
            
            # Create a combined description
            combined_description = "\n".join(descriptions)
            
            # Build a sport-specific prompt if sport type is provided
            prompt = "Analyze this sports video pose data and provide insights about the movement patterns, technique, and body mechanics:"
            if sport_type:
                prompt = f"Analyze this {sport_type} video pose data and provide detailed insights about the {sport_type} technique, movement patterns, and body mechanics:"
            
            # Analyze the data
            analysis = self.analyze_pose_data(combined_description, prompt)
            
            # Save the results
            result = {
                "model_file": str(model_path),
                "sport_type": sport_type,
                "descriptions": descriptions,
                "analysis": analysis
            }
            
            output_file = Path(output_dir) / f"{Path(model_path).stem}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Analysis saved to {output_file}")
            return result
            
        except Exception as e:
            print(f"Error analyzing model: {e}")
            return None
    
    def process_all_models(self, model_dir="models", sport_type=None, output_dir="pose_analysis_results"):
        """
        Process and analyze all model files in a directory.
        
        Args:
            model_dir (str): Directory containing pose model files
            sport_type (str, optional): Type of sport for context
            output_dir (str): Directory to save analysis results
            
        Returns:
            list: All analysis results
        """
        # Find all pose model files
        model_files = list(Path(model_dir).glob("*_model.pth"))
        print(f"Found {len(model_files)} model files")
        
        results = []
        for model_path in model_files:
            print(f"\nProcessing {model_path.name}...")
            result = self.analyze_from_model(model_path, sport_type, output_dir)
            if result:
                results.append(result)
        
        # Create a summary report
        summary_file = Path(output_dir) / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "total_models": len(model_files),
                "successful_analyses": len(results),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sport_type": sport_type,
                "api_provider": self.api_provider
            }, f, indent=2)
        
        print(f"\nProcessed {len(results)}/{len(model_files)} models successfully")
        print(f"Summary saved to {summary_file}")
        return results

def main():
    """Command-line interface for the pose analysis API."""
    parser = argparse.ArgumentParser(description="Analyze pose data using AI APIs")
    
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="API provider to use (openai or anthropic)"
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
        default="pose_analysis_results",
        help="Directory to save analysis results"
    )
    
    parser.add_argument(
        "--sport_type",
        type=str,
        help="Type of sport for specialized analysis"
    )
    
    parser.add_argument(
        "--single_model",
        type=str,
        help="Process only a single model file (provide the filename)"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = PoseAnalysisAPI(api_provider=args.api)
        
        if args.single_model:
            model_path = os.path.join(args.model_dir, args.single_model)
            if not os.path.exists(model_path):
                print(f"Error: Model file '{model_path}' does not exist")
                return 1
                
            print(f"Analyzing single model: {model_path}")
            analyzer.analyze_from_model(model_path, args.sport_type, args.output_dir)
        else:
            print(f"Processing all models in: {args.model_dir}")
            analyzer.process_all_models(args.model_dir, args.sport_type, args.output_dir)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 