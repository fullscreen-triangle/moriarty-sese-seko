#!/usr/bin/env python
import os
import json
import argparse
from pathlib import Path
import re
from typing import Dict, List, Any

class VisualizationDataStructured:
    """
    Structure benchmark data for frontend visualization.
    Converts AI responses into a format suitable for rendering.
    """
    
    def __init__(self, benchmark_dir: str = "benchmark_data"):
        """
        Initialize the visualization data structurer.
        
        Args:
            benchmark_dir (str): Directory containing benchmark data
        """
        self.benchmark_dir = Path(benchmark_dir)
        
    def extract_frame_data(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract frame-by-frame data from AI responses.
        
        Args:
            response (str): AI response text
            
        Returns:
            List[Dict[str, Any]]: Structured frame data
        """
        frames = []
        # Look for frame-specific descriptions
        frame_pattern = r"Frame\s+(\d+):\s*(.*?)(?=Frame\s+\d+|$)"
        matches = re.finditer(frame_pattern, response, re.DOTALL)
        
        for match in matches:
            frame_num = int(match.group(1))
            description = match.group(2).strip()
            frames.append({
                "frame": frame_num,
                "description": description,
                "joints": self._extract_joint_data(description)
            })
        
        return frames
    
    def _extract_joint_data(self, description: str) -> Dict[str, float]:
        """
        Extract joint angle data from descriptions.
        
        Args:
            description (str): Frame description text
            
        Returns:
            Dict[str, float]: Joint angles
        """
        joints = {}
        # Look for joint angle mentions
        angle_pattern = r"(\w+)\s+joint\s+angle:\s*(\d+(?:\.\d+)?)\s*degrees"
        matches = re.finditer(angle_pattern, description)
        
        for match in matches:
            joint = match.group(1).lower()
            angle = float(match.group(2))
            joints[joint] = angle
        
        return joints
    
    def extract_movement_phases(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract movement phase data from AI responses.
        
        Args:
            response (str): AI response text
            
        Returns:
            List[Dict[str, Any]]: Movement phases
        """
        phases = []
        # Look for phase descriptions
        phase_pattern = r"Phase\s+(\d+):\s*(.*?)(?=Phase\s+\d+|$)"
        matches = re.finditer(phase_pattern, response, re.DOTALL)
        
        for match in matches:
            phase_num = int(match.group(1))
            description = match.group(2).strip()
            phases.append({
                "phase": phase_num,
                "description": description,
                "key_frames": self._extract_key_frames(description)
            })
        
        return phases
    
    def _extract_key_frames(self, description: str) -> List[int]:
        """
        Extract key frame numbers from phase descriptions.
        
        Args:
            description (str): Phase description text
            
        Returns:
            List[int]: Key frame numbers
        """
        frames = []
        # Look for frame number mentions
        frame_pattern = r"frame\s+(\d+)"
        matches = re.finditer(frame_pattern, description.lower())
        
        for match in matches:
            frame_num = int(match.group(1))
            frames.append(frame_num)
        
        return sorted(list(set(frames)))
    
    def extract_biomechanical_insights(self, response: str) -> Dict[str, Any]:
        """
        Extract biomechanical insights from AI responses.
        
        Args:
            response (str): AI response text
            
        Returns:
            Dict[str, Any]: Structured insights
        """
        insights = {
            "improvements": [],
            "efficiencies": [],
            "risks": [],
            "recommendations": []
        }
        
        # Look for different types of insights
        improvement_pattern = r"improvement[s]?\s+needed[s]?:\s*(.*?)(?=\n|$)"
        efficiency_pattern = r"efficient\s+aspect[s]?:\s*(.*?)(?=\n|$)"
        risk_pattern = r"risk[s]?:\s*(.*?)(?=\n|$)"
        recommendation_pattern = r"recommendation[s]?:\s*(.*?)(?=\n|$)"
        
        patterns = {
            "improvements": improvement_pattern,
            "efficiencies": efficiency_pattern,
            "risks": risk_pattern,
            "recommendations": recommendation_pattern
        }
        
        for key, pattern in patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            insights[key] = [match.group(1).strip() for match in matches]
        
        return insights
    
    def structure_benchmark_file(self, benchmark_file: Path) -> Dict[str, Any]:
        """
        Structure data from a single benchmark file.
        
        Args:
            benchmark_file (Path): Path to benchmark JSON file
            
        Returns:
            Dict[str, Any]: Structured visualization data
        """
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
        
        structured_data = {
            "model_file": benchmark_data["model_file"],
            "sport_type": benchmark_data["sport_type"],
            "timestamp": benchmark_data["timestamp"],
            "visualization_data": {}
        }
        
        for query in benchmark_data["queries"]:
            query_type = query["type"]
            
            if query_type == "pose_sequence":
                # Use OpenAI response for pose sequence (usually more detailed)
                structured_data["visualization_data"]["frames"] = self.extract_frame_data(
                    query["openai_response"]
                )
            
            elif query_type == "movement_pattern":
                # Use Anthropic response for movement phases (often more structured)
                structured_data["visualization_data"]["phases"] = self.extract_movement_phases(
                    query["anthropic_response"]
                )
            
            elif query_type == "biomechanical_insights":
                # Combine insights from both APIs
                openai_insights = self.extract_biomechanical_insights(query["openai_response"])
                anthropic_insights = self.extract_biomechanical_insights(query["anthropic_response"])
                
                # Merge insights, removing duplicates
                structured_data["visualization_data"]["insights"] = {
                    key: list(set(openai_insights[key] + anthropic_insights[key]))
                    for key in openai_insights
                }
        
        return structured_data
    
    def process_all_benchmarks(self, output_dir: str = "visualization_data") -> None:
        """
        Process all benchmark files and save structured data.
        
        Args:
            output_dir (str): Directory to save structured data
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all benchmark files
        benchmark_files = list(self.benchmark_dir.glob("*_benchmark.json"))
        print(f"Found {len(benchmark_files)} benchmark files")
        
        for benchmark_file in benchmark_files:
            print(f"\nProcessing {benchmark_file.name}...")
            
            try:
                structured_data = self.structure_benchmark_file(benchmark_file)
                
                # Save structured data
                output_file = output_path / f"{benchmark_file.stem.replace('_benchmark', '_visualization')}.json"
                with open(output_file, 'w') as f:
                    json.dump(structured_data, f, indent=2)
                
                print(f"Structured data saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing {benchmark_file.name}: {e}")
        
        # Create a summary file
        summary_file = output_path / "visualization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "total_files": len(benchmark_files),
                "processed_files": len(list(output_path.glob("*_visualization.json"))),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print(f"\nSummary saved to {summary_file}")

def main():
    """Command-line interface for structuring visualization data."""
    parser = argparse.ArgumentParser(description="Structure benchmark data for frontend visualization")
    
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="benchmark_data",
        help="Directory containing benchmark data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization_data",
        help="Directory to save structured visualization data"
    )
    
    args = parser.parse_args()
    
    try:
        structurer = VisualizationDataStructured(args.benchmark_dir)
        structurer.process_all_benchmarks(args.output_dir)
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    import time
    sys.exit(main()) 