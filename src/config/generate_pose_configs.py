#!/usr/bin/env python
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import time
from dotenv import load_dotenv
import anthropic
import openai

# Load environment variables
load_dotenv()

class PoseConfigGenerator:
    """
    Generates pose configurations suitable for React Three Fiber GLB models
    using GPT-4 and Claude as teachers.
    """
    
    def __init__(self, base_model_path: str = "models/pose_model.pth"):
        self.base_model_path = Path(base_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize API clients
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Load the base pose model
        self.base_pose_model = torch.load(base_model_path, map_location=self.device)
        
        # Define the structure of pose configurations
        self.pose_config_structure = {
            "joint_positions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "joint_name": "string",
                        "position": {"type": "array", "items": {"type": "number"}},
                        "rotation": {"type": "array", "items": {"type": "number"}},
                        "scale": {"type": "array", "items": {"type": "number"}}
                    }
                }
            },
            "joint_angles": {
                "type": "object",
                "properties": {
                    "spine": {"type": "number"},
                    "neck": {"type": "number"},
                    "left_shoulder": {"type": "number"},
                    "right_shoulder": {"type": "number"},
                    "left_elbow": {"type": "number"},
                    "right_elbow": {"type": "number"},
                    "left_hip": {"type": "number"},
                    "right_hip": {"type": "number"},
                    "left_knee": {"type": "number"},
                    "right_knee": {"type": "number"}
                }
            },
            "center_of_mass": {
                "type": "array",
                "items": {"type": "number"}
            }
        }
    
    def generate_config_examples(self, num_examples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate pose configuration examples using GPT-4 and Claude.
        Each example will be a query-configuration pair suitable for GLB models.
        """
        examples = []
        
        # Example query templates focused on pose configurations
        query_templates = [
            "Describe the pose configuration at frame {frame} in terms of joint positions, rotations, and angles.",
            "What are the joint angles and positions for the pose at frame {frame}?",
            "Break down the pose at frame {frame} into its constituent joint configurations.",
            "Describe the skeletal configuration at frame {frame}, including all joint positions and rotations.",
            "What are the exact joint positions and angles needed to recreate this pose at frame {frame}?"
        ]
        
        # Get total frames from base model
        total_frames = len(self.base_pose_model.get("frames", []))
        
        for _ in range(num_examples):
            frame = np.random.randint(0, total_frames)
            template = np.random.choice(query_templates)
            query = template.format(frame=frame)
            
            # Get teacher responses
            teacher_responses = self._get_teacher_responses(query)
            
            # Extract pose data
            pose_data = self._extract_pose_data(frame)
            
            # Create training example
            example = {
                "query": query,
                "frame": frame,
                "teacher_responses": teacher_responses,
                "pose_config": self._create_pose_config(pose_data)
            }
            
            examples.append(example)
            time.sleep(1)  # Rate limiting
        
        return examples
    
    def _get_teacher_responses(self, query: str) -> Dict[str, str]:
        """Get responses from both teacher models."""
        responses = {}
        
        # Get OpenAI response with specific instructions for GLB configuration
        openai_response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a 3D pose configuration expert. 
                Your task is to describe poses in terms of joint positions, rotations, and angles 
                that can be used to animate a GLB model in React Three Fiber. 
                Focus on providing exact numerical values and clear relationships between joints."""},
                {"role": "user", "content": query}
            ]
        )
        responses["openai"] = openai_response.choices[0].message.content
        
        # Get Anthropic response
        anthropic_response = self.anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": query}]
        )
        responses["anthropic"] = anthropic_response.content[0].text
        
        return responses
    
    def _extract_pose_data(self, frame: int) -> Dict[str, Any]:
        """Extract pose data for a specific frame from the base model."""
        frame_data = self.base_pose_model["frames"][frame]
        return frame_data
    
    def _create_pose_config(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a pose configuration suitable for GLB models.
        
        Args:
            pose_data (Dict[str, Any]): Raw pose data from the base model
            
        Returns:
            Dict[str, Any]: Structured pose configuration
        """
        # Convert raw pose data into the required configuration format
        config = {
            "joint_positions": [],
            "joint_angles": {},
            "center_of_mass": [0.0, 0.0, 0.0]  # Will be calculated
        }
        
        # Process joint positions and rotations
        for joint_name, joint_data in pose_data["joints"].items():
            config["joint_positions"].append({
                "joint_name": joint_name,
                "position": joint_data["position"].tolist(),
                "rotation": joint_data["rotation"].tolist(),
                "scale": [1.0, 1.0, 1.0]  # Default scale
            })
        
        # Calculate joint angles
        config["joint_angles"] = self._calculate_joint_angles(pose_data)
        
        # Calculate center of mass
        config["center_of_mass"] = self._calculate_center_of_mass(pose_data)
        
        return config
    
    def _calculate_joint_angles(self, pose_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate joint angles from pose data."""
        angles = {}
        
        # Calculate angles for major joints
        # Example: spine angle
        if "spine" in pose_data["joints"]:
            spine_joint = pose_data["joints"]["spine"]
            angles["spine"] = self._calculate_angle(
                spine_joint["position"],
                spine_joint["rotation"]
            )
        
        # Add more joint angle calculations as needed
        
        return angles
    
    def _calculate_center_of_mass(self, pose_data: Dict[str, Any]) -> List[float]:
        """Calculate the center of mass of the pose."""
        positions = [joint["position"] for joint in pose_data["joints"].values()]
        return np.mean(positions, axis=0).tolist()
    
    def _calculate_angle(self, position: np.ndarray, rotation: np.ndarray) -> float:
        """Calculate the angle from position and rotation data."""
        # Implement angle calculation based on your specific needs
        return 0.0  # Placeholder

def main():
    """Command-line interface for generating pose configurations."""
    parser = argparse.ArgumentParser(description="Generate pose configurations for GLB models")
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="models/pose_model.pth",
        help="Path to the base pose model"
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of configuration examples to generate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pose_configs",
        help="Directory to save configuration examples"
    )
    
    args = parser.parse_args()
    
    try:
        generator = PoseConfigGenerator(args.base_model)
        examples = generator.generate_config_examples(args.num_examples)
        
        # Save examples
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "pose_config_examples.json", 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"Generated {len(examples)} pose configuration examples")
        print(f"Saved to {output_dir / 'pose_config_examples.json'}")
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.exit(main()) 