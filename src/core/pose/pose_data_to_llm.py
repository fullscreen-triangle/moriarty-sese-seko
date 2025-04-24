#!/usr/bin/env python
import os
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.decomposition import PCA

class PoseDataExtractor:
    """
    Extracts pose data from model files (pose_model.pth) and converts it to a format
    suitable for LLM training. This class creates text descriptions and training examples
    from pose data.
    """
    def __init__(self, models_dir="models", output_dir="llm_training_data"):
        """Initialize the pose data extractor."""
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for training data
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        
        # Keypoint names for human-readable descriptions
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Motion verbs for generating descriptions
        self.motion_verbs = [
            "extending", "bending", "rotating", "raising", "lowering",
            "flexing", "straightening", "twisting", "turning", "shifting"
        ]
        
        # Sports-specific terms
        self.sports_terms = {
            "running": ["stride", "pace", "foot strike", "arm swing", "posture"],
            "jumping": ["takeoff", "flight", "landing", "extension", "height"],
            "throwing": ["wind-up", "release", "follow-through", "rotation", "angle"],
            "kicking": ["approach", "contact", "follow-through", "balance", "power"],
            "swimming": ["stroke", "kick", "rotation", "breathing", "glide"],
            "weightlifting": ["setup", "pull", "drive", "catch", "recovery"],
            "martial_arts": ["stance", "strike", "block", "kick", "balance"],
            "gymnastics": ["balance", "rotation", "extension", "landing", "hold"],
            "yoga": ["pose", "alignment", "balance", "stretch", "stability"],
            "dance": ["position", "turn", "leap", "extension", "rhythm"]
        }
    
    def extract_from_model(self, model_path):
        """
        Extract pose data from a model file.
        
        Args:
            model_path (str): Path to the pose model file
            
        Returns:
            dict: Extracted pose data
        """
        model_path = Path(model_path)
        print(f"Extracting data from {model_path}")
        
        try:
            # Load the PyTorch model
            model_data = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Extract metadata if available
            metadata = {}
            if isinstance(model_data, dict) and 'metadata' in model_data:
                metadata = model_data['metadata']
            
            # Extract pose data - structure depends on your specific model format
            pose_data = self._extract_pose_features(model_data)
            
            # Create a data structure with metadata and pose features
            extracted_data = {
                "model_name": model_path.stem,
                "extraction_date": datetime.now().isoformat(),
                "metadata": metadata,
                "pose_features": pose_data
            }
            
            # Save the raw extracted data
            output_file = self.output_dir / "raw" / f"{model_path.stem}_data.json"
            with open(output_file, 'w') as f:
                json.dump(extracted_data, f, indent=2)
            
            print(f"Raw data saved to {output_file}")
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting data from {model_path}: {e}")
            return None
    
    def _extract_pose_features(self, model_data):
        """
        Extract pose features from the model data.
        The exact implementation depends on your specific model structure.
        
        Args:
            model_data: The loaded model data
            
        Returns:
            list: Extracted pose features
        """
        # This is a placeholder implementation - you'll need to adapt this
        # to match your specific model structure
        pose_features = []
        
        # If the model is a state dict, try to extract weights from relevant layers
        if isinstance(model_data, dict):
            # Case 1: It's a state dict with direct access to pose data
            if 'pose_data' in model_data:
                return model_data['pose_data']
            
            # Case 2: It's a model state dict, try to extract pose-related parameters
            pose_params = {}
            for key, value in model_data.items():
                if 'pose' in key.lower() and isinstance(value, torch.Tensor):
                    pose_params[key] = value.detach().cpu().numpy().tolist()
            
            if pose_params:
                return pose_params
        
        # Case 3: It's a direct tensor or numpy array of pose data
        if isinstance(model_data, torch.Tensor):
            return model_data.detach().cpu().numpy().tolist()
        
        # Case 4: We couldn't find pose data in an expected format
        # As a fallback, flatten and return any tensor data we can find
        for k, v in model_data.items() if isinstance(model_data, dict) else []:
            if isinstance(v, torch.Tensor):
                try:
                    data = v.detach().cpu().numpy()
                    # Sample or reduce large tensors to a manageable size
                    if data.size > 10000:
                        # Use PCA to reduce dimensionality if it's a large tensor
                        flat_data = data.reshape(data.shape[0], -1) if data.ndim > 1 else data
                        if flat_data.shape[0] > 1 and flat_data.shape[1] > 100:
                            pca = PCA(n_components=100)
                            reduced_data = pca.fit_transform(flat_data)
                            pose_features.append({
                                "name": k,
                                "shape": str(data.shape),
                                "reduced_data": reduced_data.tolist()
                            })
                        else:
                            pose_features.append({
                                "name": k,
                                "shape": str(data.shape),
                                "sample_data": data.flatten()[:1000].tolist()
                            })
                except Exception as e:
                    print(f"Could not process tensor {k}: {e}")
        
        return pose_features
    
    def convert_to_text_descriptions(self, raw_data_path):
        """
        Convert raw pose data to textual descriptions for LLM training.
        
        Args:
            raw_data_path (str): Path to the raw data JSON file
            
        Returns:
            list: List of text descriptions
        """
        raw_data_path = Path(raw_data_path)
        print(f"Converting {raw_data_path} to text descriptions")
        
        try:
            # Load the raw data
            with open(raw_data_path, 'r') as f:
                raw_data = json.load(f)
            
            # Generate descriptions
            descriptions = []
            
            # Add metadata description
            metadata = raw_data.get('metadata', {})
            meta_desc = f"Video analysis of {metadata.get('video_name', 'unknown video')}.\n"
            if 'dimensions' in metadata:
                meta_desc += f"Video dimensions: {metadata.get('dimensions')}.\n"
            if 'fps' in metadata:
                meta_desc += f"Frame rate: {metadata.get('fps')} FPS.\n"
            if 'duration' in metadata:
                meta_desc += f"Duration: {metadata.get('duration')} seconds.\n"
            
            descriptions.append(meta_desc)
            
            # Generate descriptions from pose features
            pose_features = raw_data.get('pose_features', [])
            descriptions.extend(self._generate_pose_descriptions(pose_features))
            
            # Save the processed descriptions
            output_file = self.output_dir / "processed" / f"{raw_data_path.stem}_descriptions.json"
            with open(output_file, 'w') as f:
                json.dump(descriptions, f, indent=2)
            
            print(f"Generated {len(descriptions)} descriptions, saved to {output_file}")
            return descriptions
            
        except Exception as e:
            print(f"Error converting data to descriptions: {e}")
            return []
    
    def _generate_pose_descriptions(self, pose_features):
        """
        Generate human-readable descriptions from pose features.
        
        Args:
            pose_features: The extracted pose features
            
        Returns:
            list: Text descriptions
        """
        descriptions = []
        
        # This implementation will vary based on your extracted data structure
        # Here's a generic approach that tries to build meaningful descriptions
        
        if isinstance(pose_features, list):
            # Try to interpret as a sequence of poses
            for i, feature in enumerate(pose_features[:50]):  # Limit to avoid too many
                if isinstance(feature, dict) and 'name' in feature:
                    # It's a named feature with potentially reduced data
                    desc = f"Feature {feature['name']} with shape {feature['shape']}.\n"
                    if 'reduced_data' in feature and len(feature['reduced_data']) > 0:
                        # Describe some patterns in the reduced data
                        reduced = np.array(feature['reduced_data'])
                        desc += f"This feature shows {self._describe_pattern(reduced)}.\n"
                    descriptions.append(desc)
                else:
                    # Try to interpret as a pose keypoints array
                    try:
                        desc = self._generate_keypoint_description(feature, i)
                        if desc:
                            descriptions.append(desc)
                    except:
                        # Not recognizable as keypoints, create a generic description
                        descriptions.append(f"Pose feature {i} contains complex pattern data.")
        
        elif isinstance(pose_features, dict):
            # It's a dictionary of named features
            for name, data in pose_features.items():
                if 'pose' in name.lower() or 'keypoint' in name.lower():
                    try:
                        # Try to interpret as keypoint data
                        desc = f"Analysis of {name}:\n"
                        desc += self._describe_pose_parameter(name, np.array(data))
                        descriptions.append(desc)
                    except:
                        descriptions.append(f"Parameter {name} contains complex motion data.")
        
        # If we couldn't generate specific descriptions, make some generic ones
        if not descriptions:
            descriptions.append("The pose model contains data about human body position and movement.")
            descriptions.append("The model has analyzed motion patterns and body configurations in the video.")
            descriptions.append("Body posture and joint angles were extracted and analyzed from the video frames.")
        
        return descriptions
    
    def _generate_keypoint_description(self, keypoints, frame_idx):
        """Generate description of a single pose keypoint set."""
        # Skip if not in expected format
        if not isinstance(keypoints, (list, np.ndarray)) or len(keypoints) < 15:
            return None
        
        keypoints = np.array(keypoints)
        
        # Check if it might be pose keypoints (expecting shape to match common formats)
        if keypoints.shape[-1] not in [2, 3, 4]:  # x,y or x,y,z or x,y,z,confidence
            return None
            
        # Flatten to 2D if needed
        if len(keypoints.shape) > 2:
            keypoints = keypoints.reshape(-1, keypoints.shape[-1])
            
        # Generate description
        desc = f"Frame {frame_idx} analysis:\n"
        
        # Describe overall pose
        if len(keypoints) >= 17:  # Full pose keypoints
            # Check if standing or other pose
            if keypoints[15][1] > keypoints[11][1] and keypoints[16][1] > keypoints[12][1]:
                desc += "The person is standing with "
                
                # Check arm positions
                left_arm_angle = self._calculate_angle(
                    keypoints[5], keypoints[7], keypoints[9]
                )
                right_arm_angle = self._calculate_angle(
                    keypoints[6], keypoints[8], keypoints[10]
                )
                
                if left_arm_angle < 90 and right_arm_angle < 90:
                    desc += "arms raised upward. "
                elif left_arm_angle > 150 and right_arm_angle > 150:
                    desc += "arms extended downward. "
                else:
                    desc += "arms partially bent. "
                
                # Check leg positions
                left_leg_angle = self._calculate_angle(
                    keypoints[11], keypoints[13], keypoints[15]
                )
                right_leg_angle = self._calculate_angle(
                    keypoints[12], keypoints[14], keypoints[16]
                )
                
                if left_leg_angle < 160 or right_leg_angle < 160:
                    desc += "The legs are bent, possibly in a squatting position or during movement."
                else:
                    desc += "The legs are mostly straight."
            else:
                desc += "The person appears to be in a seated, crouched, or horizontal position."
        
        return desc
    
    def _calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        if len(a) < 2 or len(b) < 2 or len(c) < 2:
            return 0
            
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _describe_pattern(self, data):
        """Generate a description of patterns in the data."""
        if data.size == 0:
            return "no clear patterns"
            
        # Check for increasing or decreasing trends
        if data.ndim == 1 or (data.ndim == 2 and data.shape[0] == 1):
            flat_data = data.flatten()
            if np.all(np.diff(flat_data) > 0):
                return "steadily increasing values"
            elif np.all(np.diff(flat_data) < 0):
                return "steadily decreasing values"
            elif np.mean(np.diff(flat_data) > 0):
                return "generally increasing values with fluctuations"
            elif np.mean(np.diff(flat_data) < 0):
                return "generally decreasing values with fluctuations"
            else:
                return "fluctuating values without a clear trend"
        
        # For 2D data, try to describe more complex patterns
        elif data.ndim == 2:
            # Check for clusters
            from sklearn.cluster import KMeans
            if data.shape[0] > 3 and data.shape[1] > 1:
                try:
                    kmeans = KMeans(n_clusters=min(3, data.shape[0]), random_state=0)
                    kmeans.fit(data)
                    return f"{len(np.unique(kmeans.labels_))} distinct motion patterns"
                except:
                    pass
            
            return "complex multi-dimensional patterns"
        
        return "patterns that indicate motion analysis"
    
    def _describe_pose_parameter(self, name, data):
        """Generate a description for a named pose parameter."""
        if data.size == 0:
            return f"{name} has no data."
            
        desc = ""
        if 'hand' in name.lower() or 'finger' in name.lower():
            desc = "Hand positions and finger articulation data, "
        elif 'face' in name.lower():
            desc = "Facial expression and head orientation data, "
        elif 'pose' in name.lower():
            desc = "Full body pose configuration data, "
        elif 'point' in name.lower() or 'keypoint' in name.lower():
            desc = "Body keypoint location data, "
            
        # Add some details about the data values
        if data.size < 100:
            desc += f"with {data.size} numerical values."
        else:
            desc += f"containing a detailed set of {data.size} measurements."
            
        return desc
    
    def prepare_training_examples(self, description_path, sport_type=None):
        """
        Convert text descriptions to LLM training examples with appropriate prompts.
        
        Args:
            description_path (str): Path to the descriptions JSON file
            sport_type (str, optional): Type of sport for specialized terms
            
        Returns:
            list: Training examples in prompt-completion format
        """
        description_path = Path(description_path)
        print(f"Creating training examples from {description_path}")
        
        try:
            # Load the descriptions
            with open(description_path, 'r') as f:
                descriptions = json.load(f)
            
            # Create training examples
            training_examples = []
            
            # General analysis examples
            general_prompts = [
                "Analyze the pose data from this video:",
                "What does the pose analysis show about this video?",
                "Describe the body positioning in this video:",
                "What motion patterns are evident in this video?",
                "Summarize the pose analysis results:",
                "What can you tell me about the movement in this video?",
                "Describe the key motions captured in this video:",
                "What body positions are detected in this video?",
                "Analyze the body mechanics shown in this video:",
                "What does the pose extraction reveal about this video?"
            ]
            
            # Create general analysis examples
            all_descriptions = " ".join(descriptions)
            for prompt in general_prompts:
                training_examples.append({
                    "instruction": prompt,
                    "input": "",
                    "output": all_descriptions[:500]  # Limit length for reasonable outputs
                })
            
            # Sport-specific prompts if sport_type is specified
            if sport_type and sport_type.lower() in self.sports_terms:
                terms = self.sports_terms[sport_type.lower()]
                sport_prompts = [
                    f"Analyze the {sport_type} technique in this video:",
                    f"What does the pose data reveal about the {sport_type} performance?",
                    f"Describe the {sport_type} mechanics shown in this video:",
                    f"What {sport_type} skills are demonstrated in this video?",
                    f"Evaluate the {sport_type} form based on the pose data:"
                ]
                
                # Add sport-specific terms to the descriptions
                enhanced_desc = all_descriptions
                for term in terms:
                    enhanced_desc += f" The {term} shows proper form and technique."
                
                for prompt in sport_prompts:
                    training_examples.append({
                        "instruction": prompt,
                        "input": "",
                        "output": enhanced_desc[:500]
                    })
            
            # Save the training examples
            output_file = self.output_dir / "processed" / f"{description_path.stem}_examples.json"
            with open(output_file, 'w') as f:
                json.dump(training_examples, f, indent=2)
            
            print(f"Created {len(training_examples)} training examples, saved to {output_file}")
            return training_examples
            
        except Exception as e:
            print(f"Error creating training examples: {e}")
            return []
    
    def process_all_models(self, sport_type=None):
        """
        Process all pose models in the models directory.
        
        Args:
            sport_type (str, optional): Type of sport for specialized terms
            
        Returns:
            list: All training examples
        """
        # Find all pose model files
        model_files = list(self.models_dir.glob("*_model.pth"))
        print(f"Found {len(model_files)} model files")
        
        all_examples = []
        
        for model_path in model_files:
            # Extract raw data
            raw_data = self.extract_from_model(model_path)
            if raw_data:
                # Generate descriptions
                raw_data_path = self.output_dir / "raw" / f"{model_path.stem}_data.json"
                descriptions = self.convert_to_text_descriptions(raw_data_path)
                
                # Create training examples
                desc_path = self.output_dir / "processed" / f"{model_path.stem}_data_descriptions.json"
                examples = self.prepare_training_examples(desc_path, sport_type)
                all_examples.extend(examples)
        
        # Combine all examples into a single dataset
        combined_file = self.output_dir / "processed" / "combined_training_data.json"
        with open(combined_file, 'w') as f:
            json.dump(all_examples, f, indent=2)
        
        print(f"Combined {len(all_examples)} examples into {combined_file}")
        return all_examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract pose data from models and convert to LLM training format")
    parser.add_argument("--models_dir", default="models", help="Directory containing pose model files")
    parser.add_argument("--output_dir", default="llm_training_data", help="Output directory for LLM training data")
    parser.add_argument("--sport_type", help="Type of sport for specialized prompts")
    
    args = parser.parse_args()
    
    extractor = PoseDataExtractor(args.models_dir, args.output_dir)
    extractor.process_all_models(args.sport_type) 