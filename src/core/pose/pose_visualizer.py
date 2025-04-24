#!/usr/bin/env python
import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseVisualizer:
    """
    Visualizes pose models generated from queries.
    Supports both 2D and 3D visualization.
    """
    
    def __init__(self, distilled_model_path: str = "./distilled_model/final"):
        """
        Initialize the pose visualizer.
        
        Args:
            distilled_model_path (str): Path to the distilled model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distilled_model = torch.load(distilled_model_path, map_location=self.device)
        self.distilled_model.eval()
        
        # Define joint connections for visualization
        self.joint_connections = [
            # Torso
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head to hips
            (1, 5), (1, 6),  # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (4, 11), (4, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]
        
        # Define joint colors
        self.joint_colors = {
            "head": (255, 0, 0),
            "torso": (0, 255, 0),
            "arms": (0, 0, 255),
            "legs": (255, 255, 0)
        }
    
    def query_to_visualization(self, query: str, output_path: str = None) -> Dict[str, Any]:
        """
        Convert a natural language query into a visualization.
        
        Args:
            query (str): Natural language query about a pose
            output_path (str, optional): Path to save visualization
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        # Convert query to pose parameters
        with torch.no_grad():
            query_tensor = self._embed_query(query)
            pose_params = self.distilled_model(query_tensor)
        
        # Convert pose parameters to joint positions
        joint_positions = self._params_to_joints(pose_params)
        
        # Generate visualization data
        visualization_data = {
            "query": query,
            "joint_positions": joint_positions.tolist(),
            "joint_angles": self._calculate_joint_angles(joint_positions),
            "center_of_mass": self._calculate_center_of_mass(joint_positions)
        }
        
        if output_path:
            self._save_visualization(visualization_data, output_path)
        
        return visualization_data
    
    def _embed_query(self, query: str) -> torch.Tensor:
        """
        Convert a query string into a tensor representation.
        This should match the embedding method used in training.
        """
        # Implement proper query embedding
        # For now, using a simple one-hot encoding
        return torch.zeros(512)  # Placeholder
    
    def _params_to_joints(self, pose_params: torch.Tensor) -> np.ndarray:
        """
        Convert pose parameters to 3D joint positions.
        
        Args:
            pose_params (torch.Tensor): Pose parameters from the model
            
        Returns:
            np.ndarray: 3D joint positions
        """
        # Reshape parameters into joint positions
        # This is a simplified version - you'll need to implement the actual conversion
        num_joints = 17  # Standard number of joints
        joint_positions = pose_params.view(-1, num_joints, 3)
        return joint_positions.cpu().numpy()
    
    def _calculate_joint_angles(self, joint_positions: np.ndarray) -> Dict[str, float]:
        """
        Calculate joint angles from joint positions.
        
        Args:
            joint_positions (np.ndarray): 3D joint positions
            
        Returns:
            Dict[str, float]: Joint angles in degrees
        """
        angles = {}
        
        # Calculate angles for major joints
        # Example: knee angle
        left_knee = self._calculate_angle(
            joint_positions[0, 11],  # Left hip
            joint_positions[0, 13],  # Left knee
            joint_positions[0, 15]   # Left ankle
        )
        angles["left_knee"] = left_knee
        
        # Add more joint angles as needed
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            p1, p2, p3 (np.ndarray): 3D points
            
        Returns:
            float: Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_center_of_mass(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the center of mass of the pose.
        
        Args:
            joint_positions (np.ndarray): 3D joint positions
            
        Returns:
            np.ndarray: Center of mass coordinates
        """
        # Simple average of joint positions
        return np.mean(joint_positions, axis=1)
    
    def _save_visualization(self, data: Dict[str, Any], output_path: str):
        """
        Save visualization data and generate plots.
        
        Args:
            data (Dict[str, Any]): Visualization data
            output_path (str): Path to save visualization
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        # Save the data
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate and save 3D plot
        self._plot_3d_pose(data["joint_positions"], str(output_dir / "pose_3d.png"))
        
        # Generate and save 2D plots (front, side, top views)
        self._plot_2d_poses(data["joint_positions"], str(output_dir / "pose_2d.png"))
    
    def _plot_3d_pose(self, joint_positions: List[List[float]], output_path: str):
        """
        Generate a 3D plot of the pose.
        
        Args:
            joint_positions (List[List[float]]): 3D joint positions
            output_path (str): Path to save the plot
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot joints
        joint_positions = np.array(joint_positions)
        ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
                  c='red', marker='o')
        
        # Plot connections
        for connection in self.joint_connections:
            start = joint_positions[connection[0]]
            end = joint_positions[connection[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(output_path)
        plt.close()
    
    def _plot_2d_poses(self, joint_positions: List[List[float]], output_path: str):
        """
        Generate 2D plots of the pose from different views.
        
        Args:
            joint_positions (List[List[float]]): 3D joint positions
            output_path (str): Path to save the plots
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        joint_positions = np.array(joint_positions)
        
        # Front view (X-Y plane)
        self._plot_2d_view(ax1, joint_positions, 0, 1, "Front View")
        
        # Side view (Y-Z plane)
        self._plot_2d_view(ax2, joint_positions, 1, 2, "Side View")
        
        # Top view (X-Z plane)
        self._plot_2d_view(ax3, joint_positions, 0, 2, "Top View")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_2d_view(self, ax, joint_positions: np.ndarray, x_idx: int, y_idx: int, title: str):
        """
        Plot a 2D view of the pose.
        
        Args:
            ax: Matplotlib axis
            joint_positions (np.ndarray): 3D joint positions
            x_idx, y_idx (int): Indices of dimensions to plot
            title (str): Plot title
        """
        # Plot joints
        ax.scatter(joint_positions[:, x_idx], joint_positions[:, y_idx],
                  c='red', marker='o')
        
        # Plot connections
        for connection in self.joint_connections:
            start = joint_positions[connection[0]]
            end = joint_positions[connection[1]]
            ax.plot([start[x_idx], end[x_idx]], [start[y_idx], end[y_idx]], 'b-')
        
        ax.set_title(title)
        ax.set_xlabel(['X', 'Y', 'Z'][x_idx])
        ax.set_ylabel(['X', 'Y', 'Z'][y_idx])
        ax.axis('equal')

def main():
    """Command-line interface for pose visualization."""
    parser = argparse.ArgumentParser(description="Visualize poses from natural language queries")
    
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query about the pose"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization results"
    )
    
    args = parser.parse_args()
    
    try:
        visualizer = PoseVisualizer()
        output_path = os.path.join(args.output_dir, "pose_visualization.json")
        visualizer.query_to_visualization(args.query, output_path)
        print(f"Visualization saved to {output_path}")
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 