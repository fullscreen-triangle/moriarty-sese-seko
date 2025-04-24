import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class JointAngles:
    knee_angle: float
    hip_angle: float
    ankle_angle: float
    shoulder_angle: float
    elbow_angle: float

class PoseAnalyzer:
    def __init__(self):
        """Initialize the pose analyzer with necessary parameters."""
        self.previous_positions = {}
        self.ground_level = None
    
    def analyze_pose(self, landmarks) -> Dict:
        """Analyze pose landmarks and return comprehensive analysis."""
        if not landmarks:
            return {}
            
        # Convert landmarks to numpy array for easier processing
        points = self._landmarks_to_numpy(landmarks)
        
        # Calculate various metrics
        joint_angles = self._calculate_joint_angles(points)
        ground_contact = self._detect_ground_contact(points)
        body_metrics = self._calculate_body_metrics(points)
        
        return {
            "joint_angles": joint_angles,
            "ground_contact": ground_contact,
            "body_metrics": body_metrics,
            "points": points
        }
    
    def _landmarks_to_numpy(self, landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array."""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    def _calculate_joint_angles(self, points) -> JointAngles:
        """Calculate angles between joints."""
        # Example joint calculations (simplified)
        knee_angle = self._angle_between_points(
            points[23],  # hip
            points[25],  # knee
            points[27]   # ankle
        )
        
        hip_angle = self._angle_between_points(
            points[11],  # shoulder
            points[23],  # hip
            points[25]   # knee
        )
        
        ankle_angle = self._angle_between_points(
            points[25],  # knee
            points[27],  # ankle
            points[31]   # foot
        )
        
        shoulder_angle = self._angle_between_points(
            points[13],  # elbow
            points[11],  # shoulder
            points[23]   # hip
        )
        
        elbow_angle = self._angle_between_points(
            points[11],  # shoulder
            points[13],  # elbow
            points[15]   # wrist
        )
        
        return JointAngles(
            knee_angle=knee_angle,
            hip_angle=hip_angle,
            ankle_angle=ankle_angle,
            shoulder_angle=shoulder_angle,
            elbow_angle=elbow_angle
        )
    
    def _detect_ground_contact(self, points) -> Dict[str, bool]:
        """Detect if feet are in contact with the ground."""
        # Use the lowest point as reference for ground level
        if self.ground_level is None:
            self.ground_level = np.max(points[:, 1])
        
        left_foot = points[31]
        right_foot = points[32]
        threshold = 0.05  # Adjustable threshold for ground contact
        
        return {
            "left_foot_contact": abs(left_foot[1] - self.ground_level) < threshold,
            "right_foot_contact": abs(right_foot[1] - self.ground_level) < threshold
        }
    
    def _calculate_body_metrics(self, points) -> Dict:
        """Calculate various body metrics."""
        # Calculate height (vertical distance from head to feet)
        height = abs(points[0][1] - np.mean([points[31][1], points[32][1]]))
        
        # Calculate stance width (distance between feet)
        stance_width = np.linalg.norm(points[31] - points[32])
        
        # Calculate center of mass (approximate)
        com = np.mean(points[[11, 12, 23, 24]], axis=0)
        
        return {
            "height": height,
            "stance_width": stance_width,
            "center_of_mass": com
        }
    
    def _angle_between_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle) 