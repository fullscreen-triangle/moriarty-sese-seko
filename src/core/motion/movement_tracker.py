import numpy as np
from typing import Dict, List
from collections import deque


class MovementTracker:
    def __init__(self, tracking_threshold: float = 0.15, window_size: int = 15):
        self.tracking_threshold = tracking_threshold
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
        self.velocity_history = deque(maxlen=window_size - 1)

    def track(self, pose_data: np.ndarray) -> Dict:
        """
        Track movement based on pose keypoints.

        Args:
            pose_data: Numpy array of shape (N, K, 3) containing keypoint data

        Returns:
            Dictionary containing movement analysis results
        """
        if pose_data is None or len(pose_data) == 0:
            return {'is_moving': False, 'velocity': 0.0, 'movement_magnitude': 0.0}

        # Calculate center of mass (COM) from keypoints
        com = self._calculate_com(pose_data)
        self.position_history.append(com)

        if len(self.position_history) < 2:
            return {'is_moving': False, 'velocity': 0.0, 'movement_magnitude': 0.0}

        # Calculate velocity
        velocity = self._calculate_velocity()
        self.velocity_history.append(velocity)

        # Calculate movement magnitude
        movement_magnitude = self._calculate_movement_magnitude()

        # Determine if movement is significant
        is_moving = movement_magnitude > self.tracking_threshold

        return {
            'is_moving': is_moving,
            'velocity': velocity.tolist(),
            'movement_magnitude': float(movement_magnitude),
            'com_position': com.tolist(),
            'movement_direction': self._calculate_movement_direction(velocity)
        }

    def _calculate_com(self, pose_data: np.ndarray) -> np.ndarray:
        """Calculate center of mass from keypoints"""
        # Use specific keypoints (e.g., hips and shoulders) to calculate COM
        relevant_keypoints = pose_data[0, [5, 6, 11, 12], :2]  # Example keypoint indices
        com = np.mean(relevant_keypoints, axis=0)
        return com

    def _calculate_velocity(self) -> np.ndarray:
        """Calculate velocity from position history"""
        current_pos = self.position_history[-1]
        previous_pos = self.position_history[-2]
        return current_pos - previous_pos

    def _calculate_movement_magnitude(self) -> float:
        """Calculate magnitude of movement from velocity history"""
        if len(self.velocity_history) < self.window_size - 1:
            return 0.0

        recent_velocities = np.array(list(self.velocity_history))
        magnitude = np.mean(np.linalg.norm(recent_velocities, axis=1))
        return magnitude

    def _calculate_movement_direction(self, velocity: np.ndarray) -> str:
        """Determine primary direction of movement"""
        if np.all(np.abs(velocity) < self.tracking_threshold):
            return "stationary"

        abs_velocity = np.abs(velocity)
        if abs_velocity[0] > abs_velocity[1]:
            return "horizontal" if velocity[0] > 0 else "horizontal_reverse"
        else:
            return "vertical" if velocity[1] > 0 else "vertical_reverse"

    def reset(self):
        """Reset tracker state"""
        self.position_history.clear()
        self.velocity_history.clear()
