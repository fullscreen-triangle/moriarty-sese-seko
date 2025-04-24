from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from scipy.signal import find_peaks
import cv2


@dataclass
class StrideMetrics:
    left_stride_length: float
    right_stride_length: float
    stride_frequency: float
    contact_time: float
    flight_time: float
    asymmetry_index: float
    phase: float


class StrideAnalyzer:
    def __init__(self, fps: int):
        self.fps = fps
        self.logger = logging.getLogger(__name__)
        self.frame_interval = 1.0 / fps
        self.previous_positions = {}

    def analyze_stride(self, skeleton_data: Dict, athlete_id: int) -> Dict:
        try:
            left_foot = np.array(skeleton_data['left_ankle'])
            right_foot = np.array(skeleton_data['right_ankle'])
            hip_center = np.array(skeleton_data['hip_center'])

            # Store positions for velocity calculation
            current_positions = {
                'left_foot': left_foot,
                'right_foot': right_foot,
                'hip_center': hip_center
            }

            # Calculate stride parameters
            left_stride = self._calculate_stride_length(
                self.previous_positions.get('left_foot'),
                current_positions['left_foot']
            )
            right_stride = self._calculate_stride_length(
                self.previous_positions.get('right_foot'),
                current_positions['right_foot']
            )

            stride_freq = self._calculate_stride_frequency(
                current_positions['hip_center'],
                self.previous_positions.get('hip_center', current_positions['hip_center'])
            )

            contact_time = self._estimate_ground_contact(left_foot, right_foot)
            flight_time = self.frame_interval - contact_time

            asymmetry = self._calculate_asymmetry(left_stride, right_stride)
            phase = self._calculate_phase(left_foot, right_foot)

            # Update previous positions
            self.previous_positions = current_positions

            return {
                'athlete_id': athlete_id,
                'metrics': StrideMetrics(
                    left_stride_length=left_stride,
                    right_stride_length=right_stride,
                    stride_frequency=stride_freq,
                    contact_time=contact_time,
                    flight_time=flight_time,
                    asymmetry_index=asymmetry,
                    phase=phase
                )
            }

        except Exception as e:
            self.logger.error(f"Error in stride analysis: {str(e)}")
            return None

    def _calculate_stride_length(self, prev_pos: Optional[np.ndarray],
                                 current_pos: np.ndarray) -> float:
        if prev_pos is None:
            return 0.0
        return np.linalg.norm(current_pos - prev_pos)

    def _calculate_stride_frequency(self, current_hip: np.ndarray,
                                    prev_hip: np.ndarray) -> float:
        if np.array_equal(current_hip, prev_hip):
            return 0.0
        vertical_displacement = abs(current_hip[1] - prev_hip[1])
        return 1.0 / (2 * self.frame_interval) if vertical_displacement > 0.05 else 0.0

    def _estimate_ground_contact(self, left_foot: np.ndarray,
                                 right_foot: np.ndarray) -> float:
        # Simple height-based estimation
        threshold = 0.1  # meters from ground
        left_contact = left_foot[1] < threshold
        right_contact = right_foot[1] < threshold
        return self.frame_interval if (left_contact or right_contact) else 0.0

    def _calculate_asymmetry(self, left_stride: float, right_stride: float) -> float:
        if left_stride == 0 or right_stride == 0:
            return 0.0
        return abs(left_stride - right_stride) / max(left_stride, right_stride)

    def _calculate_phase(self, left_foot: np.ndarray, right_foot: np.ndarray) -> float:
        # Calculate phase based on vertical positions
        left_height = left_foot[1]
        right_height = right_foot[1]
        return np.arctan2(right_height - left_height,
                          abs(right_foot[0] - left_foot[0])) / (2 * np.pi)
