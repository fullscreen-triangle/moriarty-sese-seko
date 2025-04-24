import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class ActivityMetrics:
    motion_intensity: float
    motion_area_ratio: float
    motion_centroid: Tuple[float, float]
    direction_vector: Tuple[float, float]
    is_active: bool


class PixelChangeDetector:
    def __init__(self,
                 min_intensity_threshold: float = 20.0,
                 min_area_threshold: float = 0.01,
                 history_length: int = 5):
        """
        Initialize pixel change detector for activity quantification.

        Args:
            min_intensity_threshold: Minimum pixel difference to consider as motion
            min_area_threshold: Minimum ratio of changed pixels to consider activity
            history_length: Number of frames to keep for motion history
        """
        self.min_intensity_threshold = min_intensity_threshold
        self.min_area_threshold = min_area_threshold
        self.history_length = history_length
        self.previous_frame = None
        self.motion_history = []
        self.logger = logging.getLogger(__name__)

    def detect_activity(self,
                        frame: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None) -> ActivityMetrics:
        """
        Detect and quantify activity in frame or region of interest.

        Args:
            frame: Current video frame
            roi: Region of interest as (x, y, width, height) or None for full frame

        Returns:
            ActivityMetrics containing motion analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply ROI if specified
            if roi:
                x, y, w, h = roi
                gray = gray[y:y + h, x:x + w]

            # Initialize if first frame
            if self.previous_frame is None:
                self.previous_frame = gray
                return ActivityMetrics(
                    motion_intensity=0.0,
                    motion_area_ratio=0.0,
                    motion_centroid=(0.0, 0.0),
                    direction_vector=(0.0, 0.0),
                    is_active=False
                )

            # Calculate absolute difference
            frame_diff = cv2.absdiff(gray, self.previous_frame)

            # Apply threshold
            _, motion_mask = cv2.threshold(
                frame_diff,
                self.min_intensity_threshold,
                255,
                cv2.THRESH_BINARY
            )

            # Calculate metrics
            motion_intensity = np.mean(frame_diff)
            motion_area_ratio = np.count_nonzero(motion_mask) / motion_mask.size

            # Calculate motion centroid
            if motion_area_ratio > self.min_area_threshold:
                moments = cv2.moments(motion_mask)
                if moments["m00"] != 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                else:
                    cx, cy = gray.shape[1] / 2, gray.shape[0] / 2
            else:
                cx, cy = gray.shape[1] / 2, gray.shape[0] / 2

            # Update motion history
            self.motion_history.append((cx, cy))
            if len(self.motion_history) > self.history_length:
                self.motion_history.pop(0)

            # Calculate direction vector
            if len(self.motion_history) >= 2:
                prev_x, prev_y = self.motion_history[-2]
                dx = cx - prev_x
                dy = cy - prev_y
                magnitude = np.sqrt(dx * dx + dy * dy)
                if magnitude > 0:
                    dx, dy = dx / magnitude, dy / magnitude
            else:
                dx, dy = 0.0, 0.0

            # Update previous frame
            self.previous_frame = gray

            return ActivityMetrics(
                motion_intensity=motion_intensity,
                motion_area_ratio=motion_area_ratio,
                motion_centroid=(cx, cy),
                direction_vector=(dx, dy),
                is_active=motion_area_ratio > self.min_area_threshold
            )

        except Exception as e:
            self.logger.error(f"Error in activity detection: {str(e)}")
            return None

    def get_motion_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Generate motion intensity heatmap"""
        if self.previous_frame is None:
            return np.zeros_like(frame)

        frame_diff = cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            self.previous_frame
        )

        heatmap = cv2.applyColorMap(frame_diff, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
