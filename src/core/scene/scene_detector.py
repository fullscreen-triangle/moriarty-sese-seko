import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class ChangeType(Enum):
    """Types of scene changes"""
    HARD_CUT = "hard_cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    CAMERA_MOTION = "camera_motion"
    FOCUS_CHANGE = "focus_change"


@dataclass
class SceneChange:
    """Stores information about a detected scene change"""
    frame_idx: int
    change_type: ChangeType
    confidence: float
    metrics: Dict[str, float]


class SceneDetector:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.hist_threshold = config.get('scene_detection', {}).get('hist_threshold', 0.5)
        self.flow_threshold = config.get('scene_detection', {}).get('flow_threshold', 0.7)
        self.edge_threshold = config.get('scene_detection', {}).get('edge_threshold', 0.6)

        self.frame_metrics = {
            'histogram_diff': [],
            'flow_magnitude': [],
            'edge_change': [],
            'focus_measure': []
        }

    def compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        metrics = {}

        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        metrics['histogram_diff'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        metrics['flow_magnitude'] = np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))

        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)
        metrics['edge_change'] = np.mean(np.abs(edges1.astype(float) - edges2.astype(float))) / 255.0

        metrics['focus_measure'] = self._compute_focus_measure(gray2)

        return metrics

    def _compute_focus_measure(self, gray_frame: np.ndarray) -> float:
        return cv2.Laplacian(gray_frame, cv2.CV_64F).var()

    def detect_scenes(self, frames_generator) -> List[SceneChange]:
        scene_changes = []
        prev_frame = None
        frame_idx = 0

        for frame in frames_generator:
            if prev_frame is not None:
                metrics = self.compute_frame_difference(prev_frame, frame)

                for metric_name, value in metrics.items():
                    self.frame_metrics[metric_name].append(value)

                change = self._analyze_metrics(metrics, frame_idx)
                if change:
                    scene_changes.append(change)

            prev_frame = frame.copy()
            frame_idx += 1

            if frame_idx % 100 == 0:
                self.logger.info(f"Processed {frame_idx} frames")

        return scene_changes

    def _analyze_metrics(self, metrics: Dict[str, float], frame_idx: int) -> Optional[SceneChange]:
        if metrics['histogram_diff'] < self.hist_threshold:
            return SceneChange(
                frame_idx=frame_idx,
                change_type=ChangeType.HARD_CUT,
                confidence=1.0 - metrics['histogram_diff'],
                metrics=metrics
            )

        if metrics['flow_magnitude'] > self.flow_threshold:
            return SceneChange(
                frame_idx=frame_idx,
                change_type=ChangeType.CAMERA_MOTION,
                confidence=metrics['flow_magnitude'],
                metrics=metrics
            )

        if len(self.frame_metrics['focus_measure']) > 1:
            focus_diff = abs(metrics['focus_measure'] - self.frame_metrics['focus_measure'][-1])
            if focus_diff > self.config.get('scene_detection', {}).get('focus_threshold', 1000):
                return SceneChange(
                    frame_idx=frame_idx,
                    change_type=ChangeType.FOCUS_CHANGE,
                    confidence=focus_diff / 2000,
                    metrics=metrics
                )

        return None

    def plot_metrics(self, sequence_name: str):
        plt.style.use('seaborn')
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))

        metrics_config = [
            ('histogram_diff', 'Histogram Correlation', self.hist_threshold),
            ('flow_magnitude', 'Optical Flow Magnitude', self.flow_threshold),
            ('edge_change', 'Edge Change Ratio', self.edge_threshold),
            ('focus_measure', 'Focus Measure', self.config.get('scene_detection', {}).get('focus_threshold', 1000))
        ]

        for idx, (metric_name, title, threshold) in enumerate(metrics_config):
            data = self.frame_metrics[metric_name]
            axes[idx].plot(data)
            axes[idx].set_title(title)
            axes[idx].axhline(y=threshold, color='r', linestyle='--')
            axes[idx].set_xlabel('Frame Number')
            axes[idx].set_ylabel(title)

            if len(data) > 0:
                peaks, _ = find_peaks(data, height=np.mean(data) + np.std(data))
                axes[idx].plot(peaks, [data[p] for p in peaks], 'x')

        plt.suptitle(f'Scene Detection Metrics - {sequence_name}')
        plt.tight_layout()
        plt.savefig(f"scene_detection_metrics_{sequence_name}.png")
        plt.close()

    def reset_metrics(self):
        self.frame_metrics = {
            'histogram_diff': [],
            'flow_magnitude': [],
            'edge_change': [],
            'focus_measure': []
        }
