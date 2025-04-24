# athletic_metrics/core/video_quality.py

import numpy as np
import cv2
from typing import Dict, List, Optional
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

class VideoQualityAnalyzer:
    """Analyzes video quality metrics frame by frame"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.output_dir = Path(config['output']['plots_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_frame_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for a single frame"""
        if frame is None:
            self.logger.error("Received None frame for quality analysis")
            return {}
            
        # Convert frame to grayscale for some metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute metrics
        metrics = {}
        
        # Brightness (mean pixel value)
        metrics['brightness'] = np.mean(gray)
        
        # Contrast (standard deviation of pixel values)
        metrics['contrast'] = np.std(gray)
        
        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = np.var(laplacian)
        
        # Noise estimation using median filter
        median_filtered = cv2.medianBlur(gray, 5)
        noise = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))
        metrics['noise'] = noise
        
        # Saturation
        if len(frame.shape) == 3:  # Color image
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            metrics['saturation'] = np.mean(hsv[:, :, 1])
        else:
            metrics['saturation'] = 0
            
        return metrics

    def update_metrics_history(self, frame_idx: int, metrics: Dict[str, float]):
        """Update metrics history with new frame data"""
        # Add to history with frame index
        metrics['frame_idx'] = frame_idx
        self.metrics_history.append(metrics)

    def plot_metrics(self, sequence_name: str, output_dir: Optional[str] = None):
        """
        Plot quality metrics over time
        
        Args:
            sequence_name: Name of the sequence for plot title
            output_dir: Directory to save plots (optional)
        """
        if not self.metrics_history:
            self.logger.warning("No metrics data available for plotting")
            return
            
        # Configure matplotlib with a reliable style
        try:
            # Check available styles and use a safe one
            available_styles = plt.style.available
            safe_style = 'classic'  # Fallback to classic which is always available
            
            # Try to use a more modern style if available
            preferred_styles = ['ggplot', 'fivethirtyeight', 'bmh', 'tableau-colorblind10']
            for style in preferred_styles:
                if style in available_styles:
                    safe_style = style
                    break
                    
            self.logger.info(f"Using matplotlib style: {safe_style}")
            plt.style.use(safe_style)
        except Exception as e:
            self.logger.warning(f"Error setting matplotlib style: {e}")
            # Continue with default style if there's an issue
            
        # Prepare data for plotting
        frame_indices = [m['frame_idx'] for m in self.metrics_history]
        metrics_to_plot = {
            'Brightness': [m['brightness'] for m in self.metrics_history],
            'Contrast': [m['contrast'] for m in self.metrics_history],
            'Sharpness': [m['sharpness'] for m in self.metrics_history],
            'Noise': [m['noise'] for m in self.metrics_history],
            'Saturation': [m['saturation'] for m in self.metrics_history]
        }
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Video Quality Metrics: {sequence_name}', fontsize=16)
        
        # Plot each metric
        for (ax, (metric_name, values)) in zip(axes, metrics_to_plot.items()):
            ax.plot(frame_indices, values)
            ax.set_ylabel(metric_name)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set the x-axis label on the bottom subplot
        axes[-1].set_xlabel('Frame Number')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save plot if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{sequence_name}_quality_metrics.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path))
            self.logger.info(f"Quality metrics plot saved to {output_path}")
        
        plt.close()

