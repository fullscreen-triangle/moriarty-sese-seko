import torch
import numpy as np
import logging
from pathlib import Path

class PoseDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        
        # Create a simple mock model for testing
        self.model = self._create_mock_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _create_mock_model(self):
        """Create a simple mock model that returns dummy pose data"""
        class MockModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # Return dummy keypoints: batch_size x 17 keypoints x 3 (x, y, confidence)
                return torch.rand(batch_size, 17, 3)
                
        return MockModel()

    def detect(self, frame: np.ndarray):
        """
        Detect poses in the frame
        Returns: Array of shape (N, K, 3) where:
            N = number of people (always 1 in this mock)
            K = 17 keypoints
            3 = x, y, confidence
        """
        if frame is None:
            return None
            
        # Create dummy keypoint detections
        keypoints = np.random.rand(1, 17, 3)  # 1 person, 17 keypoints, (x,y,conf)
        keypoints[..., 2] = 0.8  # Set confidence to 0.8
        
        # Scale x,y coordinates to frame size
        height, width = frame.shape[:2]
        keypoints[..., 0] *= width
        keypoints[..., 1] *= height
        
        return keypoints

    def get_model_info(self):
        return {
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'model_type': 'mock_model'
        }
