import torch
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


class ActionClassifier:
    def __init__(self, model_path: str, class_mapping: dict):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = class_mapping
        
        try:
            model_path = Path(model_path)
            if model_path.exists():
                self.model = torch.load(model_path, map_location=self.device)
                self.logger.info(f"Loaded model from {model_path}")
            else:
                self.logger.warning(f"Model file not found at {model_path}. Using mock model.")
                self.model = self._create_mock_model()
        except Exception as e:
            self.logger.warning(f"Error loading model: {e}. Using mock model.")
            self.model = self._create_mock_model()
        
        self.model.eval()

    def _create_mock_model(self):
        """Create a simple mock model that returns dummy action predictions"""
        class MockModel(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes
            
            def forward(self, x):
                batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
                # Return dummy predictions
                return torch.softmax(torch.randn(batch_size, self.num_classes), dim=1)
                
        return MockModel(len(self.class_mapping))

    def predict(self, pose_sequence):
        """
        Predict action from a sequence of poses
        Returns: Dict with action class and confidence
        """
        if not isinstance(pose_sequence, torch.Tensor):
            pose_sequence = torch.tensor(pose_sequence, device=self.device)
            
        with torch.no_grad():
            outputs = self.model(pose_sequence)
            
        # Get predicted class and confidence
        confidence, predicted = torch.max(outputs, 1)
        action_class = self.class_mapping[predicted.item()]
        
        return {
            'action': action_class,
            'confidence': confidence.item(),
            'all_confidences': {
                class_name: conf.item()
                for class_name, conf in zip(self.class_mapping.values(), outputs[0])
            }
        }

    def get_model_info(self):
        return {
            'device': str(self.device),
            'num_classes': len(self.class_mapping),
            'class_mapping': self.class_mapping,
            'model_type': 'mock_model' if isinstance(self.model, self._create_mock_model().__class__) else 'loaded_model'
        }


# motion_metrics.py
class MotionMetricsCalculator:
    def __init__(self, fps: int = 30):
        self.fps = fps

    def calculate_metrics(self, keypoints: np.ndarray) -> Dict:
        velocity = self._calculate_velocity(keypoints)
        acceleration = self._calculate_acceleration(velocity)
        jerk = self._calculate_jerk(acceleration)

        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'smoothness': self._calculate_smoothness(jerk),
            'range_of_motion': self._calculate_rom(keypoints),
            'stability': self._calculate_stability(keypoints)
        }

    def _calculate_velocity(self, keypoints: np.ndarray) -> np.ndarray:
        return np.diff(keypoints, axis=0) * self.fps

    def _calculate_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        return np.diff(velocity, axis=0) * self.fps

    def _calculate_jerk(self, acceleration: np.ndarray) -> np.ndarray:
        return np.diff(acceleration, axis=0) * self.fps

    def _calculate_smoothness(self, jerk: np.ndarray) -> float:
        return -np.sum(jerk ** 2)

    def _calculate_rom(self, keypoints: np.ndarray) -> Dict:
        ranges = np.ptp(keypoints, axis=0)
        return {'x': ranges[0], 'y': ranges[1]}

    def _calculate_stability(self, keypoints: np.ndarray) -> float:
        com = np.mean(keypoints, axis=1)
        return np.std(com)


# phase_analyzer.py
class PhaseAnalyzer:
    def __init__(self, window_size: int = 30, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap

    def analyze_phases(self, motion_data: np.ndarray) -> List[Dict]:
        phases = []
        step_size = int(self.window_size * (1 - self.overlap))

        for i in range(0, len(motion_data) - self.window_size, step_size):
            window = motion_data[i:i + self.window_size]
            phase_type = self._classify_phase(window)
            phases.append({
                'start_frame': i,
                'end_frame': i + self.window_size,
                'phase_type': phase_type
            })
        return phases

    def _classify_phase(self, window: np.ndarray) -> str:
        # Phase classification logic
        return "movement_phase"


# pattern_matcher.py
class PatternMatcher:
    def __init__(self, template_path: str, similarity_threshold: float = 0.8):
        self.templates = self._load_templates(template_path)
        self.similarity_threshold = similarity_threshold

    def match_pattern(self, motion_sequence: np.ndarray) -> Dict:
        best_match = None
        best_score = -1

        for template_name, template in self.templates.items():
            score = self._calculate_similarity(motion_sequence, template)
            if score > best_score and score > self.similarity_threshold:
                best_score = score
                best_match = template_name

        return {
            'pattern': best_match,
            'confidence': best_score
        }

    def _calculate_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        # DTW or other similarity measure
        return 0.0

    def _load_templates(self, path: str) -> Dict:
        # Load template patterns
        return {}


# sequence_analyzer.py
class SequenceAnalyzer:
    def __init__(self, min_sequence_length: int = 10):
        self.min_sequence_length = min_sequence_length
        self.logger = logging.getLogger(__name__)

    def analyze(self, sequence):
        """Analyze a sequence of movements"""
        if len(sequence) < self.min_sequence_length:
            self.logger.warning(f"Sequence too short: {len(sequence)} < {self.min_sequence_length}")
            return None
            
        # Mock analysis for now
        return {
            'length': len(sequence),
            'is_valid': len(sequence) >= self.min_sequence_length,
            'segments': self._create_mock_segments(sequence)
        }

    def _create_mock_segments(self, sequence):
        return [{
            'start': i,
            'end': i + self.min_sequence_length,
            'type': 'movement_segment',
            'confidence': np.random.uniform(0.7, 1.0)
        } for i in range(0, len(sequence), self.min_sequence_length)]


# symmetry_analyzer.py
class SymmetryAnalyzer:
    def __init__(self, symmetry_points: list = None):
        self.symmetry_points = symmetry_points or ["shoulders", "hips", "knees", "ankles"]
        self.logger = logging.getLogger(__name__)

    def analyze(self, pose_data):
        """Analyze pose symmetry"""
        try:
            # Mock symmetry analysis
            symmetry_scores = {
                point: np.random.uniform(0.7, 1.0)
                for point in self.symmetry_points
            }
            
            return {
                'overall_symmetry': np.mean(list(symmetry_scores.values())),
                'point_symmetry': symmetry_scores,
                'is_balanced': True,
                'recommendations': self._generate_mock_recommendations()
            }
        except Exception as e:
            self.logger.error(f"Error in symmetry analysis: {e}")
            return None

    def _generate_mock_recommendations(self):
        return [
            "Maintain shoulder alignment",
            "Keep hips level during movement",
            "Balance knee flexion"
        ]


# tempo_analyzer.py
class TempoAnalyzer:
    def __init__(self, fps: int = 30):
        self.fps = fps

    def analyze_tempo(self, motion_data: np.ndarray) -> Dict:
        frequency = self._calculate_frequency(motion_data)
        rhythm = self._detect_rhythm(motion_data)

        return {
            'tempo': frequency * 60,  # Convert to BPM
            'rhythm_pattern': rhythm,
            'regularity': self._calculate_regularity(motion_data)
        }

    def _calculate_frequency(self, data: np.ndarray) -> float:
        return 0.0  # Implement frequency calculation

    def _detect_rhythm(self, data: np.ndarray) -> List:
        return []  # Implement rhythm detection

    def _calculate_regularity(self, data: np.ndarray) -> float:
        return 0.0  # Implement regularity calculation


# trajectory_analyzer.py
class TrajectoryAnalyzer:
    def __init__(self, trajectory_smoothing: int = 5):
        self.trajectory_smoothing = trajectory_smoothing
        self.logger = logging.getLogger(__name__)

    def analyze(self, keypoints_sequence):
        """Analyze movement trajectory"""
        try:
            # Mock trajectory analysis
            return {
                'smoothness': np.random.uniform(0.7, 1.0),
                'efficiency': np.random.uniform(0.7, 1.0),
                'path_length': np.random.uniform(100, 200),
                'velocity': self._mock_velocity_profile(),
                'acceleration': self._mock_acceleration_profile(),
                'key_points': self._mock_key_points()
            }
        except Exception as e:
            self.logger.error(f"Error in trajectory analysis: {e}")
            return None

    def _mock_velocity_profile(self):
        return {
            'peak': np.random.uniform(5, 10),
            'average': np.random.uniform(3, 7),
            'variation': np.random.uniform(0.1, 0.3)
        }

    def _mock_acceleration_profile(self):
        return {
            'peak': np.random.uniform(2, 5),
            'average': np.random.uniform(1, 3),
            'phases': ['acceleration', 'steady', 'deceleration']
        }

    def _mock_key_points(self):
        return {
            'start': {'x': 0, 'y': 0},
            'peak': {'x': 50, 'y': 50},
            'end': {'x': 100, 'y': 100}
        }


class MotionClassifier:
    """
    Classifies motion patterns from pose sequences.
    
    This class integrates multiple analysis components to provide comprehensive
    motion classification for biomechanical analysis.
    """
    
    def __init__(self, fps: int = 30):
        """
        Initialize the motion classifier.
        
        Args:
            fps: Frames per second of the video
        """
        self.fps = fps
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis components
        self.metrics_calculator = MotionMetricsCalculator(fps)
        self.phase_analyzer = PhaseAnalyzer()
        self.pattern_matcher = PatternMatcher("", 0.7)  # Empty template path for now
        self.sequence_analyzer = SequenceAnalyzer()
        self.symmetry_analyzer = SymmetryAnalyzer()
        self.tempo_analyzer = TempoAnalyzer(fps)
        self.trajectory_analyzer = TrajectoryAnalyzer()
        
        # Classification thresholds
        self.velocity_thresholds = {
            'stationary': 0.1,
            'slow': 1.0,
            'moderate': 3.0,
            'fast': 6.0
        }
        
        self.motion_patterns = {
            'running': {'tempo_range': (120, 200), 'symmetry_min': 0.7},
            'walking': {'tempo_range': (80, 140), 'symmetry_min': 0.6},
            'jumping': {'tempo_range': (60, 120), 'velocity_peak': 5.0},
            'stretching': {'tempo_range': (10, 60), 'smoothness_min': 0.8},
            'dancing': {'tempo_range': (100, 180), 'rhythm_consistency': 0.7}
        }
    
    def classify(self, pose_sequence: np.ndarray) -> Dict:
        """
        Classify motion from a pose sequence.
        
        Args:
            pose_sequence: Array of pose keypoints [frames, keypoints, coords]
            
        Returns:
            Dictionary containing motion classification results
        """
        if pose_sequence is None or len(pose_sequence) < 2:
            return {
                'motion_class': 'insufficient_data',
                'confidence': 0.0,
                'details': {}
            }
        
        try:
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(pose_sequence)
            
            # Perform classification
            motion_class, confidence = self._classify_motion(metrics)
            
            return {
                'motion_class': motion_class,
                'confidence': confidence,
                'details': {
                    'metrics': metrics,
                    'pattern_analysis': self._analyze_patterns(pose_sequence),
                    'temporal_analysis': self._analyze_temporal_features(pose_sequence)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in motion classification: {e}")
            return {
                'motion_class': 'error',
                'confidence': 0.0,
                'details': {'error': str(e)}
            }
    
    def _calculate_comprehensive_metrics(self, pose_sequence: np.ndarray) -> Dict:
        """Calculate comprehensive motion metrics."""
        metrics = {}
        
        try:
            # Basic motion metrics
            motion_metrics = self.metrics_calculator.calculate_metrics(pose_sequence)
            metrics['motion'] = motion_metrics
            
            # Temporal analysis
            tempo_data = self.tempo_analyzer.analyze_tempo(pose_sequence)
            metrics['tempo'] = tempo_data
            
            # Symmetry analysis
            symmetry_data = self.symmetry_analyzer.analyze(pose_sequence)
            metrics['symmetry'] = symmetry_data
            
            # Trajectory analysis
            trajectory_data = self.trajectory_analyzer.analyze(pose_sequence)
            metrics['trajectory'] = trajectory_data
            
            # Sequence analysis
            sequence_data = self.sequence_analyzer.analyze(pose_sequence)
            metrics['sequence'] = sequence_data
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _classify_motion(self, metrics: Dict) -> Tuple[str, float]:
        """Classify motion based on calculated metrics."""
        if 'error' in metrics:
            return 'error', 0.0
        
        # Extract key metrics for classification
        motion_data = metrics.get('motion', {})
        tempo_data = metrics.get('tempo', {})
        symmetry_data = metrics.get('symmetry', {})
        trajectory_data = metrics.get('trajectory', {})
        
        # Calculate overall velocity
        velocity = motion_data.get('velocity', np.array([[0, 0]]))
        if isinstance(velocity, list):
            velocity = np.array(velocity)
        avg_velocity = np.mean(np.linalg.norm(velocity, axis=1)) if velocity.ndim > 1 else 0.0
        
        # Classify based on velocity
        if avg_velocity < self.velocity_thresholds['stationary']:
            motion_class = 'stationary'
            confidence = 0.9
        elif avg_velocity < self.velocity_thresholds['slow']:
            motion_class = self._classify_slow_motion(metrics)
            confidence = 0.7
        elif avg_velocity < self.velocity_thresholds['moderate']:
            motion_class = self._classify_moderate_motion(metrics)
            confidence = 0.8
        else:
            motion_class = self._classify_fast_motion(metrics)
            confidence = 0.8
        
        # Adjust confidence based on data quality
        if symmetry_data and symmetry_data.get('overall_symmetry', 0) > 0.8:
            confidence = min(1.0, confidence + 0.1)
        
        return motion_class, confidence
    
    def _classify_slow_motion(self, metrics: Dict) -> str:
        """Classify slow motion patterns."""
        tempo_data = metrics.get('tempo', {})
        smoothness = metrics.get('trajectory', {}).get('smoothness', 0.5)
        
        tempo = tempo_data.get('tempo', 0)
        
        if smoothness > 0.8 and tempo < 60:
            return 'stretching'
        elif tempo > 60:
            return 'walking'
        else:
            return 'slow_movement'
    
    def _classify_moderate_motion(self, metrics: Dict) -> str:
        """Classify moderate motion patterns."""
        tempo_data = metrics.get('tempo', {})
        symmetry_data = metrics.get('symmetry', {})
        
        tempo = tempo_data.get('tempo', 0)
        symmetry = symmetry_data.get('overall_symmetry', 0.5) if symmetry_data else 0.5
        
        if 120 <= tempo <= 180 and symmetry > 0.7:
            return 'running'
        elif 80 <= tempo <= 140 and symmetry > 0.6:
            return 'walking'
        elif 100 <= tempo <= 180:
            return 'dancing'
        else:
            return 'moderate_movement'
    
    def _classify_fast_motion(self, metrics: Dict) -> str:
        """Classify fast motion patterns."""
        trajectory_data = metrics.get('trajectory', {})
        tempo_data = metrics.get('tempo', {})
        
        velocity_peak = trajectory_data.get('velocity', {}).get('peak', 0)
        tempo = tempo_data.get('tempo', 0)
        
        if velocity_peak > 5.0 and 60 <= tempo <= 120:
            return 'jumping'
        elif tempo > 150:
            return 'sprinting'
        else:
            return 'fast_movement'
    
    def _analyze_patterns(self, pose_sequence: np.ndarray) -> Dict:
        """Analyze motion patterns."""
        try:
            # Phase analysis
            phases = self.phase_analyzer.analyze_phases(pose_sequence)
            
            # Pattern matching (simplified)
            pattern_result = self.pattern_matcher.match_pattern(pose_sequence)
            
            return {
                'phases': phases,
                'pattern_match': pattern_result,
                'complexity': self._calculate_pattern_complexity(pose_sequence)
            }
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_temporal_features(self, pose_sequence: np.ndarray) -> Dict:
        """Analyze temporal features of motion."""
        try:
            # Calculate frame-to-frame variations
            if len(pose_sequence) < 2:
                return {'insufficient_data': True}
            
            # Calculate temporal consistency
            frame_differences = np.diff(pose_sequence, axis=0)
            temporal_consistency = 1.0 / (1.0 + np.std(frame_differences))
            
            # Calculate motion phases
            motion_intensity = np.linalg.norm(frame_differences, axis=(1, 2))
            
            return {
                'temporal_consistency': float(temporal_consistency),
                'motion_intensity': motion_intensity.tolist() if len(motion_intensity) < 100 else motion_intensity[:100].tolist(),
                'duration': len(pose_sequence) / self.fps,
                'frame_count': len(pose_sequence)
            }
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_pattern_complexity(self, pose_sequence: np.ndarray) -> float:
        """Calculate the complexity of the motion pattern."""
        try:
            if len(pose_sequence) < 3:
                return 0.0
            
            # Calculate spatial complexity (variation in pose configurations)
            pose_variations = np.var(pose_sequence, axis=0)
            spatial_complexity = np.mean(pose_variations)
            
            # Calculate temporal complexity (variation in motion speed)
            frame_differences = np.diff(pose_sequence, axis=0)
            motion_speeds = np.linalg.norm(frame_differences, axis=(1, 2))
            temporal_complexity = np.std(motion_speeds) if len(motion_speeds) > 1 else 0.0
            
            # Combine complexities
            total_complexity = (spatial_complexity + temporal_complexity) / 2
            
            # Normalize to [0, 1]
            return min(1.0, total_complexity / 10.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern complexity: {e}")
            return 0.0
    
    def classify_motion_type(self, pose_sequence: np.ndarray) -> str:
        """
        Simplified motion type classification.
        
        Args:
            pose_sequence: Array of pose keypoints
            
        Returns:
            String representing the motion type
        """
        result = self.classify(pose_sequence)
        return result.get('motion_class', 'unknown')
    
    def get_motion_features(self, pose_sequence: np.ndarray) -> Dict:
        """
        Extract motion features for further analysis.
        
        Args:
            pose_sequence: Array of pose keypoints
            
        Returns:
            Dictionary of motion features
        """
        result = self.classify(pose_sequence)
        return result.get('details', {}).get('metrics', {})
