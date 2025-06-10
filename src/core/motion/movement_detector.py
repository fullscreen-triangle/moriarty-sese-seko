import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

from scipy.signal import savgol_filter


@dataclass
class StablePeriod:
    start_frame: int
    end_frame: Optional[int]
    positions: List[Tuple[float, float]]


class StabilityState(Enum):
    STABLE = "stable"
    MOVING = "moving"


class SpeedEstimator:
    def __init__(self, fps: float, track_length: float = 100,
                 stability_threshold: float = 0.5,
                 min_stable_frames: int = 30):
        self.fps = fps
        self.track_length = track_length
        self.pixel_per_meter = None
        self.stability_threshold = stability_threshold
        self.min_stable_frames = min_stable_frames
        self.stable_periods: List[StablePeriod] = []
        self.current_stable_period: Optional[StablePeriod] = None
        self.current_state = StabilityState.MOVING

    def calibrate(self, frame_width: int) -> None:
        self.pixel_per_meter = frame_width / self.track_length

    def estimate_speed_and_check_stability(self,
                                         track_history: List[Dict],
                                         current_frame_idx: int) -> Tuple[float, bool]:
        speed = self.estimate_speed(track_history)
        is_stable = self._update_stability_periods(speed, current_frame_idx,
                                                 track_history[-1]['center'])
        return speed, is_stable

    def estimate_speed(self, track_history: List[Dict]) -> float:
        if len(track_history) < 2:
            return 0.0

        start = track_history[0]['center']
        end = track_history[-1]['center']
        pixel_distance = np.linalg.norm(np.array(end) - np.array(start))
        meters = pixel_distance / self.pixel_per_meter
        time_seconds = len(track_history) / self.fps

        return meters / time_seconds if time_seconds > 0 else 0.0

    def _update_stability_periods(self,
                                speed: float,
                                frame_idx: int,
                                position: Tuple[float, float]) -> bool:
        if speed < self.stability_threshold:
            if self.current_state == StabilityState.MOVING:
                self.current_stable_period = StablePeriod(
                    start_frame=frame_idx,
                    end_frame=None,
                    positions=[position]
                )
                self.current_state = StabilityState.STABLE
            else:
                self.current_stable_period.positions.append(position)
            return True
        else:
            if self.current_state == StabilityState.STABLE:
                self.current_stable_period.end_frame = frame_idx - 1
                if (self.current_stable_period.end_frame -
                        self.current_stable_period.start_frame >= self.min_stable_frames):
                    self.stable_periods.append(self.current_stable_period)
                self.current_stable_period = None
                self.current_state = StabilityState.MOVING
            return False

    def _calculate_bilateral_symmetry(self, keypoints: np.ndarray) -> float:
        """Berechnet die bilaterale Symmetrie zwischen linker und rechter Körperhälfte"""
        left_points = keypoints[:, :len(self.reference_points) // 2]
        right_points = keypoints[:, len(self.reference_points) // 2:]

        # Spiegele die rechten Punkte
        right_points_mirrored = np.copy(right_points)
        right_points_mirrored[:, 0] *= -1

        # Berechne mittleren Abstand zwischen gespiegelten Punkten
        symmetry_error = np.mean(np.linalg.norm(
            left_points - right_points_mirrored, axis=1))

        # Normalisiere auf [0, 1], wobei 1 perfekte Symmetrie bedeutet
        return 1.0 / (1.0 + symmetry_error)

    def _calculate_temporal_symmetry(self, keypoints: np.ndarray) -> float:
        """Berechnet die zeitliche Symmetrie der Bewegung"""
        n = len(keypoints)
        if n < 2:
            return 0.0

        # Teile Sequenz in zwei Hälften
        mid = n // 2
        first_half = keypoints[:mid]
        second_half = np.flip(keypoints[mid:], axis=0)

        # Berechne Korrelation zwischen den Hälften
        min_len = min(len(first_half), len(second_half))
        correlation = np.corrcoef(
            first_half[:min_len].flatten(),
            second_half[:min_len].flatten()
        )[0, 1]

        return max(0.0, correlation)

    def _calculate_frequency(self, data: np.ndarray) -> float:
        """Berechnet die Hauptfrequenz der Bewegung mittels FFT"""
        if len(data) < 2:
            return 0.0

        # FFT durchführen
        fft = np.fft.fft(data - np.mean(data, axis=0), axis=0)
        freqs = np.fft.fftfreq(len(data), 1 / self.fps)

        # Finde dominante Frequenz
        magnitude = np.abs(fft)
        main_freq_idx = np.argmax(magnitude[1:len(freqs) // 2]) + 1
        return freqs[main_freq_idx]

    def _detect_rhythm(self, data: np.ndarray) -> List:
        """Erkennt rhythmische Muster in der Bewegung"""
        # Finde lokale Maxima als Schlagpunkte
        peaks = []
        window = int(0.1 * self.fps)  # 100ms Fenster

        smoothed = np.convolve(
            np.linalg.norm(data, axis=1),
            np.ones(window) / window,
            mode='valid'
        )

        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] < smoothed[i] > smoothed[i + 1]:
                peaks.append(i)

        return peaks

    def _calculate_regularity(self, data: np.ndarray) -> float:
        """Berechnet die Regelmäßigkeit der Bewegung"""
        peaks = self._detect_rhythm(data)
        if len(peaks) < 2:
            return 0.0

        # Berechne Standardabweichung der Intervalle
        intervals = np.diff(peaks)
        regularity = 1.0 / (1.0 + np.std(intervals))
        return regularity

    def _smooth_trajectory(self, points: np.ndarray) -> np.ndarray:
        """Glättet die Trajektorie mit einem Savitzky-Golay Filter"""
        if len(points) < self.smoothing_window:
            return points

        return savgol_filter(points, self.smoothing_window, 3, axis=0)

    def _calculate_path_length(self, points: np.ndarray) -> float:
        """Berechnet die Gesamtlänge des zurückgelegten Weges"""
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    def _calculate_curvature(self, points: np.ndarray) -> np.ndarray:
        """Berechnet die Krümmung der Trajektorie"""
        if len(points) < 3:
            return np.zeros(len(points))

        # Berechne erste und zweite Ableitung
        velocity = np.gradient(points, axis=0)
        acceleration = np.gradient(velocity, axis=0)

        # Krümmung nach Frenet-Formel
        speed = np.linalg.norm(velocity, axis=1)
        curvature = np.cross(velocity, acceleration) / (speed ** 3)
        return np.abs(curvature)

    def _calculate_complexity(self, points: np.ndarray) -> float:
        """Berechnet die Komplexität der Trajektorie"""
        if len(points) < 2:
            return 0.0

        # Kombiniere Pfadlänge und mittlere Krümmung
        path_length = self._calculate_path_length(points)
        mean_curvature = np.mean(self._calculate_curvature(points))

        return path_length * mean_curvature

    def get_stable_periods(self) -> List[StablePeriod]:
        return self.stable_periods


class MovementDetector:
    """
    Detects and analyzes movement in pose sequences.
    """
    
    def __init__(self, movement_threshold: float = 0.1, fps: float = 30.0):
        """
        Initialize the movement detector.
        
        Args:
            movement_threshold: Minimum movement magnitude to consider as movement
            fps: Frames per second of the video
        """
        self.movement_threshold = movement_threshold
        self.fps = fps
        self.logger = logging.getLogger(__name__)
        self.speed_estimator = SpeedEstimator(fps)
        
    def detect_movement(self, keypoints: np.ndarray) -> Dict:
        """
        Detect movement from keypoints data.
        
        Args:
            keypoints: Array of pose keypoints [N, K, 3] where N=frames, K=keypoints, 3=x,y,confidence
            
        Returns:
            Dictionary containing movement analysis
        """
        if keypoints is None or len(keypoints) == 0:
            return {
                'is_moving': False,
                'movement_type': 'stationary',
                'movement_magnitude': 0.0,
                'velocity': [0.0, 0.0],
                'confidence': 0.0
            }
        
        try:
            # Calculate center of mass from keypoints
            if keypoints.ndim == 3:
                # Multiple frames: [frames, keypoints, coords]
                com_sequence = self._calculate_com_sequence(keypoints)
                movement_metrics = self._analyze_movement_sequence(com_sequence)
            else:
                # Single frame: [keypoints, coords]
                com = self._calculate_com_single(keypoints)
                movement_metrics = {
                    'is_moving': False,
                    'movement_type': 'stationary',
                    'movement_magnitude': 0.0,
                    'velocity': [0.0, 0.0],
                    'confidence': 1.0
                }
            
            return movement_metrics
            
        except Exception as e:
            self.logger.error(f"Error in movement detection: {e}")
            return {
                'is_moving': False,
                'movement_type': 'error',
                'movement_magnitude': 0.0,
                'velocity': [0.0, 0.0],
                'confidence': 0.0
            }
    
    def _calculate_com_single(self, keypoints: np.ndarray) -> np.ndarray:
        """Calculate center of mass for a single frame."""
        # Use torso keypoints (shoulders and hips) for stability
        if keypoints.shape[0] >= 12:  # Ensure we have enough keypoints
            torso_indices = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip
            torso_keypoints = keypoints[torso_indices, :2]  # x, y coordinates only
            # Filter out low-confidence keypoints
            valid_keypoints = torso_keypoints[keypoints[torso_indices, 2] > 0.5]
            if len(valid_keypoints) > 0:
                return np.mean(valid_keypoints, axis=0)
        
        # Fallback: use all available keypoints
        valid_keypoints = keypoints[keypoints[:, 2] > 0.5, :2]
        if len(valid_keypoints) > 0:
            return np.mean(valid_keypoints, axis=0)
        else:
            return np.array([0.0, 0.0])
    
    def _calculate_com_sequence(self, keypoints: np.ndarray) -> np.ndarray:
        """Calculate center of mass sequence for multiple frames."""
        com_sequence = []
        for frame_keypoints in keypoints:
            com = self._calculate_com_single(frame_keypoints)
            com_sequence.append(com)
        return np.array(com_sequence)
    
    def _analyze_movement_sequence(self, com_sequence: np.ndarray) -> Dict:
        """Analyze movement from center of mass sequence."""
        if len(com_sequence) < 2:
            return {
                'is_moving': False,
                'movement_type': 'insufficient_data',
                'movement_magnitude': 0.0,
                'velocity': [0.0, 0.0],
                'confidence': 0.0
            }
        
        # Calculate velocities
        velocities = np.diff(com_sequence, axis=0) * self.fps
        
        # Calculate movement magnitude
        movement_magnitudes = np.linalg.norm(velocities, axis=1)
        avg_movement_magnitude = np.mean(movement_magnitudes)
        
        # Determine if moving
        is_moving = avg_movement_magnitude > self.movement_threshold
        
        # Calculate average velocity
        avg_velocity = np.mean(velocities, axis=0)
        
        # Classify movement type
        movement_type = self._classify_movement_type(velocities, movement_magnitudes)
        
        # Calculate confidence based on consistency
        confidence = self._calculate_movement_confidence(movement_magnitudes)
        
        return {
            'is_moving': is_moving,
            'movement_type': movement_type,
            'movement_magnitude': float(avg_movement_magnitude),
            'velocity': avg_velocity.tolist(),
            'confidence': float(confidence)
        }
    
    def _classify_movement_type(self, velocities: np.ndarray, magnitudes: np.ndarray) -> str:
        """Classify the type of movement."""
        if np.mean(magnitudes) < self.movement_threshold:
            return 'stationary'
        
        # Analyze velocity direction consistency
        if len(velocities) > 1:
            velocity_consistency = self._calculate_velocity_consistency(velocities)
            
            if velocity_consistency > 0.8:
                # Consistent direction - linear movement
                avg_velocity = np.mean(velocities, axis=0)
                if abs(avg_velocity[0]) > abs(avg_velocity[1]):
                    return 'horizontal_movement'
                else:
                    return 'vertical_movement'
            elif velocity_consistency < 0.3:
                # Very inconsistent - oscillatory or complex movement
                return 'oscillatory_movement'
            else:
                # Moderate consistency - general movement
                return 'general_movement'
        
        return 'movement'
    
    def _calculate_velocity_consistency(self, velocities: np.ndarray) -> float:
        """Calculate how consistent the velocity directions are."""
        if len(velocities) < 2:
            return 1.0
        
        # Normalize velocities to unit vectors
        magnitudes = np.linalg.norm(velocities, axis=1)
        valid_indices = magnitudes > 1e-6
        
        if np.sum(valid_indices) < 2:
            return 0.0
        
        normalized_velocities = velocities[valid_indices] / magnitudes[valid_indices, np.newaxis]
        
        # Calculate pairwise dot products
        dot_products = []
        for i in range(len(normalized_velocities) - 1):
            dot_product = np.dot(normalized_velocities[i], normalized_velocities[i + 1])
            dot_products.append(np.clip(dot_product, -1.0, 1.0))
        
        # Average consistency
        return np.mean(dot_products)
    
    def _calculate_movement_confidence(self, magnitudes: np.ndarray) -> float:
        """Calculate confidence in movement detection based on magnitude consistency."""
        if len(magnitudes) == 0:
            return 0.0
        
        # High confidence if magnitudes are consistent
        std_magnitude = np.std(magnitudes)
        mean_magnitude = np.mean(magnitudes)
        
        if mean_magnitude < 1e-6:
            return 1.0 if std_magnitude < 1e-6 else 0.0
        
        # Coefficient of variation - lower is better
        cv = std_magnitude / mean_magnitude
        confidence = 1.0 / (1.0 + cv)
        
        return confidence
    
    def detect_movement_events(self, keypoints_sequence: np.ndarray) -> List[Dict]:
        """
        Detect discrete movement events in a sequence.
        
        Args:
            keypoints_sequence: Sequence of keypoints [frames, keypoints, coords]
            
        Returns:
            List of movement events with start/end frames and characteristics
        """
        if keypoints_sequence is None or len(keypoints_sequence) < 3:
            return []
        
        com_sequence = self._calculate_com_sequence(keypoints_sequence)
        velocities = np.diff(com_sequence, axis=0) * self.fps
        magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Find movement events using peak detection
        events = []
        in_movement = False
        start_frame = 0
        
        for i, magnitude in enumerate(magnitudes):
            if not in_movement and magnitude > self.movement_threshold:
                # Start of movement
                in_movement = True
                start_frame = i
            elif in_movement and magnitude <= self.movement_threshold:
                # End of movement
                in_movement = False
                events.append({
                    'start_frame': start_frame,
                    'end_frame': i,
                    'duration': (i - start_frame) / self.fps,
                    'peak_magnitude': np.max(magnitudes[start_frame:i+1]),
                    'avg_magnitude': np.mean(magnitudes[start_frame:i+1])
                })
        
        # Handle case where movement continues to end
        if in_movement:
            events.append({
                'start_frame': start_frame,
                'end_frame': len(magnitudes),
                'duration': (len(magnitudes) - start_frame) / self.fps,
                'peak_magnitude': np.max(magnitudes[start_frame:]),
                'avg_magnitude': np.mean(magnitudes[start_frame:])
            })
        
        return events
