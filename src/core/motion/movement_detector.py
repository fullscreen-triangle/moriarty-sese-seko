import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

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
