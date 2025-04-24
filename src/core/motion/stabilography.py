import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
from scipy.spatial import ConvexHull


class StabilographyAnalyzer:
    def __init__(self, sampling_rate: float = 30.0):
        self.fs = sampling_rate
        self.window_size = int(1 * sampling_rate)  # 1 second window

    def analyze_stability(self, cop_positions: List[Tuple[float, float]]) -> Dict:
        """
        Analyze Center of Pressure (CoP) data for stability metrics

        Args:
            cop_positions: List of (x,y) coordinates representing CoP trajectory
        """
        cop_array = np.array(cop_positions)
        x_coords = cop_array[:, 0]
        y_coords = cop_array[:, 1]

        basic_params = self._calculate_basic_parameters(x_coords, y_coords)
        freq_params = self._analyze_frequency_domain(x_coords, y_coords)
        rambling_trembling = self._rambling_trembling_decomposition(x_coords, y_coords)

        return {
            'basic_parameters': basic_params,
            'frequency_parameters': freq_params,
            'rambling_trembling': rambling_trembling
        }

    def _calculate_basic_parameters(self, x: np.ndarray, y: np.ndarray) -> Dict:
        path_length = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        mean_velocity = path_length / (len(x) / self.fs)

        points = np.column_stack((x, y))
        hull = ConvexHull(points)
        sway_area = hull.area

        sd_x = np.std(x)
        sd_y = np.std(y)
        range_x = np.max(x) - np.min(x)
        range_y = np.max(y) - np.min(y)

        return {
            'path_length': path_length,
            'mean_velocity': mean_velocity,
            'sway_area': sway_area,
            'sd_x': sd_x,
            'sd_y': sd_y,
            'range_x': range_x,
            'range_y': range_y
        }

    def _analyze_frequency_domain(self, x: np.ndarray, y: np.ndarray) -> Dict:
        fx, px = signal.welch(x, fs=self.fs)
        fy, py = signal.welch(y, fs=self.fs)

        mean_freq_x = np.sum(fx * px) / np.sum(px)
        mean_freq_y = np.sum(fy * py) / np.sum(py)

        cum_sum_x = np.cumsum(px)
        cum_sum_y = np.cumsum(py)
        median_freq_x = fx[np.where(cum_sum_x >= cum_sum_x[-1] / 2)[0][0]]
        median_freq_y = fy[np.where(cum_sum_y >= cum_sum_y[-1] / 2)[0][0]]

        return {
            'mean_frequency': {'x': mean_freq_x, 'y': mean_freq_y},
            'median_frequency': {'x': median_freq_x, 'y': median_freq_y},
            'power_spectrum': {
                'x': {'frequencies': fx.tolist(), 'power': px.tolist()},
                'y': {'frequencies': fy.tolist(), 'power': py.tolist()}
            }
        }

    def _rambling_trembling_decomposition(self, x: np.ndarray, y: np.ndarray) -> Dict:
        dx = np.gradient(x)
        dy = np.gradient(y)

        zero_crossings_x = np.where(np.diff(np.signbit(dx)))[0]
        zero_crossings_y = np.where(np.diff(np.signbit(dy)))[0]

        t = np.arange(len(x))
        rambling_x = np.interp(t, zero_crossings_x, x[zero_crossings_x])
        rambling_y = np.interp(t, zero_crossings_y, y[zero_crossings_y])

        trembling_x = x - rambling_x
        trembling_y = y - rambling_y

        rambling_metrics = {
            'amplitude': {
                'x': np.std(rambling_x),
                'y': np.std(rambling_y)
            },
            'path_length': np.sum(np.sqrt(np.diff(rambling_x) ** 2 + np.diff(rambling_y) ** 2))
        }

        trembling_metrics = {
            'amplitude': {
                'x': np.std(trembling_x),
                'y': np.std(trembling_y)
            },
            'path_length': np.sum(np.sqrt(np.diff(trembling_x) ** 2 + np.diff(trembling_y) ** 2))
        }

        return {
            'rambling': {
                'x': rambling_x.tolist(),
                'y': rambling_y.tolist(),
                'metrics': rambling_metrics
            },
            'trembling': {
                'x': trembling_x.tolist(),
                'y': trembling_y.tolist(),
                'metrics': trembling_metrics
            }
        }

    def detect_stance_phase(self, positions: List[Tuple[float, float]],
                            velocity_threshold: float = 0.05) -> List[bool]:
        positions = np.array(positions)
        velocities = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        is_stance = np.concatenate(([True], velocities < velocity_threshold))
        return is_stance.tolist()
