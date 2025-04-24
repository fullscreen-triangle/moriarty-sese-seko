from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import logging
from scipy.signal import coherence
from scipy.stats import pearsonr


@dataclass
class SyncMetrics:
    phase_difference: float
    coupling_strength: float
    sync_index: float
    relative_phase: float


class SynchronizationAnalyzer:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        self.phase_history = {}
        self.stride_history = {}

    def analyze_sync(self, frame_data: Dict) -> Dict:
        try:
            athletes_data = frame_data.get('athletes', [])
            stride_data = frame_data.get('stride_data', [])

            if len(athletes_data) < 2:
                return None

            sync_results = {}

            # Update histories
            for stride_info in stride_data:
                athlete_id = stride_info['athlete_id']
                metrics = stride_info['metrics']

                if athlete_id not in self.phase_history:
                    self.phase_history[athlete_id] = []
                if athlete_id not in self.stride_history:
                    self.stride_history[athlete_id] = []

                self.phase_history[athlete_id].append(metrics.phase)
                self.stride_history[athlete_id].append(
                    (metrics.left_stride_length + metrics.right_stride_length) / 2
                )

                # Keep history within window size
                self.phase_history[athlete_id] = self.phase_history[athlete_id][-self.window_size:]
                self.stride_history[athlete_id] = self.stride_history[athlete_id][-self.window_size:]

            # Calculate synchronization metrics for each pair
            athlete_ids = list(self.phase_history.keys())
            for i in range(len(athlete_ids)):
                for j in range(i + 1, len(athlete_ids)):
                    id1, id2 = athlete_ids[i], athlete_ids[j]

                    phase_diff = self._calculate_phase_difference(id1, id2)
                    coupling = self._calculate_coupling_strength(id1, id2)
                    sync_idx = self._calculate_sync_index(id1, id2)
                    rel_phase = self._calculate_relative_phase(id1, id2)

                    sync_results[f"{id1}_{id2}"] = SyncMetrics(
                        phase_difference=phase_diff,
                        coupling_strength=coupling,
                        sync_index=sync_idx,
                        relative_phase=rel_phase
                    )

            return sync_results

        except Exception as e:
            self.logger.error(f"Error in synchronization analysis: {str(e)}")
            return None

    def _calculate_phase_difference(self, id1: int, id2: int) -> float:
        if len(self.phase_history[id1]) < 2 or len(self.phase_history[id2]) < 2:
            return 0.0

        phase1 = np.array(self.phase_history[id1])
        phase2 = np.array(self.phase_history[id2])
        return np.mean(np.abs(phase1 - phase2))

    def _calculate_coupling_strength(self, id1: int, id2: int) -> float:
        if len(self.stride_history[id1]) < 2 or len(self.stride_history[id2]) < 2:
            return 0.0

        stride1 = np.array(self.stride_history[id1])
        stride2 = np.array(self.stride_history[id2])

        correlation, _ = pearsonr(stride1, stride2)
        return abs(correlation)

    def _calculate_sync_index(self, id1: int, id2: int) -> float:
        if len(self.phase_history[id1]) < self.window_size or \
                len(self.phase_history[id2]) < self.window_size:
            return 0.0

        phase1 = np.array(self.phase_history[id1])
        phase2 = np.array(self.phase_history[id2])

        # Calculate phase coherence
        f, Cxy = coherence(phase1, phase2)
        return np.mean(Cxy)

    def _calculate_relative_phase(self, id1: int, id2: int) -> float:
        if len(self.phase_history[id1]) < 2 or len(self.phase_history[id2]) < 2:
            return 0.0

        phase1 = self.phase_history[id1][-1]
        phase2 = self.phase_history[id2][-1]

        return np.arctan2(np.sin(phase1 - phase2), np.cos(phase1 - phase2))
