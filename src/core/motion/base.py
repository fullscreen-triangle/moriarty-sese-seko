from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time




class CombatSport(Enum):
    BOXING = "boxing"
    MMA = "mma"
    FENCING = "fencing"
    KARATE = "karate"


@dataclass
class PhysicsParams:
    mass: float  # kg
    velocity: float  # m/s
    acceleration: float  # m/s²
    angle: float  # radians
    momentum: float  # kg⋅m/s
    impact_area: float  # m²


@dataclass
class Combo:
    strikes: List[str]
    start_time: float
    end_time: float
    success_rate: float
    total_force: float
    confidence: float


class BaseCombatAnalyzer(ABC):
    @abstractmethod
    def analyze_frame(self, frame: np.ndarray, poses: Dict, boxes: Dict) -> Dict:
        pass

    @abstractmethod
    def detect_strikes(self, poses: Dict, velocities: Dict) -> List[Dict]:
        pass

    @abstractmethod
    def detect_contacts(self, strikes: List[Dict], poses: Dict) -> List[Dict]:
        pass

    def calculate_impact_force(self,
                               velocity: float,
                               mass: float,
                               impact_area: float,
                               angle: float) -> float:
        """
        Calculate impact force using improved physics model.

        F = m * a + (1/2 * ρ * v² * A * Cd)

        Where:
        - m = mass of striking object
        - a = acceleration
        - ρ = air density
        - v = velocity
        - A = impact area
        - Cd = drag coefficient
        """
        air_density = 1.225  # kg/m³
        drag_coef = 0.5  # approximate for human limb

        # Force due to mass and acceleration
        force_mass = mass * (velocity ** 2 / 0.1)  # assuming 0.1s impact time

        # Force due to air resistance
        force_drag = 0.5 * air_density * (velocity ** 2) * impact_area * drag_coef

        # Angle adjustment (maximum force at 90 degrees)
        angle_factor = np.sin(angle)

        total_force = (force_mass + force_drag) * angle_factor
        return total_force

    def detect_combos(self,
                      strikes: List[Dict],
                      max_interval: float = 0.5) -> List[Combo]:
        """
        Detect combination attacks based on timing and patterns.
        """
        combos = []
        current_combo = []

        for i, strike in enumerate(strikes):
            if not current_combo:
                current_combo.append(strike)
                continue

            time_diff = strike['timestamp'] - current_combo[-1]['timestamp']

            if time_diff <= max_interval:
                current_combo.append(strike)
            else:
                if len(current_combo) >= 2:
                    combo = self._analyze_combo(current_combo)
                    combos.append(combo)
                current_combo = [strike]

        # Handle last combo
        if len(current_combo) >= 2:
            combo = self._analyze_combo(current_combo)
            combos.append(combo)

        return combos

    def _analyze_combo(self, strikes: List[Dict]) -> Combo:
        """
        Analyze a combination of strikes.
        """
        strike_types = [s['type'] for s in strikes]
        start_time = strikes[0]['timestamp']
        end_time = strikes[-1]['timestamp']

        # Calculate success rate based on impact confidence
        success_rate = np.mean([s['confidence'] for s in strikes])

        # Calculate total force
        total_force = sum(s['impact_force'] for s in strikes)

        # Calculate combo confidence based on timing and pattern recognition
        timing_score = self._evaluate_combo_timing(strikes)
        pattern_score = self._evaluate_combo_pattern(strike_types)
        confidence = (timing_score + pattern_score) / 2

        return Combo(
            strikes=strike_types,
            start_time=start_time,
            end_time=end_time,
            success_rate=success_rate,
            total_force=total_force,
            confidence=confidence
        )

    def _evaluate_combo_timing(self, strikes: List[Dict]) -> float:
        """
        Evaluate the timing quality of a combination.
        """
        intervals = []
        for i in range(1, len(strikes)):
            interval = strikes[i]['timestamp'] - strikes[i - 1]['timestamp']
            intervals.append(interval)

        # Good combinations should have consistent timing
        if not intervals:
            return 0.0

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        # Higher score for more consistent timing
        consistency_score = 1.0 / (1.0 + std_interval)

        # Penalty for too slow combinations
        speed_score = 1.0 / (1.0 + mean_interval)

        return (consistency_score + speed_score) / 2

    def _evaluate_combo_pattern(self, strike_types: List[str]) -> float:
        """
        Evaluate if the combination follows common effective patterns.
        To be implemented by specific sport analyzers.
        """
        return 0.0







@dataclass
class StrikeMetrics:
    type: str
    velocity: float
    force: float
    accuracy: float
    efficiency: float
    rotation_power: float
    hip_engagement: float
    balance_score: float
    recovery_time: float
    confidence: float


class AdvancedPhysicsEngine:
    def __init__(self):
        self.g = 9.81  # gravitational constant
        self.air_density = 1.225  # kg/m³
        self.drag_coefficient = 0.5

    def calculate_rotational_energy(self,
                                    angular_velocity: float,
                                    moment_of_inertia: float) -> float:
        """Calculate rotational energy of a striking limb."""
        return 0.5 * moment_of_inertia * angular_velocity ** 2

    def calculate_impact_force(self,
                               mass: float,
                               velocity: float,
                               impact_time: float,
                               area: float) -> float:
        """Calculate impact force with air resistance."""
        # Basic impulse-momentum
        base_force = mass * velocity / impact_time

        # Air resistance component
        drag_force = 0.5 * self.air_density * velocity ** 2 * area * self.drag_coefficient

        return base_force + drag_force

    def calculate_momentum_transfer(self,
                                    mass: float,
                                    velocity: float,
                                    coefficient_of_restitution: float) -> float:
        """Calculate momentum transfer during impact."""
        return mass * velocity * (1 + coefficient_of_restitution)

    def calculate_power_generation(self,
                                   force: float,
                                   velocity: float,
                                   distance: float) -> float:
        """Calculate power generation during strike."""
        return force * velocity * distance

    def analyze_balance_dynamics(self,
                                 center_of_mass: np.ndarray,
                                 support_polygon: List[np.ndarray]) -> float:
        """Analyze balance and stability during movement."""
        # Calculate stability score based on COM projection and support base
        com_projection = self._project_point_to_plane(center_of_mass)
        stability_score = self._calculate_stability_score(com_projection, support_polygon)
        return stability_score

    def _project_point_to_plane(self, point: np.ndarray) -> np.ndarray:
        """Project a 3D point onto the ground plane."""
        return np.array([point[0], point[1], 0])

    def _calculate_stability_score(self,
                                   point: np.ndarray,
                                   polygon: List[np.ndarray]) -> float:
        """Calculate stability score based on point location within polygon."""
        # Implementation of point-in-polygon and distance-based scoring
        return 0.0  # Placeholder


class CombatPatternRecognition:
    def __init__(self):
        self.pattern_memory = []
        self.common_combinations = {
            'striking': {
                'jab-cross': 0.8,
                'jab-cross-hook': 0.9,
                'low-kick-high-kick': 0.7
            },
            'grappling': {
                'shoot-sprawl': 0.75,
                'clinch-throw': 0.85,
                'guard-sweep': 0.8
            }
        }

    def analyze_sequence(self,
                         moves: List[str],
                         timings: List[float]) -> Dict:
        """Analyze a sequence of combat moves."""
        sequence_score = self._calculate_sequence_score(moves)
        timing_score = self._analyze_timing_pattern(timings)

        return {
            'sequence_score': sequence_score,
            'timing_score': timing_score,
            'total_score': (sequence_score + timing_score) / 2,
            'recognized_patterns': self._identify_known_patterns(moves)
        }

    def _calculate_sequence_score(self, moves: List[str]) -> float:
        """Calculate how well a sequence of moves flows together."""
        score = 0.0
        for i in range(len(moves) - 1):
            transition = f"{moves[i]}-{moves[i + 1]}"
            score += self.common_combinations['striking'].get(transition, 0.0)
            score += self.common_combinations['grappling'].get(transition, 0.0)
        return score / max(len(moves) - 1, 1)

    def _analyze_timing_pattern(self, timings: List[float]) -> float:
        """Analyze the timing pattern of a sequence."""
        if len(timings) < 2:
            return 0.0

        intervals = np.diff(timings)
        consistency = 1.0 / (1.0 + np.std(intervals))
        speed = 1.0 / np.mean(intervals)

        return (consistency + speed) / 2

    def _identify_known_patterns(self, moves: List[str]) -> List[str]:
        """Identify known patterns in a sequence of moves."""
        patterns = []
        sequence = '-'.join(moves)

        for pattern in self.common_combinations['striking'].keys():
            if pattern in sequence:
                patterns.append(f"Striking: {pattern}")

        for pattern in self.common_combinations['grappling'].keys():
            if pattern in sequence:
                patterns.append(f"Grappling: {pattern}")

        return patterns

    def update_pattern_memory(self, new_pattern: List[str]):
        """Update the memory of observed patterns."""
        self.pattern_memory.append(new_pattern)
        if len(self.pattern_memory) > 100:  # Keep last 100 patterns
            self.pattern_memory.pop(0)
