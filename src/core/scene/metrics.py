import numpy as np
from typing import Dict, List, Optional
import time
import ray
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

@ray.remote
class MetricsCalculator:
    def __init__(self):
        """Initialize the metrics calculator."""
        self.previous_positions = {}
        self.previous_time = None
        self.fps = 30  # Default FPS, will be updated based on video
        self.pool = ThreadPoolExecutor(max_workers=mp.cpu_count())
        
    def calculate_metrics(self, pose_data: Dict) -> Dict:
        """Calculate various metrics from pose data in parallel."""
        if not pose_data or "points" not in pose_data:
            return {}
            
        current_time = time.time()
        points = pose_data["points"]
        
        metrics = {}
        
        # Calculate metrics in parallel
        futures = []
        
        if self.previous_positions and self.previous_time:
            dt = 1.0 / self.fps
            
            # Submit velocity calculations
            futures.append(
                self.pool.submit(self._calculate_velocities, points, dt)
            )
            
            # Submit acceleration calculations
            futures.append(
                self.pool.submit(self._calculate_accelerations, points, dt)
            )
        
        # Submit distance calculations if multiple athletes
        if "other_athlete_points" in pose_data:
            futures.append(
                self.pool.submit(
                    self._calculate_distances, 
                    points, 
                    pose_data["other_athlete_points"]
                )
            )
        
        # Gather results
        for future in futures:
            metrics.update(future.result())
        
        # Store current data for next frame
        self.previous_positions = points
        self.previous_time = current_time
        
        return metrics
    
    def _calculate_velocities(self, points: np.ndarray, dt: float) -> Dict[str, float]:
        """Calculate velocities of key body parts in parallel."""
        velocities = {}
        
        # Key points to track (indices for MediaPipe pose landmarks)
        key_points = {
            "head": 0,
            "right_shoulder": 12,
            "left_shoulder": 11,
            "right_hip": 24,
            "left_hip": 23,
            "right_knee": 26,
            "left_knee": 25,
            "right_ankle": 28,
            "left_ankle": 27
        }
        
        def calculate_point_velocity(args):
            name, idx = args
            if idx < len(points) and idx < len(self.previous_positions):
                velocity = np.linalg.norm(
                    (points[idx] - self.previous_positions[idx]) / dt
                )
                return (f"{name}_velocity", velocity)
            return None
        
        # Calculate velocities in parallel
        with ThreadPoolExecutor() as executor:
            results = executor.map(calculate_point_velocity, key_points.items())
            
        # Collect results
        for result in results:
            if result:
                velocities[result[0]] = result[1]
        
        # Calculate overall body velocity (using center of mass)
        com_current = np.mean(points[[11, 12, 23, 24]], axis=0)
        com_prev = np.mean(self.previous_positions[[11, 12, 23, 24]], axis=0)
        velocities["body_velocity"] = np.linalg.norm((com_current - com_prev) / dt)
        
        return velocities
    
    def _calculate_accelerations(self, points: np.ndarray, dt: float) -> Dict[str, float]:
        """Calculate accelerations of key body parts."""
        accelerations = {}
        
        # Calculate acceleration for center of mass
        com_current = np.mean(points[[11, 12, 23, 24]], axis=0)
        com_prev = np.mean(self.previous_positions[[11, 12, 23, 24]], axis=0)
        
        velocity_current = (com_current - com_prev) / dt
        if hasattr(self, 'previous_velocity'):
            acceleration = (velocity_current - self.previous_velocity) / dt
            accelerations["body_acceleration"] = np.linalg.norm(acceleration)
        
        self.previous_velocity = velocity_current
        
        return accelerations
    
    def _calculate_distances(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, float]:
        """Calculate distances between two athletes in parallel."""
        distances = {}
        
        # Calculate distance between centers of mass
        com1 = np.mean(points1[[11, 12, 23, 24]], axis=0)
        com2 = np.mean(points2[[11, 12, 23, 24]], axis=0)
        distances["athlete_distance"] = np.linalg.norm(com1 - com2)
        
        def calculate_point_distances(point1_idx):
            min_dist = float('inf')
            for point2_idx in range(len(points2)):
                dist = np.linalg.norm(points1[point1_idx] - points2[point2_idx])
                min_dist = min(min_dist, dist)
            return min_dist
        
        # Calculate minimum distances in parallel
        with ThreadPoolExecutor() as executor:
            point_distances = list(executor.map(calculate_point_distances, range(len(points1))))
            
        distances["minimum_separation"] = min(point_distances)
        
        return distances
    
    def set_fps(self, fps: float):
        """Update the FPS value for velocity calculations."""
        self.fps = fps 