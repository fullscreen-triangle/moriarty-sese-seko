from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple
import mediapipe as mp  # For MediaPipe landmarks
import cv2


@dataclass
class Segment:
    length: float
    com_ratio: float


class KinematicsAnalyzer:
    def __init__(self, fps: float):
        self.fps = fps
        self.mp_pose = mp.solutions.pose
        self.segments = {
            'thigh': Segment(length=0.4, com_ratio=0.433),
            'shank': Segment(length=0.4, com_ratio=0.433),
            'foot': Segment(length=0.2, com_ratio=0.5),
            'upper_arm': Segment(length=0.3, com_ratio=0.436),
            'forearm': Segment(length=0.3, com_ratio=0.430)
        }
        self.prev_positions = {}
        self.prev_velocities = {}

    def _extract_landmarks(self, results) -> Dict:
        """Extract relevant landmarks from MediaPipe results"""
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        return {
            # Lower body
            'hip': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].z]),
            'knee': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].z]),
            'ankle': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].z]),
            # Upper body
            'shoulder': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]),
            'elbow': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].z]),
            'wrist': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].z])
        }

    def calculate_kinematics(self, athlete_id: int, pose_results) -> Dict:
        """Main function to calculate kinematics from MediaPipe pose results"""
        skeleton_data = self._extract_landmarks(pose_results)
        if skeleton_data is None:
            return None

        # Calculate joint angles
        angles = self._calculate_joint_angles(skeleton_data)

        # Calculate segment positions
        positions = self._calculate_segment_positions(skeleton_data)

        # Calculate velocities and accelerations
        velocities, accelerations = self._calculate_derivatives(athlete_id, positions)

        return {
            'joint_angles': angles,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations
        }

    def _calculate_joint_angles(self, skeleton_data: Dict) -> Dict:
        """Calculate 3D joint angles"""
        # Lower body vectors
        hip_to_knee = skeleton_data['knee'] - skeleton_data['hip']
        knee_to_ankle = skeleton_data['ankle'] - skeleton_data['knee']

        # Upper body vectors
        shoulder_to_elbow = skeleton_data['elbow'] - skeleton_data['shoulder']
        elbow_to_wrist = skeleton_data['wrist'] - skeleton_data['elbow']

        # Reference vector (vertical)
        vertical = np.array([0, 1, 0])

        angles = {
            'hip': {
                'bend': self._calculate_angle_3d(vertical, hip_to_knee, plane='sagittal'),
                'turn': self._calculate_angle_3d(vertical, hip_to_knee, plane='transverse'),
                'tilt': self._calculate_angle_3d(vertical, hip_to_knee, plane='frontal')
            },
            'knee': {
                'bend': self._calculate_angle_3d(hip_to_knee, knee_to_ankle, plane='sagittal')
            },
            'shoulder': {
                'bend': self._calculate_angle_3d(vertical, shoulder_to_elbow, plane='sagittal'),
                'turn': self._calculate_angle_3d(vertical, shoulder_to_elbow, plane='transverse'),
                'tilt': self._calculate_angle_3d(vertical, shoulder_to_elbow, plane='frontal')
            },
            'elbow': {
                'bend': self._calculate_angle_3d(shoulder_to_elbow, elbow_to_wrist, plane='sagittal')
            }
        }

        return angles

    def _calculate_angle_3d(self, v1: np.ndarray, v2: np.ndarray, plane: str) -> float:
        """Calculate angle between vectors in specified anatomical plane"""
        if plane == 'sagittal':
            # Project onto Y-Z plane
            v1_proj = np.array([0, v1[1], v1[2]])
            v2_proj = np.array([0, v2[1], v2[2]])
        elif plane == 'frontal':
            # Project onto X-Z plane
            v1_proj = np.array([v1[0], 0, v1[2]])
            v2_proj = np.array([v2[0], 0, v2[2]])
        else:  # transverse
            # Project onto X-Y plane
            v1_proj = np.array([v1[0], v1[1], 0])
            v2_proj = np.array([v2[0], v2[1], 0])

        # Rest of your angle calculation code remains the same
        dot_product = np.dot(v1_proj, v2_proj)
        norms = np.linalg.norm(v1_proj) * np.linalg.norm(v2_proj)

        if norms == 0:
            return 0

        return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

    # Your existing methods remain the same
    def _calculate_segment_positions(self, skeleton_data: Dict) -> Dict:
        return skeleton_data  # For now, just return the landmark positions

    def _calculate_derivatives(self, athlete_id: int, current_positions: Dict) -> Tuple[Dict, Dict]:
        # Your existing derivatives calculation code remains the same
        if athlete_id not in self.prev_positions:
            self.prev_positions[athlete_id] = current_positions
            self.prev_velocities[athlete_id] = {k: np.zeros(3) for k in current_positions.keys()}
            return ({k: np.zeros(3) for k in current_positions.keys()},
                    {k: np.zeros(3) for k in current_positions.keys()})

        dt = 1.0 / self.fps
        velocities = {}
        accelerations = {}

        for joint in current_positions.keys():
            velocities[joint] = (current_positions[joint] -
                                 self.prev_positions[athlete_id][joint]) / dt
            accelerations[joint] = (velocities[joint] -
                                    self.prev_velocities[athlete_id][joint]) / dt

        self.prev_positions[athlete_id] = current_positions
        self.prev_velocities[athlete_id] = velocities

        return velocities, accelerations
