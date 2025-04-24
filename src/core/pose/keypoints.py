import math
import numpy as np
from operator import itemgetter
import ray
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional
import time
import json
from pathlib import Path
import logging
from collections import defaultdict
import h5py
import pandas as pd

logger = logging.getLogger(__name__)

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])


@ray.remote
def distributed_extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    """Distributed keypoint extraction"""
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))
    keypoints = sorted(keypoints, key=itemgetter(0))

    return process_keypoints(keypoints, heatmap, all_keypoints, total_keypoint_num)


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def connections_nms(a_idx, b_idx, affinity_scores):
    # From all retrieved connections that share the same starting/ending keypoints leave only the top-scoring ones.
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]
    idx = []
    has_kpt_a = set()
    has_kpt_b = set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)
    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]


@ray.remote
def distributed_group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
    """Distributed keypoint grouping"""
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        
        # Process part in parallel
        pose_entries = process_part(part_id, kpts_a, kpts_b, part_pafs, pose_entries, all_keypoints, pose_entry_size, min_paf_score)

    return filter_poses(pose_entries), all_keypoints


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    points_per_limb = 10
    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
    all_keypoints_by_type = [np.array(keypoints, np.float32) for keypoints in all_keypoints_by_type]
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            continue

        # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
        a = kpts_a[:, :2]
        a = np.broadcast_to(a[None], (m, n, 2))
        b = kpts_b[:, :2]
        vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

        # Sample points along every candidate limb vector.
        steps = (1 / (points_per_limb - 1) * vec_raw)
        points = steps * grid + a.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Compute affinity score between candidate limb vectors and part affinity field.
        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
        vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
        vec = vec_raw / (vec_norm + 1e-6)
        affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)
        valid_affinity_scores = affinity_scores > min_paf_score
        valid_num = valid_affinity_scores.sum(1)
        affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
        success_ratio = valid_num / points_per_limb

        # Get a list of limbs according to the obtained affinity score.
        valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
        if len(valid_limbs) == 0:
            continue
        b_idx, a_idx = np.divmod(valid_limbs, n)
        affinity_scores = affinity_scores[valid_limbs]

        # Suppress incompatible connections.
        a_idx, b_idx, affinity_scores = connections_nms(a_idx, b_idx, affinity_scores)
        connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                               kpts_b[b_idx, 3].astype(np.int32),
                               affinity_scores))
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


@ray.remote
def process_keypoints(keypoints, heatmap, all_keypoints, total_keypoint_num):
    """Process detected keypoints in parallel"""
    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        
        # Parallel process each keypoint
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                        (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], 
                                     heatmap[keypoints[i][1], keypoints[i][0]],
                                     total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


@ray.remote
def filter_poses(pose_entries):
    """Filter pose entries in parallel"""
    filtered_entries = []
    
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    
    return np.asarray(filtered_entries)


@ray.remote
def process_part(part_id, kpts_a, kpts_b, part_pafs, pose_entries, all_keypoints, pose_entry_size, min_paf_score):
    """Process body part connections in parallel"""
    if len(kpts_a) == 0 or len(kpts_b) == 0:
        return pose_entries

    # Get vectors between all pairs of keypoints, i.e. candidate limb vectors
    a = kpts_a[:, :2]
    a = np.broadcast_to(a[None], (len(kpts_b), len(kpts_a), 2))
    b = kpts_b[:, :2]
    vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

    # Sample points along every candidate limb vector
    steps = 10
    points = np.zeros((len(vec_raw), steps, 2), dtype=np.int32)
    for i in range(steps):
        points[:, i, :] = a.reshape(-1, 2) + vec_raw.reshape(-1, 2) * i / (steps - 1)
    points = points.reshape(-1, 2)

    # Compute affinity score between candidate limb vectors and part affinity field
    field = part_pafs[points[:, 1], points[:, 0]].reshape(-1, steps, 2)
    vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
    vec = vec_raw / (vec_norm + 1e-6)
    affinity_scores = (field * vec).sum(-1).reshape(-1, steps)
    valid_affinity_scores = affinity_scores > min_paf_score
    valid_num = valid_affinity_scores.sum(1)
    affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
    success_ratio = valid_num / steps

    # Get list of limbs according to the obtained affinity score
    valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
    if len(valid_limbs) == 0:
        return pose_entries
    b_idx, a_idx = np.divmod(valid_limbs, len(kpts_a))
    affinity_scores = affinity_scores[valid_limbs]

    # Suppress incompatible connections
    a_idx, b_idx, affinity_scores = connections_nms(a_idx, b_idx, affinity_scores)
    connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                         kpts_b[b_idx, 3].astype(np.int32),
                         affinity_scores))

    if len(connections) == 0:
        return pose_entries

    if part_id == 0:
        pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
        for i in range(len(connections)):
            pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
            pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
            pose_entries[i][-1] = 2
            pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
    elif part_id == 17 or part_id == 18:
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        for i in range(len(connections)):
            for j in range(len(pose_entries)):
                if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                    pose_entries[j][kpt_b_id] = connections[i][1]
                elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                    pose_entries[j][kpt_a_id] = connections[i][0]
    else:
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        for i in range(len(connections)):
            num = 0
            for j in range(len(pose_entries)):
                if pose_entries[j][kpt_a_id] == connections[i][0]:
                    pose_entries[j][kpt_b_id] = connections[i][1]
                    num += 1
                    pose_entries[j][-1] += 1
                    pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
            if num == 0:
                pose_entry = np.ones(pose_entry_size) * -1
                pose_entry[kpt_a_id] = connections[i][0]
                pose_entry[kpt_b_id] = connections[i][1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                pose_entries.append(pose_entry)

    return pose_entries


@dataclass
class PoseKeypoint:
    """Data class for a single keypoint in a pose."""
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert keypoint to numpy array [x, y, z, confidence]."""
        return np.array([self.x, self.y, self.z, self.confidence], dtype=np.float32)
    
    @classmethod
    def from_array(cls, name: str, array: np.ndarray):
        """Create keypoint from numpy array [x, y, z, confidence]."""
        if len(array) == 4:
            return cls(name=name, x=float(array[0]), y=float(array[1]), 
                      z=float(array[2]), confidence=float(array[3]))
        elif len(array) == 3:
            return cls(name=name, x=float(array[0]), y=float(array[1]), 
                      confidence=float(array[2]))
        elif len(array) == 2:
            return cls(name=name, x=float(array[0]), y=float(array[1]))


class PoseData:
    """
    Efficient data structure for storing and querying pose keypoints.
    
    This class stores pose data in a more efficient format:
    - Uses numpy arrays for faster processing
    - Supports HDF5 storage for large datasets
    - Provides indexing and querying capabilities
    """
    
    # Standard keypoint names
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, keypoint_names: List[str] = None):
        """
        Initialize the pose data structure.
        
        Args:
            keypoint_names: List of keypoint names. If None, use standard names.
        """
        self.keypoint_names = keypoint_names or self.KEYPOINT_NAMES
        self.num_keypoints = len(self.keypoint_names)
        self.keypoint_indices = {name: idx for idx, name in enumerate(self.keypoint_names)}
        
        # Initialize data structures
        # Use a 3D numpy array for all pose data: [frame_idx, keypoint_idx, values]
        # where values are [x, y, z, confidence]
        self.poses = {}  # Dictionary mapping person_id -> numpy array
        self.frame_timestamps = {}  # Dictionary mapping frame_idx -> timestamp
        self.metadata = {}  # Additional metadata
        
        # Efficient tracking of keypoint positions across frames
        self.keypoint_history = defaultdict(dict)  # keypoint_name -> frame_idx -> position
        
    def add_pose(self, frame_idx: int, person_id: int, keypoints: Union[np.ndarray, List[PoseKeypoint]], 
                timestamp: float = None):
        """
        Add pose data for a single frame and person.
        
        Args:
            frame_idx: Frame index
            person_id: Person identifier
            keypoints: Array of keypoints with shape (num_keypoints, 4) or list of PoseKeypoint objects
            timestamp: Frame timestamp
        """
        # Initialize person's data if not already present
        if person_id not in self.poses:
            self.poses[person_id] = {}
            
        # Handle PoseKeypoint objects
        if isinstance(keypoints, list) and isinstance(keypoints[0], PoseKeypoint):
            keypoint_array = np.zeros((self.num_keypoints, 4), dtype=np.float32)
            for kp in keypoints:
                if kp.name in self.keypoint_indices:
                    idx = self.keypoint_indices[kp.name]
                    keypoint_array[idx] = kp.to_array()
            keypoints = keypoint_array
        
        # Store the pose data
        self.poses[person_id][frame_idx] = keypoints
        
        # Store timestamp if provided
        if timestamp:
            self.frame_timestamps[frame_idx] = timestamp
            
        # Update keypoint history for efficient querying
        for kp_idx, kp_name in enumerate(self.keypoint_names):
            if kp_idx < len(keypoints):
                self.keypoint_history[kp_name][frame_idx] = keypoints[kp_idx]
    
    def get_pose(self, frame_idx: int, person_id: int) -> Optional[np.ndarray]:
        """Get pose data for a specific frame and person."""
        if person_id in self.poses and frame_idx in self.poses[person_id]:
            return self.poses[person_id][frame_idx]
        return None
    
    def get_keypoint(self, frame_idx: int, person_id: int, keypoint_name: str) -> Optional[np.ndarray]:
        """Get a specific keypoint for a frame and person."""
        pose = self.get_pose(frame_idx, person_id)
        if pose is not None and keypoint_name in self.keypoint_indices:
            return pose[self.keypoint_indices[keypoint_name]]
        return None
    
    def get_keypoint_trajectory(self, keypoint_name: str, person_id: int, 
                               start_frame: int = 0, end_frame: int = None) -> np.ndarray:
        """
        Get the trajectory of a specific keypoint across frames.
        
        Returns:
            Array with shape (num_frames, 4) for [x, y, z, confidence]
        """
        if keypoint_name not in self.keypoint_indices:
            raise ValueError(f"Invalid keypoint name: {keypoint_name}")
            
        if person_id not in self.poses:
            return np.array([])
            
        # Determine frame range
        all_frames = sorted(self.poses[person_id].keys())
        if not all_frames:
            return np.array([])
            
        start_frame = max(start_frame, min(all_frames))
        end_frame = min(end_frame or max(all_frames), max(all_frames))
        
        # Extract trajectory data
        kp_idx = self.keypoint_indices[keypoint_name]
        trajectory = []
        
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in self.poses[person_id]:
                trajectory.append(self.poses[person_id][frame_idx][kp_idx])
            else:
                # Add NaN values for missing frames
                trajectory.append(np.array([np.nan, np.nan, np.nan, 0.0]))
                
        return np.array(trajectory)
    
    def get_all_frames(self, person_id: int = None) -> List[int]:
        """Get all frame indices with pose data."""
        if person_id is not None:
            return sorted(self.poses.get(person_id, {}).keys())
        
        # Combine frames from all persons
        all_frames = set()
        for p_id in self.poses:
            all_frames.update(self.poses[p_id].keys())
        return sorted(all_frames)
    
    def get_person_ids(self) -> List[int]:
        """Get all person IDs in the data."""
        return list(self.poses.keys())
    
    def save_to_hdf5(self, file_path: str):
        """
        Save pose data to HDF5 file for efficient storage and retrieval.
        
        Args:
            file_path: Path to save the HDF5 file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        with h5py.File(file_path, 'w') as f:
            # Create metadata group
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['num_keypoints'] = self.num_keypoints
            meta_grp.attrs['keypoint_names'] = json.dumps(self.keypoint_names)
            
            for key, value in self.metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    meta_grp.attrs[key] = value
                else:
                    try:
                        meta_grp.attrs[key] = json.dumps(value)
                    except:
                        logger.warning(f"Could not save metadata key {key} - unsupported type")
            
            # Create timestamps dataset
            timestamps = np.array([(k, v) for k, v in self.frame_timestamps.items()], 
                                 dtype=[('frame', 'i4'), ('timestamp', 'f8')])
            if len(timestamps) > 0:
                f.create_dataset('timestamps', data=timestamps)
            
            # Create poses group
            poses_grp = f.create_group('poses')
            
            # Store poses for each person
            for person_id, frames in self.poses.items():
                person_grp = poses_grp.create_group(str(person_id))
                
                # Convert to more efficient format for storing
                frame_indices = []
                poses_data = []
                
                for frame_idx, pose in frames.items():
                    frame_indices.append(frame_idx)
                    poses_data.append(pose)
                
                if frame_indices:
                    # Store frame indices
                    person_grp.create_dataset('frames', data=np.array(frame_indices, dtype=np.int32))
                    
                    # Store pose data as 3D array [frame, keypoint, values]
                    person_grp.create_dataset('keypoints', data=np.array(poses_data, dtype=np.float32))
        
        logger.info(f"Saved pose data to {file_path}")
    
    @classmethod
    def load_from_hdf5(cls, file_path: str) -> 'PoseData':
        """
        Load pose data from HDF5 file.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            PoseData object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            # Load metadata
            meta_grp = f['metadata']
            keypoint_names = json.loads(meta_grp.attrs['keypoint_names'])
            
            # Create PoseData object
            pose_data = cls(keypoint_names=keypoint_names)
            
            # Load other metadata
            for key, value in meta_grp.attrs.items():
                if key not in ['num_keypoints', 'keypoint_names']:
                    try:
                        pose_data.metadata[key] = json.loads(value) if isinstance(value, str) else value
                    except:
                        pose_data.metadata[key] = value
            
            # Load timestamps
            if 'timestamps' in f:
                timestamps = f['timestamps'][:]
                pose_data.frame_timestamps = {int(t[0]): float(t[1]) for t in timestamps}
            
            # Load poses
            if 'poses' in f:
                poses_grp = f['poses']
                
                for person_id in poses_grp:
                    person_grp = poses_grp[person_id]
                    
                    if 'frames' in person_grp and 'keypoints' in person_grp:
                        frames = person_grp['frames'][:]
                        keypoints = person_grp['keypoints'][:]
                        
                        # Add poses to the data structure
                        for i, frame_idx in enumerate(frames):
                            pose_data.add_pose(
                                frame_idx=int(frame_idx),
                                person_id=int(person_id),
                                keypoints=keypoints[i]
                            )
        
        logger.info(f"Loaded pose data from {file_path}")
        return pose_data
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert pose data to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with columns [frame, person_id, keypoint, x, y, z, confidence]
        """
        data = []
        
        for person_id, frames in self.poses.items():
            for frame_idx, keypoints in frames.items():
                timestamp = self.frame_timestamps.get(frame_idx, np.nan)
                
                for kp_idx, (x, y, z, conf) in enumerate(keypoints):
                    kp_name = self.keypoint_names[kp_idx]
                    data.append([
                        frame_idx, timestamp, person_id, kp_name, x, y, z, conf
                    ])
        
        return pd.DataFrame(
            data, 
            columns=['frame', 'timestamp', 'person_id', 'keypoint', 'x', 'y', 'z', 'confidence']
        )
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'PoseData':
        """
        Create PoseData object from pandas DataFrame.
        
        Args:
            df: DataFrame with columns [frame, person_id, keypoint, x, y, z, confidence]
            
        Returns:
            PoseData object
        """
        # Get unique keypoint names from the DataFrame
        keypoint_names = df['keypoint'].unique().tolist()
        pose_data = cls(keypoint_names=keypoint_names)
        
        # Add timestamps
        timestamp_df = df[['frame', 'timestamp']].drop_duplicates()
        for _, row in timestamp_df.iterrows():
            pose_data.frame_timestamps[int(row['frame'])] = float(row['timestamp'])
        
        # Process data for each frame and person
        for (frame, person_id), group in df.groupby(['frame', 'person_id']):
            # Sort by keypoint name to ensure correct order
            group = group.set_index('keypoint')
            
            # Create keypoint array
            keypoints = np.zeros((len(keypoint_names), 4), dtype=np.float32)
            
            for kp_idx, kp_name in enumerate(keypoint_names):
                if kp_name in group.index:
                    kp_data = group.loc[kp_name]
                    keypoints[kp_idx] = [kp_data['x'], kp_data['y'], kp_data['z'], kp_data['confidence']]
            
            # Add pose to data structure
            pose_data.add_pose(
                frame_idx=int(frame),
                person_id=int(person_id),
                keypoints=keypoints
            )
        
        return pose_data
    
    def get_statistics(self) -> Dict:
        """
        Calculate statistics about the pose data.
        
        Returns:
            Dictionary with statistics
        """
        num_persons = len(self.poses)
        total_poses = sum(len(frames) for frames in self.poses.values())
        
        # Calculate frames per person
        frames_per_person = {person_id: len(frames) for person_id, frames in self.poses.items()}
        
        # Calculate keypoint confidence
        confidences = []
        for person_id, frames in self.poses.items():
            for frame_idx, keypoints in frames.items():
                # Extract confidence values (4th element of each keypoint)
                conf = keypoints[:, 3]
                confidences.append(conf)
        
        if confidences:
            confidences = np.concatenate(confidences)
            avg_confidence = float(np.mean(confidences))
            min_confidence = float(np.min(confidences))
            max_confidence = float(np.max(confidences))
        else:
            avg_confidence = min_confidence = max_confidence = 0.0
        
        return {
            'num_persons': num_persons,
            'total_poses': total_poses,
            'frames_per_person': frames_per_person,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'num_frames': len(self.frame_timestamps),
            'keypoint_names': self.keypoint_names,
            'num_keypoints': self.num_keypoints
        }
