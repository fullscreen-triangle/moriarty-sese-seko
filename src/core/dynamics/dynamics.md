# Moriarty Dynamics Module Documentation

This document provides detailed descriptions of the files and functions in the dynamics module (`src/core/dynamics`). The module contains components for analyzing athlete movements, calculating kinematic and dynamic properties, and assessing synchronization between multiple athletes.

## 1. `kinematics_analyzer.py`

This file contains classes for analyzing the kinematic properties of athlete movements, such as joint angles, velocities, and accelerations.

### `Segment` Class

- **Description**: A dataclass for representing body segments with their physical properties.
- **Properties**:
  - `length` (float): Length of the segment
  - `com_ratio` (float): Center of mass position as a ratio of segment length

### `KinematicsAnalyzer` Class

#### `__init__(self, fps: float)`
- **Description**: Initializes the KinematicsAnalyzer with framerate information.
- **Parameters**:
  - `fps` (float): Frames per second of the video
- **Behavior**:
  - Sets up MediaPipe pose detection
  - Defines body segments with their physical properties
  - Initializes previous positions and velocities for tracking

#### `_extract_landmarks(self, results) -> Dict`
- **Description**: Extracts relevant landmarks from MediaPipe pose detection results.
- **Parameters**:
  - `results`: MediaPipe pose detection results
- **Returns**: Dictionary containing landmark positions for key joints
- **Behavior**:
  - Converts MediaPipe landmarks to a dictionary of joint positions
  - Extracts positions for hips, knees, ankles, shoulders, elbows, and wrists

#### `calculate_kinematics(self, athlete_id: int, pose_results) -> Dict`
- **Description**: Main function to calculate all kinematic properties.
- **Parameters**:
  - `athlete_id` (int): Identifier for the athlete
  - `pose_results`: MediaPipe pose detection results
- **Returns**: Dictionary containing joint angles, positions, velocities, and accelerations
- **Behavior**:
  - Extracts landmark positions
  - Calculates joint angles
  - Calculates segment positions
  - Calculates velocities and accelerations using finite differences

#### `_calculate_joint_angles(self, skeleton_data: Dict) -> Dict`
- **Description**: Calculates 3D joint angles from skeleton data.
- **Parameters**:
  - `skeleton_data` (Dict): Dictionary of joint positions
- **Returns**: Dictionary mapping joint names to angle measurements
- **Behavior**:
  - Calculates vectors between adjacent joints
  - Computes angles between these vectors in different anatomical planes
  - Returns a structured dictionary of joint angles

#### `_calculate_angle_3d(self, v1: np.ndarray, v2: np.ndarray, plane: str) -> float`
- **Description**: Calculates angle between vectors in a specified anatomical plane.
- **Parameters**:
  - `v1` (np.ndarray): First vector
  - `v2` (np.ndarray): Second vector
  - `plane` (str): Anatomical plane ('sagittal', 'frontal', or 'transverse')
- **Returns**: Angle in radians
- **Behavior**:
  - Projects vectors onto the specified anatomical plane
  - Calculates the angle between the projected vectors using dot product

#### `_calculate_segment_positions(self, skeleton_data: Dict) -> Dict`
- **Description**: Calculates positions of body segments.
- **Parameters**:
  - `skeleton_data` (Dict): Dictionary of joint positions
- **Returns**: Dictionary of segment positions
- **Behavior**: Returns the landmark positions (placeholder implementation)

#### `_calculate_derivatives(self, athlete_id: int, current_positions: Dict) -> Tuple[Dict, Dict]`
- **Description**: Calculates velocities and accelerations using finite differences.
- **Parameters**:
  - `athlete_id` (int): Identifier for the athlete
  - `current_positions` (Dict): Current joint positions
- **Returns**: Tuple of dictionaries containing velocities and accelerations
- **Behavior**:
  - Uses the previous frame's positions to calculate velocities
  - Uses the previous frame's velocities to calculate accelerations
  - Handles the first frame as a special case

## 2. `stride_analyzer.py`

This file contains classes for analyzing stride characteristics of athletes.

### `StrideMetrics` Class

- **Description**: A dataclass for storing metrics related to an athlete's stride.
- **Properties**:
  - `left_stride_length` (float): Length of left leg stride
  - `right_stride_length` (float): Length of right leg stride
  - `stride_frequency` (float): Stride frequency in Hz
  - `contact_time` (float): Time in contact with ground in seconds
  - `flight_time` (float): Time in air during stride in seconds
  - `asymmetry_index` (float): Measure of stride asymmetry
  - `phase` (float): Phase of the stride cycle

### `StrideAnalyzer` Class

#### `__init__(self, fps: int)`
- **Description**: Initializes the StrideAnalyzer with framerate information.
- **Parameters**:
  - `fps` (int): Frames per second of the video
- **Behavior**:
  - Sets up logging
  - Calculates frame interval from fps
  - Initializes storage for previous positions

#### `analyze_stride(self, skeleton_data: Dict, athlete_id: int) -> Dict`
- **Description**: Main function to analyze stride characteristics.
- **Parameters**:
  - `skeleton_data` (Dict): Dictionary of joint positions
  - `athlete_id` (int): Identifier for the athlete
- **Returns**: Dictionary containing stride metrics
- **Behavior**:
  - Extracts ankle and hip positions
  - Calculates stride length for left and right legs
  - Estimates stride frequency
  - Calculates ground contact and flight times
  - Measures asymmetry between left and right strides
  - Calculates stride phase

#### `_calculate_stride_length(self, prev_pos: Optional[np.ndarray], current_pos: np.ndarray) -> float`
- **Description**: Calculates stride length between consecutive frames.
- **Parameters**:
  - `prev_pos` (Optional[np.ndarray]): Previous foot position
  - `current_pos` (np.ndarray): Current foot position
- **Returns**: Stride length in distance units
- **Behavior**: Calculates Euclidean distance between consecutive foot positions

#### `_calculate_stride_frequency(self, current_hip: np.ndarray, prev_hip: np.ndarray) -> float`
- **Description**: Estimates stride frequency based on hip movement.
- **Parameters**:
  - `current_hip` (np.ndarray): Current hip position
  - `prev_hip` (np.ndarray): Previous hip position
- **Returns**: Stride frequency in Hz
- **Behavior**:
  - Calculates vertical displacement of hip
  - Estimates frequency based on vertical oscillation

#### `_estimate_ground_contact(self, left_foot: np.ndarray, right_foot: np.ndarray) -> float`
- **Description**: Estimates ground contact time based on foot height.
- **Parameters**:
  - `left_foot` (np.ndarray): Left foot position
  - `right_foot` (np.ndarray): Right foot position
- **Returns**: Estimated contact time in seconds
- **Behavior**:
  - Checks if either foot is below a height threshold
  - Returns full frame interval if in contact, zero otherwise

#### `_calculate_asymmetry(self, left_stride: float, right_stride: float) -> float`
- **Description**: Calculates asymmetry between left and right strides.
- **Parameters**:
  - `left_stride` (float): Left stride length
  - `right_stride` (float): Right stride length
- **Returns**: Asymmetry index between 0 and 1
- **Behavior**: Calculates normalized difference between stride lengths

#### `_calculate_phase(self, left_foot: np.ndarray, right_foot: np.ndarray) -> float`
- **Description**: Calculates the phase of the stride cycle.
- **Parameters**:
  - `left_foot` (np.ndarray): Left foot position
  - `right_foot` (np.ndarray): Right foot position
- **Returns**: Phase value between 0 and 1
- **Behavior**: Uses vertical and horizontal foot positions to estimate cycle phase

## 3. `sync_analyzer.py`

This file contains classes for analyzing synchronization between multiple athletes.

### `SyncMetrics` Class

- **Description**: A dataclass for storing metrics related to synchronization between athletes.
- **Properties**:
  - `phase_difference` (float): Average phase difference between athletes
  - `coupling_strength` (float): Strength of coupling between athletes' movements
  - `sync_index` (float): Overall synchronization index
  - `relative_phase` (float): Instantaneous relative phase

### `SynchronizationAnalyzer` Class

#### `__init__(self, window_size: int)`
- **Description**: Initializes the SynchronizationAnalyzer.
- **Parameters**:
  - `window_size` (int): Number of frames to use for analysis window
- **Behavior**:
  - Sets up logging
  - Initializes storage for phase and stride history

#### `analyze_sync(self, frame_data: Dict) -> Dict`
- **Description**: Main function to analyze synchronization between athletes.
- **Parameters**:
  - `frame_data` (Dict): Dictionary containing athlete data for a frame
- **Returns**: Dictionary mapping athlete pairs to synchronization metrics
- **Behavior**:
  - Updates phase and stride history for each athlete
  - Calculates synchronization metrics for each pair of athletes
  - Returns a dictionary of metrics for each athlete pair

#### `_calculate_phase_difference(self, id1: int, id2: int) -> float`
- **Description**: Calculates the average phase difference between two athletes.
- **Parameters**:
  - `id1` (int): ID of first athlete
  - `id2` (int): ID of second athlete
- **Returns**: Average phase difference
- **Behavior**: Calculates mean absolute difference between phase histories

#### `_calculate_coupling_strength(self, id1: int, id2: int) -> float`
- **Description**: Calculates the strength of coupling between two athletes.
- **Parameters**:
  - `id1` (int): ID of first athlete
  - `id2` (int): ID of second athlete
- **Returns**: Coupling strength between 0 and 1
- **Behavior**: Uses Pearson correlation between stride histories

#### `_calculate_sync_index(self, id1: int, id2: int) -> float`
- **Description**: Calculates the overall synchronization index.
- **Parameters**:
  - `id1` (int): ID of first athlete
  - `id2` (int): ID of second athlete
- **Returns**: Synchronization index between 0 and 1
- **Behavior**: Uses coherence function to measure frequency-domain synchronization

#### `_calculate_relative_phase(self, id1: int, id2: int) -> float`
- **Description**: Calculates the instantaneous relative phase.
- **Parameters**:
  - `id1` (int): ID of first athlete
  - `id2` (int): ID of second athlete
- **Returns**: Relative phase angle in radians
- **Behavior**: Calculates phase angle between most recent phase values

## 4. `grf_analyzer.py`

This file contains a class for estimating ground reaction forces (GRF) from kinematic data.

### `GRFAnalyzer` Class

#### `__init__(self)`
- **Description**: Initializes the GRFAnalyzer.
- **Behavior**:
  - Sets gravitational constant
  - Sets default body mass

#### `estimate_grf(self, positions: Dict, accelerations: Dict) -> Dict`
- **Description**: Main function to estimate ground reaction forces.
- **Parameters**:
  - `positions` (Dict): Joint positions
  - `accelerations` (Dict): Joint accelerations
- **Returns**: Dictionary containing vertical and horizontal GRF and impact force
- **Behavior**:
  - Estimates vertical GRF
  - Estimates horizontal GRF
  - Estimates impact force

#### `_estimate_vertical_grf(self, accelerations: Dict) -> float`
- **Description**: Estimates vertical ground reaction force.
- **Parameters**:
  - `accelerations` (Dict): Joint accelerations
- **Returns**: Estimated vertical GRF in Newtons
- **Behavior**: Uses center of mass acceleration and body mass to estimate force

#### `_estimate_horizontal_grf(self, accelerations: Dict) -> float`
- **Description**: Estimates horizontal ground reaction force.
- **Parameters**:
  - `accelerations` (Dict): Joint accelerations
- **Returns**: Estimated horizontal GRF in Newtons
- **Behavior**: Uses center of mass horizontal acceleration and body mass

#### `_estimate_impact_force(self, positions: Dict) -> float`
- **Description**: Estimates impact force during landing.
- **Parameters**:
  - `positions` (Dict): Joint positions
- **Returns**: Estimated impact force in Newtons
- **Behavior**: Provides a simplified estimate based on body weight

## 5. `dynamics_analyzer.py`

This file contains classes for inverse dynamics analysis to estimate forces and moments at joints.

### `Segment` Class

- **Description**: A dataclass for representing body segments with their physical properties.
- **Properties**:
  - `mass` (float): Mass of the segment in kg
  - `length` (float): Length of the segment in meters
  - `inertia` (float): Moment of inertia of the segment

### `DynamicsAnalyzer` Class

#### `__init__(self)`
- **Description**: Initializes the DynamicsAnalyzer.
- **Behavior**:
  - Sets gravitational constant
  - Defines body segments with their physical properties

#### `calculate_dynamics(self, positions: Dict, velocities: Dict, accelerations: Dict) -> Dict`
- **Description**: Main function to calculate forces and moments using inverse dynamics.
- **Parameters**:
  - `positions` (Dict): Joint positions
  - `velocities` (Dict): Joint velocities
  - `accelerations` (Dict): Joint accelerations
- **Returns**: Dictionary containing forces and moments at each joint
- **Behavior**:
  - Starts with ground reaction force
  - Applies inverse dynamics to each segment moving up the kinetic chain
  - Returns forces and moments at proximal and distal ends of each segment

#### `_inverse_dynamics(self, positions: Dict, accelerations: Dict, segment: str, seg_data: Segment, Fd: np.ndarray, Md: float) -> Tuple[np.ndarray, float]`
- **Description**: Calculates forces and moments for a segment using inverse dynamics.
- **Parameters**:
  - `positions` (Dict): Joint positions
  - `accelerations` (Dict): Joint accelerations
  - `segment` (str): Segment name
  - `seg_data` (Segment): Segment physical properties
  - `Fd` (np.ndarray): Distal force
  - `Md` (float): Distal moment
- **Returns**: Tuple containing proximal force and moment
- **Behavior**:
  - Uses Newton's equations of motion
  - Calculates proximal force based on distal force, mass, and acceleration
  - Provides simplified moment calculation
