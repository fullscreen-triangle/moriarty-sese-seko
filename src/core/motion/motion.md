# Moriarty Motion Module Documentation

This document provides detailed descriptions of the files and functions in the motion module (`src/core/motion`). The module contains components for analyzing, detecting, tracking, and visualizing motion in sports videos.

## 1. `base.py`

This file contains base classes and data structures for combat sports motion analysis.

### Enums

#### `CombatSport`
- **Description**: Enumeration of supported combat sport types.
- **Values**:
  - `BOXING`: Boxing
  - `MMA`: Mixed Martial Arts
  - `FENCING`: Fencing
  - `KARATE`: Karate

### Data Classes

#### `PhysicsParams`
- **Description**: Dataclass for storing physics parameters related to motion.
- **Fields**:
  - `mass` (float): Mass in kg
  - `velocity` (float): Velocity in m/s
  - `acceleration` (float): Acceleration in m/s²
  - `angle` (float): Angle in radians
  - `momentum` (float): Momentum in kg⋅m/s
  - `impact_area` (float): Impact area in m²

#### `Combo`
- **Description**: Dataclass for representing a combination of strikes.
- **Fields**:
  - `strikes` (List[str]): List of strike types in the combo
  - `start_time` (float): Start time of the combo
  - `end_time` (float): End time of the combo
  - `success_rate` (float): Success rate of the combo
  - `total_force` (float): Total force of the combo
  - `confidence` (float): Confidence level of the combo detection

#### `StrikeMetrics`
- **Description**: Dataclass for storing metrics about a strike.
- **Fields**:
  - `type` (str): Type of strike
  - `velocity` (float): Velocity of the strike
  - `force` (float): Force of the strike
  - `accuracy` (float): Accuracy of the strike
  - `efficiency` (float): Efficiency of the strike
  - `rotation_power` (float): Rotational power component
  - `hip_engagement` (float): Hip engagement score
  - `balance_score` (float): Balance score during strike
  - `recovery_time` (float): Recovery time after strike
  - `confidence` (float): Confidence level of measurement

### Classes

#### `BaseCombatAnalyzer`
- **Description**: Abstract base class for combat sport analysis.
- **Methods**:
  - `analyze_frame(frame, poses, boxes)`: Abstract method to analyze a frame
  - `detect_strikes(poses, velocities)`: Abstract method to detect strikes
  - `detect_contacts(strikes, poses)`: Abstract method to detect contacts
  - `calculate_impact_force(velocity, mass, impact_area, angle)`: Calculates impact force using physics model
  - `detect_combos(strikes, max_interval)`: Detects combination attacks based on timing and patterns
  - `_analyze_combo(strikes)`: Analyzes a combination of strikes
  - `_evaluate_combo_timing(strikes)`: Evaluates the timing quality of a combination
  - `_evaluate_combo_pattern(strike_types)`: Evaluates if the combination follows common patterns

#### `AdvancedPhysicsEngine`
- **Description**: Engine for advanced physics calculations in combat sports.
- **Methods**:
  - `calculate_rotational_energy(angular_velocity, moment_of_inertia)`: Calculates rotational energy
  - `calculate_impact_force(mass, velocity, impact_time, area)`: Calculates impact force with air resistance
  - `calculate_momentum_transfer(mass, velocity, coefficient_of_restitution)`: Calculates momentum transfer
  - `calculate_power_generation(force, velocity, distance)`: Calculates power generation
  - `analyze_balance_dynamics(center_of_mass, support_polygon)`: Analyzes balance and stability
  - `_project_point_to_plane(point)`: Projects a 3D point onto the ground plane
  - `_calculate_stability_score(point, polygon)`: Calculates stability score within support polygon

#### `CombatPatternRecognition`
- **Description**: System for recognizing patterns in combat movements.
- **Methods**:
  - `analyze_sequence(moves, timings)`: Analyzes a sequence of moves
  - `_calculate_sequence_score(moves)`: Calculates a score for a sequence
  - `_analyze_timing_pattern(timings)`: Analyzes timing patterns
  - `_identify_known_patterns(moves)`: Identifies known patterns in move sequence
  - `update_pattern_memory(new_pattern)`: Updates the pattern memory

## 2. `motion_classifier.py`

This file contains classes for classifying motion and analyzing motion metrics.

### `ActionClassifier`
- **Description**: Classifies actions from pose sequences using a trained model.
- **Methods**:
  - `__init__(model_path, class_mapping)`: Initializes the classifier with model and class mapping
  - `_create_mock_model()`: Creates a simple mock model for testing
  - `predict(pose_sequence)`: Predicts action from a sequence of poses
  - `get_model_info()`: Returns information about the model

### `MotionMetricsCalculator`
- **Description**: Calculates various metrics from motion data.
- **Methods**:
  - `calculate_metrics(keypoints)`: Calculates comprehensive motion metrics
  - `_calculate_velocity(keypoints)`: Calculates velocity from keypoints
  - `_calculate_acceleration(velocity)`: Calculates acceleration from velocity
  - `_calculate_jerk(acceleration)`: Calculates jerk from acceleration
  - `_calculate_smoothness(jerk)`: Calculates motion smoothness
  - `_calculate_rom(keypoints)`: Calculates range of motion
  - `_calculate_stability(keypoints)`: Calculates stability

### `PhaseAnalyzer`
- **Description**: Analyzes phases in motion data.
- **Methods**:
  - `analyze_phases(motion_data)`: Identifies phases in motion data
  - `_classify_phase(window)`: Classifies a window of motion data into a phase type

### `PatternMatcher`
- **Description**: Matches motion patterns against templates.
- **Methods**:
  - `match_pattern(motion_sequence)`: Matches a motion sequence against templates
  - `_calculate_similarity(seq1, seq2)`: Calculates similarity between sequences
  - `_load_templates(path)`: Loads template patterns from path

### `SequenceAnalyzer`
- **Description**: Analyzes sequences of movements.
- **Methods**:
  - `analyze(sequence)`: Analyzes a sequence of movements
  - `_create_mock_segments(sequence)`: Creates mock segments for testing

### `SymmetryAnalyzer`
- **Description**: Analyzes symmetry in pose data.
- **Methods**:
  - `analyze(pose_data)`: Analyzes pose symmetry
  - `_generate_mock_recommendations()`: Generates mock recommendations

### `TempoAnalyzer`
- **Description**: Analyzes tempo and rhythm in motion data.
- **Methods**:
  - `analyze_tempo(motion_data)`: Analyzes tempo in motion data
  - `_calculate_frequency(data)`: Calculates frequency in data
  - `_detect_rhythm(data)`: Detects rhythm patterns
  - `_calculate_regularity(data)`: Calculates regularity of motion

### `TrajectoryAnalyzer`
- **Description**: Analyzes trajectories of movement.
- **Methods**:
  - `analyze(keypoints_sequence)`: Analyzes a sequence of keypoints
  - `_mock_velocity_profile()`: Creates mock velocity profile
  - `_mock_acceleration_profile()`: Creates mock acceleration profile
  - `_mock_key_points()`: Creates mock key points for testing

## 3. `movement_detector.py`

This file contains classes for detecting movement in video frames.

### Data Classes

#### `StablePeriod`
- **Description**: Dataclass representing a period of stability.
- **Fields**:
  - `start_frame` (int): Starting frame of the stable period
  - `end_frame` (Optional[int]): Ending frame of the stable period
  - `positions` (List[Tuple[float, float]]): List of positions during the stable period

### Enums

#### `StabilityState`
- **Description**: Enumeration of stability states.
- **Values**:
  - `STABLE`: Stable state
  - `MOVING`: Moving state

### Classes

#### `SpeedEstimator`
- **Description**: Estimates speed and detects stable periods.
- **Methods**:
  - `__init__(fps, track_length, stability_threshold, min_stable_frames)`: Initializes the estimator
  - `calibrate(frame_width)`: Calibrates the pixel-to-meter conversion
  - `estimate_speed_and_check_stability(track_history, current_frame_idx)`: Estimates speed and checks stability
  - `estimate_speed(track_history)`: Estimates speed from track history
  - `_update_stability_periods(speed, frame_idx, position)`: Updates stable period detection
  - `_calculate_bilateral_symmetry(keypoints)`: Calculates bilateral symmetry
  - `_calculate_temporal_symmetry(keypoints)`: Calculates temporal symmetry
  - `_calculate_frequency(data)`: Calculates frequency using FFT
  - `_detect_rhythm(data)`: Detects rhythmic patterns
  - `_calculate_regularity(data)`: Calculates movement regularity
  - `_smooth_trajectory(points)`: Smooths a trajectory
  - `_calculate_path_length(points)`: Calculates path length
  - `_calculate_curvature(points)`: Calculates trajectory curvature
  - `_calculate_complexity(points)`: Calculates trajectory complexity
  - `get_stable_periods()`: Returns detected stable periods

## 4. `movement_tracker.py`

This file contains the MovementTracker class for tracking movement based on pose data.

### `MovementTracker`
- **Description**: Tracks movement based on pose keypoints.
- **Methods**:
  - `__init__(tracking_threshold, window_size)`: Initializes the tracker
  - `track(pose_data)`: Tracks movement based on pose keypoints
  - `_calculate_com(pose_data)`: Calculates center of mass
  - `_calculate_velocity()`: Calculates velocity from position history
  - `_calculate_movement_magnitude()`: Calculates magnitude of movement
  - `_calculate_movement_direction(velocity)`: Determines movement direction
  - `reset()`: Resets the tracker state

## 5. `visualization.py`

This file contains the CombatVisualizer class for visualizing combat analysis.

### `CombatVisualizer`
- **Description**: Visualizes combat analysis on video frames.
- **Methods**:
  - `__init__(sport_type)`: Initializes the visualizer for a specific sport
  - `draw_analysis(frame, analysis_data)`: Draws analysis overlays on frame
  - `_draw_skeleton(frame, poses)`: Draws sport-specific skeleton
  - `_draw_techniques(frame, techniques)`: Draws technique-specific visualizations
  - `_draw_force_vectors(frame, forces)`: Draws force vectors
  - `_draw_metrics(frame, metrics)`: Draws performance metrics overlay

## 6. `pixel_change_detector.py`

This file contains classes for detecting pixel changes in video frames.

### Data Classes

#### `ActivityMetrics`
- **Description**: Dataclass for activity metrics from pixel change detection.
- **Fields**:
  - `motion_intensity` (float): Intensity of motion
  - `motion_area_ratio` (float): Ratio of motion area to total area
  - `motion_centroid` (Tuple[float, float]): Center of motion
  - `direction_vector` (Tuple[float, float]): Direction of motion
  - `is_active` (bool): Whether activity is detected

### Classes

#### `PixelChangeDetector`
- **Description**: Detects pixel changes for activity quantification.
- **Methods**:
  - `__init__(min_intensity_threshold, min_area_threshold, history_length)`: Initializes the detector
  - `detect_activity(frame, roi)`: Detects and quantifies activity
  - `get_motion_heatmap(frame)`: Generates motion intensity heatmap

## 7. `stabilography.py`

This file contains the StabilographyAnalyzer class for stability analysis.

### `StabilographyAnalyzer`
- **Description**: Analyzes Center of Pressure (CoP) data for stability metrics.
- **Methods**:
  - `__init__(sampling_rate)`: Initializes the analyzer
  - `analyze_stability(cop_positions)`: Analyzes CoP data for stability metrics
  - `_calculate_basic_parameters(x, y)`: Calculates basic stability parameters
  - `_analyze_frequency_domain(x, y)`: Analyzes frequency domain of CoP data
  - `_rambling_trembling_decomposition(x, y)`: Decomposes CoP into rambling and trembling components
  - `detect_stance_phase(positions, velocity_threshold)`: Detects stance phases based on velocity
