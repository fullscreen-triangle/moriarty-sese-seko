# Moriarty

A framework for sports video analysis using computer vision and biomechanical principles.

<p align="center">
  <img src="greyhound_sketch_fixed.png" alt="Moriarty Logo" width="250" />
</p>

## Overview

Moriarty is a framework for analyzing sports videos through computer vision techniques to extract biomechanical data. The system is designed to help coaches, athletes, and researchers gain insights about athletic performance and technique.

## Key Features

- **Video Analysis**: Process sports videos to extract pose data
- **Pose Estimation**: Detect and track human pose throughout video sequences
- **Biomechanical Analysis**: Calculate joint angles, velocities, and other metrics
- **Distributed Processing**: Efficiently process videos using parallel computing
- **Visualization**: Generate annotated videos with overlay metrics

## Architecture

Moriarty consists of several integrated components:

1. **Core Video Processing**: Frame extraction and preprocessing
2. **Pose Analysis**: Human detection and skeleton tracking
3. **Biomechanics Engine**: Computation of kinematics and dynamics
4. **Distributed Computing**: Parallel processing using Ray and Dask
5. **Visualization Tools**: Overlay metrics and skeleton rendering

## Technical Implementation

### Pose Analysis Module

The pose analysis module implements several key algorithms:

- **Human Detection**: Uses YOLOv8 for person detection with confidence thresholds optimized for athletic movements
- **Pose Estimation**: Implements MediaPipe's BlazePose model (33-point skeleton) with custom adaptations for sports movements
- **Skeleton Tracking**: Uses DeepSORT with Kalman filtering for maintaining identity across frames
- **Skeleton Drawing**: Custom OpenCV drawing with joint confidence visualization

Model specifications:
- MediaPipe pose complexity: Levels 0 (fast), 1 (balanced), 2 (accurate)
- YOLOv8 with IOU threshold of 0.45 and confidence threshold of 0.6 for athlete detection

### Dynamics Analysis Module

The dynamics module implements biomechanical analysis algorithms:

- **Kinematics Analyzer**: Uses numerical differentiation (central difference method) with Savitzky-Golay filtering to compute velocities and accelerations
- **Stride Analyzer**: Implements a contact detection algorithm based on ankle velocity and acceleration patterns with zero-crossing detection
- **Synchronization Analyzer**: Cross-correlation analysis between joint movements, with Dynamic Time Warping (DTW) for temporal alignment
- **GRF Analyzer**: Implements a modified spring-mass model for ground reaction force estimation with body segment parameters derived from anthropometric tables

Mathematical approaches:
- Quaternion-based joint angle calculations to avoid gimbal lock
- Butterworth filtering (4th order, low-pass) for noise reduction
- Principal Component Analysis for movement pattern extraction

### Distributed Processing Implementation

The distributed system enables efficient parallel processing:

- **Memory Monitor**: Real-time memory tracking using `psutil` with adaptive throttling
- **Worker Management**: Dynamic worker allocation based on task complexity and available resources
- **Task Distribution**: Coarse-grained parallelism at the video level, fine-grained at the frame level
- **Checkpointing**: Periodic state saving to enable recovery from interruptions

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- FFmpeg for video processing

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/moriarty.git
cd moriarty

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For detailed installation options, see [INSTALL.md](docs/INSTALL.md).

## Usage

### Basic Video Analysis

```python
from src.pipeline import VideoPipeline

# Initialize pipeline
pipeline = VideoPipeline(memory_limit=0.4)

# Analyze a video
result = pipeline.process_video(
    video_path="videos/sprint.mp4",
    output_dir="results/",
    create_visualizations=True
)
```

### Command Line Interface

```bash
# Process a single video
python -m src.cli.main analyze --input path/to/video.mp4 --output results

# Batch process multiple videos
python -m src.cli.main batch --input-dir videos --output-dir results --pattern "*.mp4"
```

### Turbulance DSL Integration

Moriarty integrates with the **Turbulance** domain-specific language for advanced sports analysis and evidence-based reasoning. Turbulance provides structured constructs for hypothesis testing, probabilistic reasoning, and pattern analysis.

#### Basic Turbulance Syntax

```turbulance
// Sprint analysis using propositions and motions
proposition SprintAnalysis:
    context athlete_data = load_athlete_profile("sprinter.json")
    
    motion OptimalTechnique("Athlete demonstrates optimal sprint technique"):
        target_stride_frequency: 4.5..5.2
        ground_contact_time: 0.08..0.12
        maximum_velocity_zone: 60..80  // meters
    
    within video_analysis_results:
        given stride_frequency in OptimalTechnique.target_stride_frequency:
            support OptimalTechnique with_confidence(
                measurement_quality * environmental_factor
            )
        given ground_contact_time in OptimalTechnique.ground_contact_time:
            support OptimalTechnique with_confidence(0.85)

// Bayesian network for performance analysis
bayesian_network PerformanceNetwork:
    nodes:
        - technique: TechniqueEvidence(confidence_threshold: 0.8)
        - biomechanics: BiomechanicalEvidence(precision: 0.02)
        - performance: PerformanceEvidence(measurement_accuracy: 0.95)
    
    edges:
        - technique -> performance: causal_strength(0.85, fuzziness: 0.15)
        - biomechanics -> technique: influence_strength(0.75, fuzziness: 0.2)
    
    inference:
        method: "variational_bayes"
        convergence_threshold: 0.001
        max_iterations: 1000

// Sensor fusion for multi-modal analysis
sensor_fusion MultiModalAnalysis:
    sensors:
        - video_pose: PoseDetection(confidence: 0.9, fps: 60)
        - force_plates: ForceData(sampling_rate: 1000)
        - emg_sensors: EMGData(channels: 8, filtering: "bandpass")
    
    fusion_method: "kalman_filter"
    uncertainty_quantification: true
    temporal_alignment: "cross_correlation"
```

#### Key Turbulance Features

- **Propositions**: Structured hypothesis testing framework for sports performance analysis
- **Motions**: Sub-hypotheses within propositions for granular biomechanical analysis
- **Evidence Integration**: Multi-modal data validation and uncertainty quantification
- **Bayesian Networks**: Probabilistic reasoning with causal relationships
- **Sensor Fusion**: Multi-sensor data integration with temporal alignment
- **Fuzzy Systems**: Handle measurement uncertainty and imprecision
- **Real-time Analysis**: Streaming analysis pipelines for live performance monitoring
- **Pattern Recognition**: Automated detection of movement patterns and technique markers

#### Compiling Turbulance Code

```bash
# Compile Turbulance analysis to Python
turbulance compile sprint_analysis.tbn -o analysis_output/

# Validate Turbulance syntax
turbulance validate sprint_analysis.tbn --detailed

# Generate documentation from Turbulance code  
turbulance doc sprint_analysis.tbn --format html
```

For detailed Turbulance language documentation, see [TURBULANCE_README.md](docs/TURBULANCE_README.md).

## Examples and Results

### Sprint Analysis with Skeletal Tracking

![Sprint Analysis with Skeletal Tracking](public/results/gif/sprint_skeleton_tracking.gif)

The system automatically detects athletes, tracks their skeletons, and analyzes biomechanical parameters in real-time. This visualization shows full skeleton tracking with joint confidence visualization.

### Berlin 2009 World Record Analysis

![Berlin 2009 Maximum Velocity Phase](public/results/gif/annotated_hundred-side.gif)

This visualization captures the maximum velocity phase of Usain Bolt's iconic 9.58s 100m world record from the 2009 World Championships in Berlin. During the 60-80m segment (Bolt's fastest interval of the race at 1.61s for 20m), Bolt:
- Reached his maximum velocity of 12.34 m/s at the 67.90m mark
- Averaged a stride length of 2.77m
- Maintained a stride frequency of 4.49 strides/second

### Ground Reaction Force Estimation

![Ground Reaction Force Analysis](public/results/gif/grf_visualization.gif)

Moriarty estimates ground reaction forces without force plates, using video analysis and biomechanical modeling. The visualization shows force vectors during foot contact.

## Performance Considerations

- **Memory Usage**: Default limit is 40% of system RAM, adjustable via `memory_limit` parameter
- **Batch Size**: Larger values increase speed but require more memory
- **Worker Count**: Defaults to CPU count - 1, can be manually specified
- **GPU Acceleration**: Utilized when available for pose detection and dynamics analysis
- **Mixed Precision**: FP16 computations used when supported by the GPU

## Development

See [tasks.md](docs/tasks.md) for the current development roadmap and task status.

### Project Structure

```
moriarty/
├── docs/             # Detailed documentation
├── public/           # Example videos
├── src/              # Source code
│   ├── core/         # Core analysis modules
│   ├── distributed/  # Parallel processing tools
│   ├── solver/       # Biomechanical algorithms
│   ├── utils/        # Helper utilities
│   └── cli/          # Command line interface
├── scripts/          # Utility scripts
└── requirements.txt  # Dependencies
```

## Documentation

Detailed documentation for specific components:

- [Pipeline Documentation](docs/pipeline.md)
- [Distributed System](docs/README_PIPELINE.md)
- [Biomechanical Analysis](docs/README_Graffiti.md)
- [Orchestration System](docs/README_orchestration.md)

## License

MIT License 