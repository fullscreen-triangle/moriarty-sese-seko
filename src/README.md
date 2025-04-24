# Moriarty Source Code

This directory contains the source code for the Moriarty project, a comprehensive sports video analysis system.

## Package Structure

- **api/**: API integration with LLMs and endpoints for analysis
- **core/**: Core processing functionality for video analysis
- **distributed/**: Distributed computing tools (Ray, Dask)
- **models/**: Model definitions and training functionality
- **utils/**: Utility functions used throughout the system
- **datasets/**: Dataset handling functionality
- **config/**: Configuration settings and parameters
- **solver/**: Biomechanical solvers and analysis tools

## Main Modules

- **main.py**: Entry point for the command-line interface
- **pipeline.py**: Core video processing pipeline

## Usage

You can use the package by importing it in your Python code:

```python
from moriarty.core import pose_estimation
from moriarty.api import client
from moriarty.models import training

# Example usage
client.query_video("What technique issues are visible in the athlete's movement?")
```

Or by using the command-line interface:

```bash
# Process a single video
python -m moriarty --video path/to/video.mp4

# Process all videos in a directory
python -m moriarty --input videos_directory --train_llm
```

See the main README.md at the repository root for more detailed usage instructions. 