# Utils Documentation

This directory contains utility modules that support the core functionality of the project.

## Files Overview

### `simpleprocessor.py`
A video processing utility that uses MediaPipe for pose estimation and analysis.

Key components:
- `SimpleVideoProcessor` class: Processes videos with MediaPipe's pose estimation
  - Detects human pose in video frames
  - Calculates metrics like joint angles, velocity, and distances
  - Annotates videos with pose landmarks and metrics
  - Saves processed videos with overlaid information

### `rag_system.py`
A Retrieval-Augmented Generation (RAG) system for sports video analysis.

Key components:
- `SportsVideoRAG` class: Creates and maintains a vector database of video analysis data
  - Uses ChromaDB for efficient vector storage and retrieval
  - Integrates with OpenAI API for embeddings and LLM responses
  - Indexes video metadata, frame data, and pose analysis
  - Provides natural language query capability for video content
  - Supports adding new videos to the database

### `structure_visualization_data.py`
Utility for structuring AI-generated benchmark data for frontend visualization.

Key components:
- `VisualizationDataStructured` class: Converts AI responses into structured data
  - Extracts frame-by-frame data using regex patterns
  - Identifies movement phases from descriptions
  - Extracts biomechanical insights including improvements and risks
  - Processes benchmark files into JSON format for visualization

### `train_llm_cli.py`
Command-line interface for training or updating language models on pose data.

Features:
- Configuration options for model selection, training parameters, and optimization
- Device detection for CPU/GPU training
- Support for progressive training to update existing models
- Model quantization options for efficient deployment
- Testing capabilities to validate trained models

### `pose_data_to_llm_cli.py`
Command-line interface for extracting pose data from models and converting to LLM training data.

Features:
- Options for specifying model and output directories
- Sport type selection for specialized prompt generation
- Single model or batch processing options
- Integration with the PoseDataExtractor class

### `train_llm.py`
Comprehensive utility for training language models on pose analysis data.

Key components:
- `LLMTrainer` class: Handles the training pipeline
  - Supports progressive training as new pose models are added
  - Implements parameter-efficient fine-tuning with LoRA
  - Provides 4-bit quantization for efficient training
  - Formats data for instruction-tuning
  - Includes utilities for model saving, loading, and response generation

### `__init__.py`
Empty initialization file marking this directory as a Python package.
