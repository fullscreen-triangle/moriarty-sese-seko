# Distributed Video Processing Pipeline

This pipeline processes videos using distributed computing to extract pose data, generate annotated videos, and train LLMs on the extracted data. It uses a combination of:

- **Ray** for distributed analysis of pose data
- **Dask** for parallel video frame processing 
- **MediaPipe** for pose estimation
- **Memory monitoring** to prevent crashes by limiting memory usage to 40% of system RAM
- **LLM integration** with OpenAI and Claude for synthetic data generation

## Features

- Process videos with controlled memory usage (default: 40% of system RAM)
- Generate two output formats:
  - Annotated videos with pose landmarks and metrics
  - Pose model data suitable for training
- Train LLMs on pose data or generate synthetic data with OpenAI/Claude APIs
- Fully distributed solution optimized for personal computers

## Prerequisites

Ensure you have Python 3.8+ installed, then install the required packages:

```bash
pip install -r requirements.txt
```

## Basic Usage

Process a single video:

```bash
python pipeline.py --video public/your_video.mp4
```

Process all videos in a folder:

```bash
python pipeline.py --input public
```

Generate annotated videos and train an LLM:

```bash
python pipeline.py --input public --train_llm
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input folder containing videos (default: public) |
| `--video`, `-v` | Process a single video file |
| `--output`, `-o` | Output folder for processed videos (default: output) |
| `--models`, `-m` | Folder for pose models (default: models) |
| `--llm_data`, `-l` | Folder for LLM training data (default: llm_training_data) |
| `--llm_models` | Folder for trained LLM models (default: llm_models) |
| `--no_video` | Skip generating annotated videos (models only) |
| `--train_llm` | Train LLM using the generated pose models |
| `--memory_limit` | Memory limit as fraction of total (default: 0.4) |
| `--workers` | Number of worker processes (default: auto) |
| `--batch_size` | Frames to process per batch (default: 5) |
| `--sport_type` | Type of sport for context (e.g., basketball, soccer) |
| `--use_openai` | Use OpenAI API for synthetic data generation |
| `--use_claude` | Use Claude API for synthetic data generation |

## Advanced Usage

### Memory Control

Limit memory usage to 30% of system RAM:

```bash
python pipeline.py --memory_limit 0.3
```

### Worker Control

Override automatic worker detection:

```bash
python pipeline.py --workers 4
```

### API Integration

Use OpenAI or Claude to generate synthetic training data:

```bash
python pipeline.py --train_llm --use_openai
```

or 

```bash
python pipeline.py --train_llm --use_claude
```

Ensure you have added your API keys to a `.env` file:

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Architecture Overview

The pipeline consists of several key components:

1. **VideoPipeline** - Main class orchestrating the entire process
2. **MediapipeProcessor** - Handles pose estimation (outside of Ray)
3. **MemoryMonitor** - Controls resource usage
4. **Dask Distributed** - Handles frame batch processing
5. **Ray Distributed** - Handles pose analysis in parallel
6. **PoseDataExtractor** - Converts pose data to LLM training examples
7. **LLMTrainer** - Trains models or generates synthetic data

## Processing Flow

1. Video is loaded and split into batches
2. Batches are processed with MediaPipe via Dask
3. Pose landmarks are analyzed with Ray
4. Results are combined and saved as:
   - Annotated video
   - Pose model JSON
5. Pose data is converted to training examples
6. LLMs are trained or used to generate synthetic data

## Notes on Serialization

MediaPipe results cannot be directly serialized in Ray. The pipeline handles this by:

1. Performing MediaPipe processing outside of Ray in Dask workers
2. Converting landmarks to numpy arrays before sending to Ray
3. Only sending serializable data between distributed components

This ensures efficient operation without serialization errors.

## Troubleshooting

If you encounter memory issues:
- Reduce `--memory_limit` (e.g., to 0.3 or 0.2)
- Reduce `--batch_size` (e.g., to 3 or 2)
- Reduce `--workers` to limit CPU usage

If videos fail to process correctly:
- Check that input videos are valid mp4 files
- Ensure sufficient disk space for output
- Check logs for specific error messages 