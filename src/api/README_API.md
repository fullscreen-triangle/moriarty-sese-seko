# VisualKinetics API-based Analysis System

This system extracts pose data from sports videos and uses powerful AI models (OpenAI GPT-4 or Anthropic Claude) to analyze the data. Instead of training small custom LLMs, this approach leverages industry-leading pre-trained models for superior analysis.

## Features

- Extract pose data from processed sports videos
- Convert pose data to text descriptions suitable for AI analysis
- Analyze pose data using OpenAI's GPT-4 or Anthropic's Claude
- Complete pipeline from video processing to AI analysis
- Save analyses as structured JSON files for further use
- API service for querying video content

## Prerequisites

- Python 3.8 or higher
- API keys for OpenAI and/or Anthropic
- Processed sports videos or raw videos to process

## Installation

1. Install the required dependencies:

```bash
pip install -r src/api/requirements_api.txt
```

2. Set up your API keys:

```bash
# Create a .env file
echo "OPENAI_API_KEY=your-openai-key" > .env
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
```

## Usage

### Complete Pipeline

To process a new video and analyze it in one step:

```bash
python src/core/scene/analyze_video.py /path/to/your/video.mp4 --api openai --sport_type basketball
```

This will:
1. Process the video to extract pose data
2. Find the latest generated pose model
3. Send the data to the AI API for analysis
4. Save the results in the `pose_analysis_results` directory

### Step-by-Step Approach

#### 1. Process a video (if needed)

```bash
python src/main.py --video /path/to/your/video.mp4
```

#### 2. Analyze a specific model file

```bash
python src/api/pose_analysis_api.py --api openai --single_model pose_model.pth --sport_type basketball
```

#### 3. Analyze all model files in a directory

```bash
python src/api/pose_analysis_api.py --api openai --model_dir models --sport_type basketball
```

### Using the API Service

#### 1. Start the API server

```bash
python src/api/api_service.py
```

#### 2. Use the client to query the API

```bash
# Query the data
python src/api/client_example.py query "What sports techniques are shown in the videos?"

# List all processed videos
python src/api/client_example.py list

# Add a new video
python src/api/client_example.py add output/annotated_new_video.mp4

# Reindex all videos
python src/api/client_example.py reindex
```

## API Options

The system supports two powerful AI APIs:

### OpenAI API (Default)
- Uses GPT-4 for deep insights into pose data
- Generally provides more detailed biomechanical analysis
- Use with: `--api openai`

### Anthropic API
- Uses Claude models for alternative analysis
- May offer different perspectives on the same data
- Use with: `--api anthropic`

## Output Format

All analyses are saved as JSON files in the output directory (default: `pose_analysis_results`). Each file contains:

- The original model file path
- Sport type (if specified)
- Raw text descriptions of the pose data
- AI analysis of the pose data

A summary file is also created with overview statistics about the analysis run.

## Advantages Over Custom LLM Training

This API-based approach offers several advantages over training custom small LLMs:

1. **Superior Quality**: State-of-the-art models like GPT-4 and Claude provide much higher quality analysis than small fine-tuned models
2. **No Training Required**: Skip the complex process of training and tuning models
3. **Immediate Results**: Get detailed analyses immediately without waiting for training
4. **Less Resource Intensive**: No need for GPUs or extensive computing resources
5. **Regular Improvements**: Benefit from ongoing improvements to the underlying models

## Advanced Usage

### RAG System Integration

The system includes a RAG (Retrieval-Augmented Generation) system that allows natural language queries against your video data:

```bash
# Start the RAG API server
python src/api/api_service.py

# Query your data
python src/api/client_example.py query "What are the key motion patterns in the video?"
```

### Custom Prompts

Edit the `pose_analysis_api.py` file to customize the prompts sent to the AI models for specialized analysis.

## License

MIT License

## Acknowledgments

- OpenAI API for GPT models
- Anthropic API for Claude models
- Based on the VisualKinetics video processing system 