# VisualKinetics LLM Training System

This system allows you to extract pose data from sports videos, convert it into a format suitable for language model training, and train or fine-tune a small language model (LLM) to generate insights about the pose data.

## Features

- Extract pose data from processed sports videos
- Convert pose data to text descriptions suitable for LLM training
- Train a small language model on the pose data
- Progressive learning: update the model with new videos over time
- Complete pipeline from video processing to LLM training

## Prerequisites

- Python 3.8 or higher
- PyTorch
- CUDA-compatible GPU (recommended but not required)
- Source videos for processing

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_llm.txt
```

2. Ensure you have processed videos or the ability to process new videos through the VisualKinetics system.

## Usage

### Complete Pipeline

To process a new video and train the LLM in one step:

```bash
python train_from_new_video.py /path/to/your/video.mp4 --sport_type basketball
```

This will:
1. Process the video to extract pose data
2. Convert the pose data to training examples
3. Train or update the LLM with the new data

### Step-by-Step Approach

#### 1. Extract Pose Data from Models

```bash
# Import the extractor
from pose_data_to_llm import PoseDataExtractor

# Create an extractor instance
extractor = PoseDataExtractor(
    model_dir="models",              # Directory containing pose models
    output_dir="llm_training_data"   # Directory to save training data
)

# Process all models and generate training examples
examples = extractor.process_all_models(sport_type="basketball")
```

#### 2. Train the LLM on Pose Data

```bash
# Import the trainer
from train_llm import LLMTrainer

# Create a trainer instance
trainer = LLMTrainer(
    data_dir="llm_training_data",    # Directory containing training data
    model_dir="llm_models"           # Directory to save trained models
)

# Prepare the training data
trainer.prepare_training_data()

# Train the model
trainer.train_model(
    epochs=3,
    batch_size=4,
    learning_rate=1e-4
)

# Generate a response from the trained model
response = trainer.generate_response(
    "Describe the motion pattern in this basketball sequence:"
)
print(response)
```

## Command Line Interface

### Extract Pose Data CLI

```bash
python pose_data_to_llm.py --model_dir models --output_dir llm_training_data --sport_type basketball
```

### Train LLM CLI

```bash
python train_llm.py --data_dir llm_training_data --model_dir llm_models --base_model facebook/opt-350m --epochs 3 --batch_size 4 --test
```

## Model Options

The system supports various base models, including:

- `facebook/opt-125m` (tiny model, good for CPU)
- `facebook/opt-350m` (small model, default choice)
- `facebook/opt-1.3b` (medium model, recommended for GPU)
- `facebook/opt-2.7b` (larger model, requires a good GPU)

For CPU-only environments, the system automatically selects a smaller model.

## Progressive Learning

The system supports progressive learning, where the model is updated with new data over time:

```bash
# After processing a new video and extracting data
trainer = LLMTrainer(model_dir="llm_models")
trainer.progressive_training(
    new_data_file="llm_training_data/latest_examples.json",
    epochs=2
)
```

## Advanced Configuration

For advanced users, the system allows customization of:

- Training parameters (learning rate, weight decay, etc.)
- Model architecture (using different base models)
- LoRA parameters for efficient fine-tuning
- Quantization options for smaller model footprint

## License

MIT License

## Acknowledgments

- This system uses the HuggingFace Transformers library
- Base models from Meta AI's OPT family of language models
- PyTorch for deep learning capabilities 