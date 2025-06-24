# Moriarty Pipeline Documentation

This document provides a detailed breakdown of the key pipelines within the Moriarty system, including function calls, inputs/outputs, and execution flow.

## Table of Contents
1. [Sprint Running Video Analysis Pipeline](#sprint-running-video-analysis-pipeline)
2. [Domain Expert LLM Training Pipeline](#domain-expert-llm-training-pipeline)

## Sprint Running Video Analysis Pipeline

This pipeline processes sprint running videos to extract and analyze pose, dynamics, and motion characteristics. The process runs through multiple stages, leveraging distributed computing for efficiency.

### Overview Diagram

```
Input Video → Frame Extraction → Pose Detection → Motion Tracking → Biomechanical Analysis → Results Storage
```

### Stage 1: Video Preprocessing

**Functions:**
- `src.utils.video_utils.validate_video(video_path)` - Validates video format and readability
- `src.distributed.memory_monitor.MemoryMonitor.initialize(memory_limit=0.4)` - Sets up memory monitoring
- `src.pipeline.VideoPipeline.preprocess_video(video_path, output_dir)` - Prepares video for processing

**Inputs:**
- `video_path` (str): Path to the sprint running video
- `output_dir` (str): Directory to store processed results
- `memory_limit` (float): Fraction of system memory to use (default: 0.4)

**Outputs:**
- Validated video metadata (dimensions, frame count, fps)
- Initialized memory monitor
- Preprocessing status (boolean)

**Example:**
```python
from src.pipeline import VideoPipeline
from src.distributed.memory_monitor import MemoryMonitor

# Initialize pipeline with memory monitoring
memory_monitor = MemoryMonitor.initialize(memory_limit=0.4)
pipeline = VideoPipeline(memory_monitor=memory_monitor)

# Preprocess the sprint video
video_path = "public/sprint_100m.mp4"
output_dir = "output/"
status = pipeline.preprocess_video(video_path, output_dir)
```

### Stage 2: Frame Extraction and Batching

**Functions:**
- `src.core.pose.frame_extractor.extract_frames(video_path, batch_size=5)` - Extracts frames into batches
- `src.distributed.dask_manager.initialize_client(n_workers=None)` - Sets up Dask distributed client
- `src.distributed.task_manager.distribute_batches(frame_batches, client)` - Distributes frame batches to workers

**Inputs:**
- `video_path` (str): Path to the preprocessed video
- `batch_size` (int): Number of frames per batch (default: 5)
- `n_workers` (int, optional): Number of Dask workers (default: auto)

**Outputs:**
- List of frame batch objects ready for processing
- Initialized Dask client
- Distributed task futures

**Example:**
```python
from src.core.pose.frame_extractor import extract_frames
from src.distributed.dask_manager import initialize_client
from src.distributed.task_manager import distribute_batches

# Extract frames into batches
frame_batches = extract_frames(video_path, batch_size=5)

# Initialize distributed processing
client = initialize_client(n_workers=4)  # Adjust based on system capabilities
futures = distribute_batches(frame_batches, client)
```

### Stage 3: Pose Detection and Tracking

**Functions:**
- `src.core.pose.mediapipe_processor.process_batch(batch, tracking=True)` - Detects pose in frame batch
- `src.core.pose.tracker.PoseTracker.track_poses(pose_detections)` - Maintains identity across frames
- `src.utils.data_conversion.serialize_pose_data(pose_results)` - Converts pose data to serializable format

**Inputs:**
- `batch`: Frame batch object
- `tracking` (bool): Whether to enable pose tracking (default: True)
- `pose_detections`: Raw pose detection results

**Outputs:**
- Pose landmarks for each frame
- Tracked pose identities across frames
- Serialized pose data ready for analysis

**Example:**
```python
from src.core.pose.mediapipe_processor import process_batch
from src.core.pose.tracker import PoseTracker
from src.utils.data_conversion import serialize_pose_data

# Process each batch (typically happens in Dask workers)
def process_frame_batch(batch):
    # Detect poses in frames
    pose_results = process_batch(batch, tracking=True)
    
    # Track poses across frames
    tracker = PoseTracker()
    tracked_poses = tracker.track_poses(pose_results)
    
    # Convert to serializable format for Ray
    return serialize_pose_data(tracked_poses)

# Results are collected from futures
pose_data_batches = [future.result() for future in futures]
```

### Stage 4: Biomechanical Analysis - Sprint Specific

**Functions:**
- `src.solver.sprint_analyzer.analyze_sprint_mechanics(pose_data)` - Analyzes sprint-specific biomechanics
- `src.solver.posture_solver.PostureSolver.analyze(pose_data, sport_type="sprint")` - General posture analysis
- `src.solver.velocity_analyzer.calculate_velocities(pose_data, fps)` - Calculates joint velocities
- `src.solver.acceleration_analyzer.calculate_accelerations(velocity_data, fps)` - Calculates accelerations
- `src.solver.gait_analyzer.analyze_gait_cycle(pose_data, fps)` - Analyzes running gait cycle
- `src.solver.ground_contact.detect_ground_contacts(pose_data)` - Detects foot strikes and toe-offs

**Inputs:**
- `pose_data`: Serialized pose landmark data
- `sport_type` (str): Sport type for specific analysis ("sprint")
- `fps` (float): Video frames per second
- `velocity_data`: Velocity data from previous calculations

**Outputs:**
- Sprint mechanics analysis (stride length, frequency, etc.)
- Posture quality assessments
- Joint velocities and accelerations
- Gait cycle analysis (stance/swing phases)
- Ground contact events and durations

**Example:**
```python
from src.solver.sprint_analyzer import analyze_sprint_mechanics
from src.solver.posture_solver import PostureSolver
from src.solver.velocity_analyzer import calculate_velocities
from src.solver.acceleration_analyzer import calculate_accelerations
from src.solver.gait_analyzer import analyze_gait_cycle
from src.solver.ground_contact import detect_ground_contacts

# Combine pose data from all batches
combined_pose_data = pipeline.combine_pose_batches(pose_data_batches)

# Get video metadata
video_metadata = pipeline.get_video_metadata(video_path)
fps = video_metadata["fps"]

# Run sprint-specific analyses
sprint_mechanics = analyze_sprint_mechanics(combined_pose_data)
posture_solver = PostureSolver()
posture_analysis = posture_solver.analyze(combined_pose_data, sport_type="sprint")
velocity_data = calculate_velocities(combined_pose_data, fps)
acceleration_data = calculate_accelerations(velocity_data, fps)
gait_analysis = analyze_gait_cycle(combined_pose_data, fps)
ground_contacts = detect_ground_contacts(combined_pose_data)

# Combine all analyses
analysis_results = {
    "sprint_mechanics": sprint_mechanics,
    "posture_analysis": posture_analysis,
    "velocity_data": velocity_data,
    "acceleration_data": acceleration_data,
    "gait_analysis": gait_analysis,
    "ground_contacts": ground_contacts
}
```

### Stage 5: Advanced Motion Analysis

**Functions:**
- `src.solver.joint_kinematics.calculate_joint_angles(pose_data)` - Calculates key joint angles
- `src.solver.asymmetry_detector.detect_asymmetries(pose_data)` - Detects left/right asymmetries
- `src.solver.efficiency_analyzer.analyze_sprint_efficiency(pose_data, mechanics_data)` - Analyzes movement efficiency
- `src.solver.fatigue_detector.detect_fatigue_indicators(pose_data, time_series=True)` - Detects signs of fatigue
- `src.solver.phase_detector.detect_sprint_phases(pose_data, ground_contacts)` - Identifies sprint phases

**Inputs:**
- `pose_data`: Combined pose data
- `mechanics_data`: Sprint mechanics results from previous stage
- `ground_contacts`: Ground contact data from previous stage

**Outputs:**
- Joint angle measurements throughout sprint
- Asymmetry scores between left and right sides
- Running efficiency metrics
- Fatigue indicators over time
- Sprint phase classification (acceleration, maximum velocity, deceleration)

**Example:**
```python
from src.solver.joint_kinematics import calculate_joint_angles
from src.solver.asymmetry_detector import detect_asymmetries
from src.solver.efficiency_analyzer import analyze_sprint_efficiency
from src.solver.fatigue_detector import detect_fatigue_indicators
from src.solver.phase_detector import detect_sprint_phases

# Run advanced motion analyses
joint_angles = calculate_joint_angles(combined_pose_data)
asymmetry_scores = detect_asymmetries(combined_pose_data)
efficiency_metrics = analyze_sprint_efficiency(combined_pose_data, sprint_mechanics)
fatigue_indicators = detect_fatigue_indicators(combined_pose_data, time_series=True)
sprint_phases = detect_sprint_phases(combined_pose_data, ground_contacts)

# Add to analysis results
analysis_results.update({
    "joint_angles": joint_angles,
    "asymmetry_scores": asymmetry_scores,
    "efficiency_metrics": efficiency_metrics,
    "fatigue_indicators": fatigue_indicators,
    "sprint_phases": sprint_phases
})
```

### Stage 6: Results Processing and Storage

**Functions:**
- `src.utils.result_formatter.format_analysis_results(analysis_results)` - Formats results for storage/display
- `src.utils.visualization.create_annotated_video(video_path, pose_data, analysis_results, output_path)` - Creates visualization
- `src.utils.data_exporter.export_to_json(analysis_results, output_path)` - Exports results to JSON
- `src.utils.report_generator.generate_sprint_report(analysis_results, output_path)` - Generates detailed report

**Inputs:**
- `analysis_results`: Combined analysis data from previous stages
- `video_path`: Original video path
- `pose_data`: Combined pose data
- `output_path`: Path for results storage

**Outputs:**
- Formatted analysis results
- Annotated video with overlaid metrics
- JSON file with complete analysis data
- PDF or HTML report with visualizations and metrics

**Example:**
```python
from src.utils.result_formatter import format_analysis_results
from src.utils.visualization import create_annotated_video
from src.utils.data_exporter import export_to_json
from src.utils.report_generator import generate_sprint_report

# Format results
formatted_results = format_analysis_results(analysis_results)

# Create visualizations and exports
annotated_video_path = os.path.join(output_dir, "annotated_sprint.mp4")
create_annotated_video(video_path, combined_pose_data, analysis_results, annotated_video_path)

json_path = os.path.join(output_dir, "sprint_analysis.json")
export_to_json(analysis_results, json_path)

report_path = os.path.join(output_dir, "sprint_report.html")
generate_sprint_report(analysis_results, report_path)

print(f"Analysis complete. Results saved to {output_dir}")
```

### Stage 7: LLM-Ready Data Preparation

**Functions:**
- `src.models.data_converter.convert_analysis_to_training_examples(analysis_results)` - Converts analyses to LLM training examples
- `src.models.data_storage.store_training_examples(training_examples, dataset_path)` - Stores examples for future LLM training

**Inputs:**
- `analysis_results`: Complete analysis results from previous stages
- `dataset_path`: Path to store training examples

**Outputs:**
- LLM-ready training examples
- Stored training data for future LLM use

**Example:**
```python
from src.models.data_converter import convert_analysis_to_training_examples
from src.models.data_storage import store_training_examples

# Convert analysis to LLM training data
training_examples = convert_analysis_to_training_examples(analysis_results)

# Store for future LLM training
llm_data_path = os.path.join(output_dir, "llm_training_data")
store_training_examples(training_examples, llm_data_path)

print(f"Generated {len(training_examples)} training examples for LLM")
```

### Complete Pipeline Execution

```python
from src.pipeline import VideoPipeline

# Create pipeline instance
pipeline = VideoPipeline(
    memory_limit=0.4,
    batch_size=5,
    n_workers=4
)

# Execute the complete sprint analysis pipeline
result = pipeline.process_video(
    video_path="public/sprint_100m.mp4",
    sport_type="sprint",
    output_dir="output/",
    create_visualizations=True,
    generate_report=True,
    prepare_llm_data=True
)

print(f"Analysis complete: {result['success']}")
print(f"Processed {result['frame_count']} frames in {result['processing_time']:.2f} seconds")
print(f"Results saved to: {result['output_paths']}")
```

## Domain Expert LLM Training Pipeline

This pipeline trains a domain expert LLM on sports biomechanics data from multiple datasets (CUD, MAXPLANCK, NOMO), designed to run as a background process over extended periods (weeks).

### Overview Diagram

```
Data Collection → Preprocessing → Training Data Generation → Model Setup → Training Loop → Evaluation → Deployment
```

### Stage 1: Data Collection and Integration

**Functions:**
- `src.datasets.cud_loader.load_cud_dataset(cud_path)` - Loads CUD dataset
- `src.datasets.maxplanck_loader.load_maxplanck_dataset(maxplanck_path)` - Loads MAXPLANCK dataset
- `src.datasets.nomo_loader.load_nomo_dataset(nomo_path)` - Loads NOMO dataset
- `src.datasets.dataset_merger.merge_datasets(datasets)` - Merges multiple datasets

**Inputs:**
- `cud_path` (str): Path to CUD dataset
- `maxplanck_path` (str): Path to MAXPLANCK dataset
- `nomo_path` (str): Path to NOMO dataset

**Outputs:**
- Loaded dataset objects for each source
- Merged dataset with standardized format

**Example:**
```python
from src.datasets.cud_loader import load_cud_dataset
from src.datasets.maxplanck_loader import load_maxplanck_dataset
from src.datasets.nomo_loader import load_nomo_dataset
from src.datasets.dataset_merger import merge_datasets

# Load individual datasets
cud_dataset = load_cud_dataset("datasets/CUD/")
maxplanck_dataset = load_maxplanck_dataset("datasets/MAXPLANCK/")
nomo_dataset = load_nomo_dataset("datasets/NOMO/")

# Merge datasets
datasets = [cud_dataset, maxplanck_dataset, nomo_dataset]
merged_dataset = merge_datasets(datasets)

print(f"Combined dataset size: {len(merged_dataset)} examples")
```

### Stage 2: Dataset Processing and Splitting

**Functions:**
- `src.datasets.data_processor.process_biomechanics_data(merged_dataset)` - Processes raw data
- `src.utils.data_splitter.split_dataset(processed_dataset, split_ratios=[0.7, 0.15, 0.15])` - Splits data into train/val/test
- `src.datasets.data_augmentation.augment_training_data(train_dataset)` - Augments training data

**Inputs:**
- `merged_dataset`: Combined dataset from previous stage
- `split_ratios` (list): Ratios for train/validation/test splits

**Outputs:**
- Processed dataset with standardized features
- Split datasets (train, validation, test)
- Augmented training dataset

**Example:**
```python
from src.datasets.data_processor import process_biomechanics_data
from src.utils.data_splitter import split_dataset
from src.datasets.data_augmentation import augment_training_data

# Process the merged dataset
processed_dataset = process_biomechanics_data(merged_dataset)

# Split into train, validation, test
train_dataset, val_dataset, test_dataset = split_dataset(
    processed_dataset, 
    split_ratios=[0.7, 0.15, 0.15]
)

# Augment training data
augmented_train_dataset = augment_training_data(train_dataset)

print(f"Training set: {len(augmented_train_dataset)} examples")
print(f"Validation set: {len(val_dataset)} examples")
print(f"Test set: {len(test_dataset)} examples")
```

### Stage 3: Training Data Generation

**Functions:**
- `src.solver.posture_solver.PostureSolver.generate_distillation_trio(biomechanics_data)` - Generates question/context/answer trios
- `src.models.data_formatter.format_as_instruction_tuning(distillation_trios)` - Formats for instruction tuning
- `src.models.tokenization.tokenize_datasets(formatted_datasets, tokenizer)` - Tokenizes the datasets

**Inputs:**
- `biomechanics_data`: Processed biomechanics data
- `distillation_trios`: Generated Q/C/A trios
- `formatted_datasets`: Datasets ready for tokenization
- `tokenizer`: Tokenizer for the target model

**Outputs:**
- Knowledge distillation trios (question/context/answer)
- Instruction-tuned format datasets
- Tokenized datasets ready for training

**Example:**
```python
from src.solver.posture_solver import PostureSolver
from src.models.data_formatter import format_as_instruction_tuning
from src.models.tokenization import tokenize_datasets
from transformers import AutoTokenizer

# Generate knowledge distillation trios
solver = PostureSolver()
distillation_trios = []

# Process in manageable chunks to avoid memory issues
for chunk in data_chunker(augmented_train_dataset, chunk_size=1000):
    chunk_trios = solver.generate_distillation_trio(chunk)
    distillation_trios.extend(chunk_trios)

# Format for instruction tuning
formatted_train = format_as_instruction_tuning(distillation_trios)
formatted_val = format_as_instruction_tuning(
    solver.generate_distillation_trio(val_dataset)
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Tokenize datasets
tokenized_train = tokenize_datasets(formatted_train, tokenizer)
tokenized_val = tokenize_datasets(formatted_val, tokenizer)
```

### Stage 4: Long-Running Training Setup

**Functions:**
- `src.models.llm_trainer.LLMTrainer.setup_training(model_name, tokenized_datasets, training_args)` - Sets up training
- `src.distributed.checkpoint_manager.setup_checkpointing(output_dir, checkpoint_interval)` - Configures checkpointing
- `src.utils.gpu_manager.setup_gpu_memory_management()` - Configures GPU memory handling

**Inputs:**
- `model_name` (str): Base model to fine-tune
- `tokenized_datasets`: Tokenized training and validation data
- `training_args`: Configuration for training parameters
- `output_dir` (str): Directory for checkpoints and outputs
- `checkpoint_interval` (int): How often to save checkpoints (steps)

**Outputs:**
- Configured training environment
- Checkpoint management system
- GPU memory handling setup

**Example:**
```python
from src.models.llm_trainer import LLMTrainer
from src.distributed.checkpoint_manager import setup_checkpointing
from src.utils.gpu_manager import setup_gpu_memory_management
from transformers import TrainingArguments

# Configure training arguments for long-running process
training_args = TrainingArguments(
    output_dir="models/biomechanics_llm",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    fp16=True,
    optim="adamw_torch",
    report_to="tensorboard"
)

# Setup checkpointing for long-running process
setup_checkpointing(
    output_dir="models/biomechanics_llm",
    checkpoint_interval=500  # Save every 500 steps
)

# Configure GPU memory management
setup_gpu_memory_management()

# Initialize trainer
trainer = LLMTrainer()
trainer.setup_training(
    model_name="facebook/opt-1.3b",
    tokenized_datasets={
        "train": tokenized_train,
        "validation": tokenized_val
    },
    training_args=training_args
)
```

### Stage 5: Background Training Process

**Functions:**
- `src.models.llm_trainer.LLMTrainer.start_background_training(resume_from_checkpoint=None)` - Starts background training
- `src.utils.process_manager.detach_process()` - Detaches process to run in background
- `src.utils.logging_manager.setup_training_logs(log_dir)` - Sets up detailed logging

**Inputs:**
- `resume_from_checkpoint` (str, optional): Path to checkpoint to resume from
- `log_dir` (str): Directory for training logs

**Outputs:**
- Background training process
- Logging system for monitoring
- Process ID for management

**Example:**
```python
from src.utils.process_manager import detach_process
from src.utils.logging_manager import setup_training_logs
import os

# Setup logging for long-running process
log_dir = "logs/biomechanics_llm_training"
setup_training_logs(log_dir)

# Option to resume from checkpoint
resume_checkpoint = None  # Set to checkpoint path if resuming

# Start training in background process
if __name__ == "__main__":
    # Create detached process for long-running training
    pid = detach_process()
    
    if pid == 0:  # Child process
        # This runs in the background
        print(f"Starting background training process (PID: {os.getpid()})")
        trainer.start_background_training(resume_from_checkpoint=resume_checkpoint)
    else:  # Parent process
        print(f"Background training started with PID: {pid}")
        print(f"Monitor progress with: tensorboard --logdir={log_dir}")
        print(f"Training logs available at: {log_dir}")
```

### Stage 6: Monitoring and Checkpointing

**Functions:**
- `src.utils.training_monitor.create_monitoring_dashboard(log_dir, port=8888)` - Creates monitoring dashboard
- `src.utils.checkpoint_validator.validate_checkpoints(checkpoint_dir)` - Validates saved checkpoints
- `src.models.model_evaluator.evaluate_latest_checkpoint(checkpoint_dir, test_dataset)` - Evaluates model quality

**Inputs:**
- `log_dir` (str): Directory containing training logs
- `port` (int): Port for dashboard server
- `checkpoint_dir` (str): Directory containing model checkpoints
- `test_dataset`: Test dataset for evaluation

**Outputs:**
- Monitoring dashboard
- Checkpoint validation results
- Evaluation metrics for latest checkpoint

**Example:**
```python
from src.utils.training_monitor import create_monitoring_dashboard
from src.utils.checkpoint_validator import validate_checkpoints
from src.models.model_evaluator import evaluate_latest_checkpoint

# Create monitoring dashboard
dashboard_url = create_monitoring_dashboard(log_dir, port=8888)
print(f"Training monitoring dashboard available at: {dashboard_url}")

# Function to periodically check training progress
def check_training_progress():
    # Validate recent checkpoints
    valid_checkpoints = validate_checkpoints("models/biomechanics_llm")
    
    if valid_checkpoints:
        latest_checkpoint = valid_checkpoints[-1]
        
        # Evaluate latest checkpoint
        metrics = evaluate_latest_checkpoint(latest_checkpoint, tokenize_datasets(test_dataset, tokenizer))
        
        print(f"Latest checkpoint: {latest_checkpoint}")
        print(f"Current metrics: {metrics}")
    else:
        print("No valid checkpoints found yet.")

# This can be scheduled to run periodically
```

### Stage 7: Resumable Training Management

**Functions:**
- `src.models.checkpoint_manager.find_latest_checkpoint(checkpoint_dir)` - Finds latest valid checkpoint
- `src.models.llm_trainer.LLMTrainer.resume_training(checkpoint_path)` - Resumes training from checkpoint
- `src.utils.training_scheduler.schedule_training_jobs(total_epochs, epochs_per_job)` - Schedules training in chunks

**Inputs:**
- `checkpoint_dir` (str): Directory containing checkpoints
- `checkpoint_path` (str): Path to specific checkpoint
- `total_epochs` (int): Total number of epochs to train
- `epochs_per_job` (int): Epochs to run per job

**Outputs:**
- Path to latest checkpoint
- Resumed training process
- Scheduled training jobs

**Example:**
```python
from src.models.checkpoint_manager import find_latest_checkpoint
from src.utils.training_scheduler import schedule_training_jobs

# Find latest checkpoint if training was interrupted
latest_checkpoint = find_latest_checkpoint("models/biomechanics_llm")

# Schedule long-running training in manageable chunks
# This allows for easier recovery from failures
schedule_training_jobs(
    total_epochs=20,
    epochs_per_job=2,
    checkpoint_dir="models/biomechanics_llm",
    script_path="src/models/run_training_job.py",
    log_dir="logs/biomechanics_llm_training"
)

print("Long-running training has been scheduled in background")
print("Training will continue for approximately 2-3 weeks")
print("Progress can be monitored via the dashboard and log files")
```

### Complete LLM Training Pipeline Execution

```python
from src.models.training_pipeline import DomainExpertTrainingPipeline

# Initialize the training pipeline
training_pipeline = DomainExpertTrainingPipeline(
    datasets={
        "cud": "datasets/CUD/",
        "maxplanck": "datasets/MAXPLANCK/",
        "nomo": "datasets/NOMO/"
    },
    base_model="facebook/opt-1.3b",
    output_dir="models/biomechanics_llm",
    log_dir="logs/biomechanics_llm_training",
    total_epochs=20,
    background=True,
    resume_if_exists=True
)

# Start the long-running background process
job_id = training_pipeline.start()

print(f"Domain expert LLM training started with job ID: {job_id}")
print("Training is configured to run for multiple weeks as a background process")
print(f"Monitor progress at: http://localhost:8888 or through log files in {training_pipeline.log_dir}")
```

## Pipeline Integration

Both pipelines can be integrated, allowing video analysis results to continuously enhance the LLM training:

```python
from src.pipeline import VideoPipeline
from src.models.training_pipeline import DomainExpertTrainingPipeline

# Initialize both pipelines
video_pipeline = VideoPipeline(memory_limit=0.4, batch_size=5)
training_pipeline = DomainExpertTrainingPipeline(
    datasets={
        "cud": "datasets/CUD/",
        "maxplanck": "datasets/MAXPLANCK/",
        "nomo": "datasets/NOMO/"
    },
    base_model="facebook/opt-1.3b",
    output_dir="models/biomechanics_llm",
    background=True
)

# Process new sprint video
analysis_result = video_pipeline.process_video(
    video_path="public/sprint_100m.mp4",
    sport_type="sprint",
    output_dir="output/",
    prepare_llm_data=True
)

# Add new analysis to training data
if analysis_result["success"]:
    training_pipeline.add_training_data(analysis_result["llm_data_path"])
    print(f"Added new training data from {analysis_result['video_path']}")

# Check training status
status = training_pipeline.get_status()
print(f"Training status: {status['state']}")
print(f"Current epoch: {status['current_epoch']}/{status['total_epochs']}")
print(f"Estimated completion: {status['estimated_completion']}")
```
