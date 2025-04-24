# Motion Analysis Orchestration System

## Overview

This orchestration system integrates VisualKinetics and Graffiti packages into a unified, distributed processing pipeline for sports video analysis. It leverages message queuing for communication between components, providing a scalable, fault-tolerant architecture that maximizes the strengths of both packages.

## Architecture

### System Components

1. **Orchestrator**: Central coordination service that manages workflows and task distribution
2. **Message Queue**: RabbitMQ instance for communication between components
3. **Workers**: Specialized processing units that perform specific tasks
4. **Storage Manager**: Handles data persistence and sharing between components
5. **API Gateway**: Provides unified access to system functionality

### Information Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Input      │────▶│   Video Queue   │────▶│  VisualKinetics │
│  Videos     │     └────────────────┘     │  Video Worker   │
└─────────────┘                            └─────────┬───────┘
                                                     │
                                                     ▼
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Results &  │◀────│  Results Queue │◀────│  Visualization  │
│  Storage    │     └────────────────┘     │     Worker      │
└─────────────┘                            └─────────▲───────┘
                                                     │
┌─────────────┐     ┌────────────────┐     ┌─────────┴───────┐
│  Commands   │────▶│  Control Queue │────▶│   Orchestrator  │
│  & Config   │     └────────────────┘     └─────────────────┘
└─────────────┘                             │      ▲
                                            ▼      │
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Pose Data   │◀───▶│   Pose Queue   │◀───▶│    Graffiti     │
│  Storage    │     └────────────────┘     │  Analysis Worker│
└─────────────┘                            └─────────────────┘
```

### Queue Structure

1. **video_queue**: Input videos awaiting processing
2. **pose_queue**: Extracted pose data for biomechanical analysis
3. **analysis_queue**: Completed biomechanical analysis results
4. **results_queue**: Final processed results
5. **control_queue**: System commands and configuration updates
6. **status_queue**: Worker status updates and heartbeats
7. **error_queue**: Failed tasks and error reports

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- VisualKinetics package
- Graffiti package
- 50GB+ disk space for video storage and intermediate results

### Setup Instructions

1. Clone the orchestration repository:

```bash
git clone https://github.com/yourusername/motion-orchestrator.git
cd motion-orchestrator
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start RabbitMQ:

```bash
docker-compose up -d rabbitmq
```

4. Initialize the system:

```bash
python -m orchestrator.init
```

## Project Structure

```
/motion-orchestrator/
├── docker-compose.yml           # Container definitions
├── requirements.txt             # Python dependencies
├── config/
│   ├── settings.py              # Global configuration
│   ├── queues.py                # Queue definitions
│   └── workflows.py             # Workflow definitions
├── orchestrator/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── init.py                  # Initialization script
│   ├── api.py                   # API Gateway
│   ├── dispatcher.py            # Task dispatcher
│   └── monitor.py               # System monitoring
├── storage/
│   ├── __init__.py
│   ├── manager.py               # Storage management
│   ├── models.py                # Data models
│   └── adapters.py              # Format conversion
├── workers/
│   ├── __init__.py
│   ├── base_worker.py           # Abstract worker class
│   ├── video_worker.py          # VisualKinetics video processing
│   ├── pose_worker.py           # Pose data extraction
│   ├── biomech_worker.py        # Graffiti biomechanical analysis
│   ├── llm_worker.py            # LLM analysis integration
│   └── visualization_worker.py  # Results visualization
└── utils/
    ├── __init__.py
    ├── logger.py                # Logging utilities
    ├── metrics.py               # Performance metrics
    └── validators.py            # Input validation
```

## Message Queue Configuration

### RabbitMQ Setup

The system uses RabbitMQ for message passing. Configuration is defined in `config/queues.py`:

```python
# Queue definitions
QUEUE_DEFINITIONS = {
    'video_queue': {
        'durable': True,
        'arguments': {'x-max-priority': 10}
    },
    'pose_queue': {
        'durable': True,
        'arguments': {'x-message-ttl': 3600000}  # 1 hour TTL
    },
    # Additional queues...
}

# Exchange definitions
EXCHANGE_DEFINITIONS = {
    'video_exchange': {
        'type': 'direct',
        'durable': True
    },
    # Additional exchanges...
}
```

### Message Format

All messages use JSON format with the following structure:

```json
{
    "task_id": "unique-task-identifier",
    "type": "task_type",
    "created_at": "ISO-timestamp",
    "priority": 5,
    "payload": {
        "task-specific": "data"
    },
    "metadata": {
        "source": "component_name",
        "attempts": 0
    }
}
```

## Core Components Implementation

### 1. Orchestrator Service

The orchestrator (`orchestrator/main.py`) is responsible for:
- Initializing the system
- Registering and managing workers
- Monitoring queue health
- Dispatching tasks based on workflow definitions

Implementation example:

```python
class Orchestrator:
    def __init__(self, config_path='config/settings.py'):
        self.config = self._load_config(config_path)
        self.connection = self._establish_connection()
        self.channel = self.connection.channel()
        self.workers = {}
        self.initialize_queues()
        self.dispatcher = Dispatcher(self.channel)
        
    def _load_config(self, config_path):
        # Load configuration
        
    def _establish_connection(self):
        # Connect to RabbitMQ
        
    def initialize_queues(self):
        # Declare queues, exchanges, and bindings
        
    def register_worker(self, worker_id, worker_type):
        # Register a worker
        
    def start(self):
        # Start the orchestrator service
        # Listen on control queue
        
    def handle_command(self, command):
        # Process control commands
        
    def dispatch_workflow(self, workflow_name, input_data):
        # Trigger a workflow
```

### 2. Worker Implementation

Workers (`workers/base_worker.py`) are specialized processes that:
- Connect to specific queues
- Process messages as they arrive
- Publish results to output queues
- Report status and errors

Base worker class:

```python
class BaseWorker:
    def __init__(self, worker_id, input_queue, output_queue):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.connection = self._establish_connection()
        self.channel = self.connection.channel()
        self.storage = StorageManager()
        
    def _establish_connection(self):
        # Connect to RabbitMQ
        
    def setup(self):
        # Declare queues and set up consumer
        self.channel.queue_declare(queue=self.input_queue, durable=True)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.input_queue,
            on_message_callback=self.process_message
        )
    
    def process_message(self, ch, method, properties, body):
        # Default message processor
        try:
            data = json.loads(body)
            result = self.process_task(data)
            self.publish_result(result)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            self.handle_error(e, method.delivery_tag)
    
    def process_task(self, data):
        # To be implemented by subclasses
        raise NotImplementedError
    
    def publish_result(self, result):
        # Publish to output queue
        
    def handle_error(self, error, delivery_tag):
        # Handle and report errors
        
    def start(self):
        # Start consuming messages
        self.setup()
        self.channel.start_consuming()
```

Specific worker implementation example:

```python
class VideoProcessingWorker(BaseWorker):
    def __init__(self, worker_id):
        super().__init__(
            worker_id=worker_id,
            input_queue='video_queue',
            output_queue='pose_queue'
        )
        self.processor = VideoProcessor()
        
    def process_task(self, data):
        # Get video path from task data
        video_path = data['payload']['video_path']
        
        # Process with VisualKinetics
        result_path = self.processor.process_video(
            video_path, 
            n_workers=data['payload'].get('n_workers', 4)
        )
        
        # Extract pose data
        pose_data = self.processor.extract_pose_data(result_path)
        
        # Save to shared storage
        data_id = self.storage.save_pose_data(pose_data)
        
        # Return result for next stage
        return {
            'task_id': data['task_id'],
            'type': 'pose_data',
            'payload': {
                'data_id': data_id,
                'original_video': video_path
            }
        }
```

### 3. Storage Manager

The Storage Manager (`storage/manager.py`) provides:
- Consistent storage for intermediate results
- Data format conversions between components
- Caching for frequently accessed data

Implementation example:

```python
class StorageManager:
    def __init__(self, base_path='./data'):
        self.base_path = base_path
        self.ensure_directories()
        self.db = self._connect_db()
        
    def ensure_directories(self):
        # Create necessary directories
        
    def _connect_db(self):
        # Connect to metadata database (SQLite for local)
        
    def generate_id(self):
        # Generate unique ID for data
        
    def save_pose_data(self, pose_data, metadata=None):
        # Save pose data and return ID
        data_id = self.generate_id()
        path = f"{self.base_path}/pose_data/{data_id}.json"
        
        with open(path, 'w') as f:
            json.dump(pose_data, f)
            
        self._save_metadata(data_id, 'pose_data', path, metadata)
        return data_id
        
    def get_pose_data(self, data_id):
        # Retrieve pose data by ID
        
    def convert_format(self, data, source_format, target_format):
        # Convert between VisualKinetics and Graffiti formats
```

### 4. API Gateway

The API Gateway (`orchestrator/api.py`) provides:
- RESTful API for system interaction
- Task submission and status checking
- Result retrieval

Implementation example:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import uuid

app = FastAPI(title="Motion Analysis Orchestrator API")
dispatcher = Dispatcher()

@app.post("/videos/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded file
    video_id = str(uuid.uuid4())
    file_path = f"./data/uploads/{video_id}.mp4"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Create processing task
    task_id = dispatcher.create_task('video_processing', {
        'video_path': file_path
    })
    
    return {"task_id": task_id, "video_id": video_id}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    # Check task status
    status = dispatcher.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    # Get task results
    results = dispatcher.get_task_results(task_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    return results
```

## Workflow Configurations

Workflows (`config/workflows.py`) define processing sequences:

```python
WORKFLOWS = {
    'standard_analysis': [
        {
            'name': 'video_processing',
            'queue': 'video_queue',
            'next': 'biomechanical_analysis'
        },
        {
            'name': 'biomechanical_analysis',
            'queue': 'pose_queue',
            'next': 'visualization'
        },
        {
            'name': 'visualization',
            'queue': 'analysis_queue',
            'next': None
        }
    ],
    'llm_enhanced_analysis': [
        # Steps for analysis with LLM integration
    ]
}
```

## Worker Execution

To start a worker:

```bash
# Start a video processing worker
python -m workers.start_worker --type video --id worker1

# Start a biomechanical analysis worker
python -m workers.start_worker --type biomech --id worker2
```

## System Monitoring

The system includes a monitoring dashboard (`orchestrator/monitor.py`):

```python
class SystemMonitor:
    def __init__(self):
        self.connection = self._establish_connection()
        self.channel = self.connection.channel()
        self.queue_stats = {}
        self.worker_stats = {}
        
    def get_queue_stats(self):
        # Query RabbitMQ for queue statistics
        
    def get_worker_status(self):
        # Get worker status from status queue
        
    def generate_report(self):
        # Generate system status report
```

## Error Handling

The system uses a dedicated error queue and retry mechanism:

```python
def handle_error(channel, task, error, delivery_tag):
    # Increment retry count
    if 'attempts' not in task['metadata']:
        task['metadata']['attempts'] = 0
    task['metadata']['attempts'] += 1
    
    # Add error information
    task['metadata']['last_error'] = str(error)
    
    if task['metadata']['attempts'] < MAX_RETRY_ATTEMPTS:
        # Requeue for retry
        channel.basic_publish(
            exchange='',
            routing_key=task['metadata']['source_queue'],
            body=json.dumps(task)
        )
    else:
        # Send to error queue for manual handling
        channel.basic_publish(
            exchange='',
            routing_key='error_queue',
            body=json.dumps(task)
        )
    
    # Acknowledge the original message
    channel.basic_ack(delivery_tag=delivery_tag)
```

## Integration Points

### VisualKinetics Integration

```python
from visualkinetics import VideoProcessor

def process_with_visualkinetics(video_path, options):
    processor = VideoProcessor(n_workers=options.get('n_workers', 4))
    result = processor.process_video(video_path)
    return result
```

### Graffiti Integration

```python
from graffiti import BiomechanicalAnalyzer

def analyze_with_graffiti(pose_data, options):
    analyzer = BiomechanicalAnalyzer()
    # Convert VisualKinetics pose format to Graffiti format
    converted_data = convert_pose_format(pose_data)
    analysis_result = analyzer.analyze_motion(converted_data)
    return analysis_result
```

## Configuration Options

System configuration (`config/settings.py`):

```python
# RabbitMQ configuration
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'guest'
RABBITMQ_PASS = 'guest'
RABBITMQ_VHOST = '/'

# Storage configuration
STORAGE_BASE_PATH = './data'
TEMP_STORAGE_PATH = './data/temp'
VIDEO_STORAGE_PATH = './data/videos'
RESULTS_STORAGE_PATH = './data/results'

# Worker configuration
DEFAULT_WORKER_COUNT = {
    'video': 2,
    'biomech': 2,
    'llm': 1,
    'visualization': 1
}
MAX_RETRY_ATTEMPTS = 3

# System settings
API_HOST = '0.0.0.0'
API_PORT = 8000
LOG_LEVEL = 'INFO'
```

## Example Workflows

### Standard Video Analysis

1. Submit a video:
```bash
curl -X POST -F "file=@sports_video.mp4" http://localhost:8000/videos/
```

2. System processes:
   - Video processing worker extracts pose data
   - Biomechanical worker analyzes movement patterns
   - Visualization worker creates annotated video
   - Results stored and made available via API

### Batch Processing

```python
# Submit multiple videos for processing
import requests
import os

def process_batch(directory):
    task_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                response = requests.post(
                    'http://localhost:8000/videos/',
                    files={'file': f}
                )
                task_ids.append(response.json()['task_id'])
    return task_ids
```

## Troubleshooting

### Common Issues

1. **Worker Connection Failures**
   - Check RabbitMQ is running: `docker ps`
   - Verify credentials in settings.py
   - Check network connectivity

2. **Processing Errors**
   - Check error queue for failed tasks
   - Verify input video formats
   - Check storage permissions

3. **System Bottlenecks**
   - Monitor queue depths with RabbitMQ Management UI
   - Adjust worker counts for specific stages
   - Check disk space for storage manager

### Logs

System logs are stored in `./logs` directory:
- `orchestrator.log` - Main orchestrator logs
- `worker_<type>_<id>.log` - Individual worker logs
- `api.log` - API gateway logs

### Debugging

Enable debug mode for detailed logging:

```bash
LOG_LEVEL=DEBUG python -m orchestrator.main
```

## Scaling Considerations

### Local Scaling

For increased local performance:
- Increase worker counts for CPU-bound tasks
- Use memory mapping for large datasets
- Enable process-based parallelism

### Server Deployment

When moving to server environments:
- Containerize each component with Docker
- Use proper authentication for RabbitMQ
- Implement monitoring with Prometheus/Grafana
- Consider cloud storage instead of local filesystem

## License

MIT License 