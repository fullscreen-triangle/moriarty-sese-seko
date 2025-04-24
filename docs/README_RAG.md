# VisualKinetics RAG System

This system allows you to create an LLM-powered query interface for your processed sports videos. It uses a Retrieval-Augmented Generation (RAG) approach to make your video analysis data queryable using natural language.

## Features

- Extract pose data and metrics from processed videos
- Store video analysis data in a vector database for efficient retrieval
- Query the data using natural language via OpenAI LLMs
- Progressively add more videos to expand the knowledge base
- API for integrating with external applications

## Getting Started

### Prerequisites

- Python 3.8+
- Processed videos from VisualKinetics
- OpenAI API key

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
# Either set it in your environment
export OPENAI_API_KEY=your-api-key

# Or create a .env file
echo "OPENAI_API_KEY=your-api-key" > .env
```

### Usage

#### 1. Extract data from processed videos

First, extract data from your processed videos:

```bash
python data_extractor.py
```

This will:
- Scan the `output` directory for processed videos
- Extract frame data at regular intervals
- Save the data to JSON files in the `data_store` directory

#### 2. Start the API server

Start the API server to make your data queryable:

```bash
python api_service.py
```

The server will:
- Initialize the RAG system
- Load and index the video data
- Start an API server on port 8000

#### 3. Query your data

You can query your data using the client example:

```bash
# Query the data
python client_example.py query "What sports techniques are shown in the videos?"

# List all processed videos
python client_example.py list

# Add a new video
python client_example.py add output/annotated_new_video.mp4

# Reindex all videos
python client_example.py reindex
```

### Integration with OpenAI LLMs

To query your sports video data from an OpenAI-powered application:

1. Make a request to your API server:

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What are the key motion patterns in the judo video?",
    "max_results": 5,
    "openai_api_key": "your-openai-api-key"  # Optional
})

result = response.json()
print(result["response"])
```

2. Or use the provided client:

```python
from client_example import SportsVideoClient

client = SportsVideoClient()
result = client.query("What are the key motion patterns in the judo video?")
print(result["response"])
```

## System Architecture

The RAG system consists of the following components:

1. **Data Extractor**: Extracts pose data and metrics from processed videos
2. **RAG System**: Stores and indexes the video data in a vector database
3. **API Service**: Provides a REST API for querying the data
4. **Client**: Example client for interacting with the API

## Adding More Videos

To add more videos to the system:

1. Process the videos using VisualKinetics
2. Add them to the RAG system using the API:

```bash
python client_example.py add output/annotated_new_video.mp4
```

The system will:
- Extract data from the new video
- Add it to the vector database
- Make it available for querying

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of VisualKinetics
- Uses LangChain and ChromaDB for RAG capabilities
- Uses OpenAI models for natural language understanding 