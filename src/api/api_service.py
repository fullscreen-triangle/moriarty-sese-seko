import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the RAG system
from rag_system import SportsVideoRAG
from data_extractor import VideoDataExtractor

# Create the FastAPI app
app = FastAPI(
    title="Sports Video Analysis API",
    description="API for querying sports video analysis data using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models for the API
class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query to process")
    max_results: int = Field(5, description="Maximum number of results to return")
    openai_api_key: Optional[str] = Field(None, description="Optional OpenAI API key")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The response from the LLM")
    sources: List[Dict[str, Any]] = Field([], description="Sources used to generate the response")

class VideoInfo(BaseModel):
    filename: str = Field(..., description="Filename of the video")
    original_name: str = Field(..., description="Original name of the video")
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")
    fps: float = Field(..., description="Frames per second")
    total_frames: int = Field(..., description="Total number of frames")
    processed_date: str = Field(..., description="Date when the video was processed")

class AddVideoRequest(BaseModel):
    video_path: str = Field(..., description="Path to the processed video")
    sample_rate: int = Field(5, description="Sample 1 frame every N frames")
    openai_api_key: Optional[str] = Field(None, description="Optional OpenAI API key")

class AddVideoResponse(BaseModel):
    success: bool = Field(..., description="Whether the video was successfully added")
    video_name: str = Field(..., description="Name of the added video")
    message: str = Field(..., description="Status message")

# Global variable for the RAG system
# We'll initialize it on startup
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        # Get OpenAI API key from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
        
        # Initialize the RAG system
        rag_system = SportsVideoRAG(
            data_dir="data_store",
            db_dir="vector_db",
            openai_api_key=openai_api_key
        )
        
        # Create the data directories if they don't exist
        os.makedirs("data_store", exist_ok=True)
        os.makedirs("vector_db", exist_ok=True)
        
        print("RAG system initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")

def get_rag_system(api_key: str = None):
    """Helper function to get the RAG system with an optional API key."""
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Override API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    return rag_system

@app.post("/query", response_model=QueryResponse)
async def query_videos(
    request: QueryRequest,
    rag=Depends(lambda: get_rag_system(request.openai_api_key))
):
    """
    Query the video analysis data using natural language.
    This endpoint uses the RAG system to retrieve relevant information
    and generate a response using an LLM.
    """
    try:
        response = rag.query(request.query, k=request.max_results)
        
        # For now, we'll return a simplified sources list
        # In a more advanced implementation, you'd extract the actual sources
        sources = []
        
        return QueryResponse(
            response=response,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/videos", response_model=List[VideoInfo])
async def list_videos(rag=Depends(get_rag_system)):
    """
    List all processed videos in the system.
    This endpoint returns metadata about all videos that have been processed.
    """
    try:
        # Check if the catalog file exists
        catalog_file = Path("data_store") / "video_catalog.json"
        if not catalog_file.exists():
            return []
        
        # Load the catalog
        with open(catalog_file, 'r') as f:
            catalog = json.load(f)
        
        # Convert to Pydantic models
        videos = []
        for video_meta in catalog.get("videos", []):
            videos.append(VideoInfo(**video_meta))
        
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

@app.post("/videos/add", response_model=AddVideoResponse)
async def add_video(
    request: AddVideoRequest,
    background_tasks: BackgroundTasks,
    rag=Depends(lambda: get_rag_system(request.openai_api_key))
):
    """
    Add a new video to the system.
    This endpoint extracts data from a processed video and adds it to the RAG system.
    The processing happens in the background.
    """
    video_path = request.video_path
    
    # Validate the video path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
    
    if not video_path.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 videos are supported")
    
    # Extract the video name
    video_name = Path(video_path).stem
    
    # Process in the background
    def process_video_task():
        try:
            rag.add_new_video(video_path, request.sample_rate)
            print(f"Successfully added video: {video_name}")
        except Exception as e:
            print(f"Error adding video {video_name}: {e}")
    
    # Add the task to the background
    background_tasks.add_task(process_video_task)
    
    return AddVideoResponse(
        success=True,
        video_name=video_name,
        message="Video processing started in the background"
    )

@app.post("/index")
async def reindex_videos(
    background_tasks: BackgroundTasks,
    rag=Depends(get_rag_system)
):
    """
    Reindex all videos in the system.
    This endpoint triggers a reindexing of all video data in the RAG system.
    The processing happens in the background.
    """
    def reindex_task():
        try:
            rag.load_and_index_video_data(reindex=True)
            print("Reindexing completed successfully")
        except Exception as e:
            print(f"Error during reindexing: {e}")
    
    # Add the task to the background
    background_tasks.add_task(reindex_task)
    
    return {"message": "Reindexing started in the background"}

# CLI entry point
if __name__ == "__main__":
    # Check if the .env file exists, create it if not
    env_file = Path(".env")
    if not env_file.exists():
        # Get the API key from the user
        api_key = input("Enter your OpenAI API key: ")
        with open(env_file, 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print(f".env file created with API key")
    
    # Start the API server
    print("Starting Sports Video Analysis API server...")
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True) 