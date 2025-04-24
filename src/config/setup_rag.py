#!/usr/bin/env python
import os
import argparse
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import requests
        import numpy as np
        import cv2
        import mediapipe as mp
        import chromadb
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        print("✅ All required dependencies are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install all required dependencies:")
        print("pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found.")
        api_key = input("Please enter your OpenAI API key: ")
        if api_key:
            # Save to .env file
            with open(".env", "w") as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ API key saved to .env file.")
            return True
        else:
            print("❌ No API key provided.")
            return False
    else:
        print("✅ OpenAI API key found in environment.")
        return True

def check_processed_videos():
    """Check if there are processed videos in the output directory."""
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ Output directory not found.")
        return False
    
    videos = list(output_dir.glob("annotated_*.mp4"))
    if not videos:
        print("❌ No processed videos found in output directory.")
        return False
    
    print(f"✅ Found {len(videos)} processed videos:")
    for video in videos:
        print(f"  - {video.name}")
    
    return True

def extract_data():
    """Extract data from processed videos."""
    print("\n📊 Extracting data from processed videos...")
    
    # Create data store directory if it doesn't exist
    data_dir = Path("data_store")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Run the data extractor
        from data_extractor import VideoDataExtractor
        extractor = VideoDataExtractor()
        data = extractor.extract_all_videos(sample_rate=10)
        print(f"✅ Extracted data from {len(data)} videos")
        return True
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return False

def setup_vector_db():
    """Set up and index the vector database."""
    print("\n🔍 Setting up vector database...")
    
    try:
        # Initialize the RAG system and index the data
        from rag_system import SportsVideoRAG
        rag = SportsVideoRAG()
        count = rag.load_and_index_video_data()
        print(f"✅ Indexed {count} documents in the vector database")
        return True
    except Exception as e:
        print(f"❌ Error setting up vector database: {e}")
        return False

def start_api_server():
    """Start the API server."""
    print("\n🚀 Starting API server...")
    
    try:
        # Start the API server as a subprocess
        api_process = subprocess.Popen(
            ["python", "api_service.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        time.sleep(3)
        
        # Check if the process is still running
        if api_process.poll() is None:
            print("✅ API server started successfully")
            return api_process
        else:
            stdout, stderr = api_process.communicate()
            print(f"❌ API server failed to start")
            print(f"Error: {stderr}")
            return None
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def test_query(query="What sports are shown in the videos?"):
    """Test querying the RAG system."""
    print(f"\n❓ Testing query: '{query}'")
    
    try:
        # Import the client and query the system
        from client_example import SportsVideoClient
        client = SportsVideoClient()
        result = client.query(query)
        
        if result and "response" in result:
            print("\n✅ Query successful!")
            print("\nResponse:")
            print(result["response"])
            return True
        else:
            print("❌ Query failed")
            return False
    except Exception as e:
        print(f"❌ Error querying system: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up and run the VisualKinetics RAG system")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and API key checks")
    parser.add_argument("--extract-only", action="store_true", help="Only extract data, don't set up the database or start the server")
    parser.add_argument("--setup-only", action="store_true", help="Only set up the database, don't start the server")
    parser.add_argument("--test-query", type=str, default=None, help="Test query to run")
    
    args = parser.parse_args()
    
    print("🏁 Setting up VisualKinetics RAG System...")
    
    # Run checks
    if not args.skip_checks:
        if not check_dependencies():
            return
        
        if not check_api_key():
            return
        
        if not check_processed_videos():
            print("Please process some videos first using the visualkinetics package.")
            return
    
    # Extract data
    if not extract_data():
        return
    
    if args.extract_only:
        print("\n✅ Data extraction completed. Exiting.")
        return
    
    # Set up vector database
    if not setup_vector_db():
        return
    
    if args.setup_only:
        print("\n✅ Vector database setup completed. Exiting.")
        return
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        return
    
    # Test query
    if args.test_query:
        test_query(args.test_query)
    else:
        test_query()
    
    print("\n✅ VisualKinetics RAG System is ready!")
    print("To query the system, use the client example:")
    print("python client_example.py query 'Your question about the videos?'")
    
    try:
        print("\nPress Ctrl+C to stop the server...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        if api_process:
            api_process.terminate()
            api_process.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main() 