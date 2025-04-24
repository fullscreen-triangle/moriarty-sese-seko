import requests
import json
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SportsVideoClient:
    """
    Client for the Sports Video Analysis API.
    This client shows how external applications can query the sports video analysis data.
    """
    
    def __init__(self, api_url="http://localhost:8000", api_key=None):
        """
        Initialize the client.
        
        Args:
            api_url (str): Base URL for the API
            api_key (str): OpenAI API key (if None, will try to load from environment)
        """
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided.")
    
    def query(self, query_text, max_results=5):
        """
        Query the video analysis data.
        
        Args:
            query_text (str): Natural language query
            max_results (int): Maximum number of results to return
            
        Returns:
            dict: Response from the API
        """
        endpoint = f"{self.api_url}/query"
        payload = {
            "query": query_text,
            "max_results": max_results,
            "openai_api_key": self.api_key
        }
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying API: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def list_videos(self):
        """
        List all processed videos in the system.
        
        Returns:
            list: List of video metadata
        """
        endpoint = f"{self.api_url}/videos"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing videos: {e}")
            return None
    
    def add_video(self, video_path, sample_rate=5):
        """
        Add a new video to the system.
        
        Args:
            video_path (str): Path to the processed video
            sample_rate (int): Sample 1 frame every N frames
            
        Returns:
            dict: Response from the API
        """
        endpoint = f"{self.api_url}/videos/add"
        payload = {
            "video_path": video_path,
            "sample_rate": sample_rate,
            "openai_api_key": self.api_key
        }
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error adding video: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def trigger_reindexing(self):
        """
        Trigger reindexing of all videos.
        
        Returns:
            dict: Response from the API
        """
        endpoint = f"{self.api_url}/index"
        
        try:
            response = requests.post(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error triggering reindexing: {e}")
            return None


def main():
    """Main function to demonstrate the client usage."""
    parser = argparse.ArgumentParser(description="Sports Video Analysis Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the video analysis data")
    query_parser.add_argument("query_text", help="Natural language query")
    query_parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results")
    
    # List videos command
    list_parser = subparsers.add_parser("list", help="List all processed videos")
    
    # Add video command
    add_parser = subparsers.add_parser("add", help="Add a new video")
    add_parser.add_argument("video_path", help="Path to the processed video")
    add_parser.add_argument("--sample-rate", type=int, default=5, help="Sample 1 frame every N frames")
    
    # Reindex command
    reindex_parser = subparsers.add_parser("reindex", help="Trigger reindexing of all videos")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create client
    client = SportsVideoClient()
    
    # Execute command
    if args.command == "query":
        result = client.query(args.query_text, args.max_results)
        if result:
            print(f"\nQuery: {args.query_text}")
            print("\nResponse:")
            print(result["response"])
    
    elif args.command == "list":
        videos = client.list_videos()
        if videos:
            print(f"Found {len(videos)} videos:")
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video['original_name']} ({video['width']}x{video['height']}, {video['fps']} FPS, {video['total_frames']} frames)")
    
    elif args.command == "add":
        result = client.add_video(args.video_path, args.sample_rate)
        if result:
            print(f"Video processing started: {result['video_name']}")
            print(f"Message: {result['message']}")
    
    elif args.command == "reindex":
        result = client.trigger_reindexing()
        if result:
            print(f"Message: {result['message']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 