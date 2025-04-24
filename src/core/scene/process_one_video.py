import os
import sys
from pathlib import Path
import shutil
import argparse
import ray
import time

def process_single_video(video_path, output_folder='output'):
    """
    Process a single video with Ray, ensuring that Ray is initialized and
    shut down properly to avoid filling up disk space.
    
    Args:
        video_path (str): Path to the video file to process
        output_folder (str): Path to folder where processed video will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Clean up any existing Ray session files
    ray_temp_dir = "/tmp/ray"
    if os.path.exists(ray_temp_dir):
        try:
            # Only remove subdirectories not the main directory
            for item in os.listdir(ray_temp_dir):
                item_path = os.path.join(ray_temp_dir, item)
                if os.path.isdir(item_path) and "session" in item:
                    print(f"Cleaning up Ray session: {item_path}")
                    shutil.rmtree(item_path, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up Ray directories: {e}")
    
    # Import here to ensure Ray isn't initialized on import
    from rayprocessor import RayVideoProcessor
    
    try:
        print(f"\nProcessing {os.path.basename(video_path)}...")
        
        # Initialize Ray with limited memory
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, 
                    object_store_memory=1 * 1024 * 1024 * 1024,  # 1GB object store
                    _temp_dir=f"/tmp/ray_video_{int(time.time())}")  # Unique temp dir
        
        # Create video processor
        processor = RayVideoProcessor()
        
        # Process the video
        output_file = processor.process_video(video_path)
        
        # Always shut down Ray after processing a video
        ray.shutdown()
        
        print(f"Successfully processed {os.path.basename(video_path)}")
        print(f"Output saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        if ray.is_initialized():
            ray.shutdown()
        return None
    finally:
        # Ensure Ray is shut down
        if ray.is_initialized():
            ray.shutdown()
            
def main():
    parser = argparse.ArgumentParser(description='Process a single video with Ray')
    parser.add_argument('video', help='Path to the video file to process')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output folder path (default: output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
        
    process_single_video(args.video, args.output)

if __name__ == "__main__":
    print("Starting Ray-powered video processing for a single video...")
    main()
    print("\nVideo processing completed!") 