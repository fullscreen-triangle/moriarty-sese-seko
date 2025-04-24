import os
from pathlib import Path
from rayprocessor import RayVideoProcessor

def process_all_videos(input_folder='public', output_folder='output'):
    """
    Process all videos in the input folder and save results to output folder.
    
    Args:
        input_folder (str): Path to folder containing input videos
        output_folder (str): Path to folder where processed videos will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all mp4 files in the input folder
    input_path = Path(input_folder)
    video_files = list(input_path.glob('*.mp4'))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create video processor
    processor = RayVideoProcessor()
    
    # Process each video
    for video_file in video_files:
        print(f"\nProcessing {video_file.name}...")
        
        try:
            # Process the video
            output_file = processor.process_video(str(video_file))
            print(f"Successfully processed {video_file.name}")
            print(f"Output saved to {output_file}")
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Ray-powered video processing pipeline...")
    process_all_videos()
    print("\nAll videos have been processed!") 