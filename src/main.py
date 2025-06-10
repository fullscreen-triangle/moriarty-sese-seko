"""
Moriarty - SUPER SIMPLE Video Processing Script

Just change the video path below and run: python src/main.py
NO RAY, NO COMPLEX PIPELINE - Just OpenCV + MediaPipe
"""

import cv2
import json
import numpy as np
from pathlib import Path
import logging

#########################################################################
# 🎯 CHANGE THIS VIDEO PATH TO PROCESS DIFFERENT VIDEOS:
#########################################################################
VIDEO_PATH = "/Users/kundai/Development/computer-vision/moriarty-sese-seko/public/boundary-nz.mp4"  # <-- CHANGE THIS LINE
#########################################################################

# Simple settings
OUTPUT_DIR = "output"
MODELS_DIR = "models"
GENERATE_VIDEO = True  # Set to False to skip video output (faster)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("moriarty")

def main():
    """Process video with simple OpenCV + MediaPipe."""
    
    logger.info("="*60)
    logger.info("🎯 SUPER SIMPLE MORIARTY PROCESSOR")
    logger.info("="*60)
    
    # Check video exists
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        logger.error(f"❌ Video not found: {video_path}")
        return
    
    logger.info(f"📹 Processing: {video_path.name}")
    
    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    models_dir = Path(MODELS_DIR)
    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Import MediaPipe
        import mediapipe as mp
        
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Could not open video: {video_path}")
            return
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Setup output video if needed
        video_writer = None
        if GENERATE_VIDEO:
            output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            logger.info(f"🎬 Will save annotated video to: {output_video_path}")
        
        # Process frames
        frame_count = 0
        pose_data = []
        valid_poses = 0
        
        logger.info("🚀 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            
            # Extract pose data
            frame_data = {
                "frame": frame_count,
                "timestamp": frame_count / fps,
                "poses": []
            }
            
            if results.pose_landmarks:
                valid_poses += 1
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                
                frame_data["poses"].append({
                    "landmarks": landmarks,
                    "confidence": sum(lm.visibility for lm in results.pose_landmarks.landmark) / len(results.pose_landmarks.landmark)
                })
                
                # Draw pose on frame if generating video
                if GENERATE_VIDEO and video_writer:
                    annotated_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    video_writer.write(annotated_frame)
            else:
                # No pose detected, write original frame
                if GENERATE_VIDEO and video_writer:
                    video_writer.write(frame)
            
            pose_data.append(frame_data)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"   Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Valid poses: {valid_poses}")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        pose.close()
        
        # Save pose data
        pose_model_path = models_dir / f"{video_path.stem}_pose_data.json"
        with open(pose_model_path, 'w') as f:
            json.dump({
                "video_info": {
                    "filename": video_path.name,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames
                },
                "pose_data": pose_data,
                "summary": {
                    "valid_poses": valid_poses,
                    "total_frames": frame_count,
                    "pose_detection_rate": valid_poses / frame_count if frame_count > 0 else 0
                }
            }, f, indent=2)
        
        # Success!
        logger.info("="*60)
        logger.info("🎉 SUCCESS!")
        logger.info("="*60)
        logger.info(f"✅ Processed {frame_count} frames")
        logger.info(f"✅ Valid poses: {valid_poses}/{frame_count} ({valid_poses/frame_count*100:.1f}%)")
        logger.info(f"🤖 Pose data saved: {pose_model_path}")
        
        if GENERATE_VIDEO:
            logger.info(f"🎬 Annotated video saved: {output_video_path}")
        
        logger.info("="*60)
        logger.info("👉 To process another video:")
        logger.info("   1. Change VIDEO_PATH in main.py (line 12)")
        logger.info("   2. Run: python src/main.py")
        logger.info("="*60)
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Try: pip install opencv-python mediapipe")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 