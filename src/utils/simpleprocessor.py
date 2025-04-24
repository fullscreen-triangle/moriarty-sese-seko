import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os
import time

class SimpleVideoProcessor:
    def __init__(self):
        """Initialize the video processor with MediaPipe pose model."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # For velocity calculations
        self.prev_landmarks = None
        self.prev_time = None
        
    def process_video(self, video_path):
        """Process video and save annotated output."""
        print(f"Starting to process {video_path}...")
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        output_path = self.output_dir / f"annotated_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            raise Exception(f"Could not open output video file {output_path}")
        
        frame_count = 0
        
        # Process each frame
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processing frame {frame_count}/{total_frames}")
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.pose.process(frame_rgb)
            
            # Draw annotations if pose landmarks are detected
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Calculate and draw metrics
                metrics = self._calculate_metrics(results.pose_landmarks, fps)
                self._draw_metrics(frame, metrics)
                
                # Add a timestamp to confirm this is being processed
                cv2.putText(
                    frame, 
                    f"Frame: {frame_count}/{total_frames}", 
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            
            # Write the frame to output video
            out.write(frame)
            
            # Update previous landmarks for next frame
            self.prev_landmarks = results.pose_landmarks if results.pose_landmarks else self.prev_landmarks
            self.prev_time = time.time()
        
        # Release resources
        cap.release()
        out.release()
        
        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"Successfully processed video. Output saved to {output_path}")
            print(f"Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return str(output_path)
        else:
            raise Exception(f"Output file is missing or too small: {output_path}")
    
    def _calculate_metrics(self, landmarks, fps):
        """Calculate metrics based on pose landmarks."""
        metrics = {}
        
        # Extract key points
        if landmarks:
            # Get positions of key landmarks
            nose = (landmarks.landmark[0].x, landmarks.landmark[0].y)
            left_shoulder = (landmarks.landmark[11].x, landmarks.landmark[11].y)
            right_shoulder = (landmarks.landmark[12].x, landmarks.landmark[12].y)
            left_hip = (landmarks.landmark[23].x, landmarks.landmark[23].y)
            right_hip = (landmarks.landmark[24].x, landmarks.landmark[24].y)
            left_knee = (landmarks.landmark[25].x, landmarks.landmark[25].y)
            right_knee = (landmarks.landmark[26].x, landmarks.landmark[26].y)
            left_ankle = (landmarks.landmark[27].x, landmarks.landmark[27].y)
            right_ankle = (landmarks.landmark[28].x, landmarks.landmark[28].y)
            
            # Calculate shoulder width
            shoulder_width = self._euclidean_distance(left_shoulder, right_shoulder)
            metrics["shoulder_width"] = shoulder_width
            
            # Calculate hip width
            hip_width = self._euclidean_distance(left_hip, right_hip)
            metrics["hip_width"] = hip_width
            
            # Calculate leg angles
            left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            metrics["left_leg_angle"] = left_leg_angle
            metrics["right_leg_angle"] = right_leg_angle
            
            # Calculate body height (simplified)
            head_to_foot = self._euclidean_distance(nose, left_ankle)
            metrics["height"] = head_to_foot
            
            # If we have previous landmarks, calculate velocity
            if self.prev_landmarks and self.prev_time:
                prev_nose = (self.prev_landmarks.landmark[0].x, self.prev_landmarks.landmark[0].y)
                dt = 1.0 / fps  # Time difference based on fps
                
                # Calculate displacement
                displacement = self._euclidean_distance(nose, prev_nose)
                
                # Calculate velocity
                velocity = displacement / dt
                metrics["velocity"] = velocity
        
        return metrics
    
    def _euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points."""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def _draw_metrics(self, frame, metrics):
        """Draw metrics on the frame."""
        y_pos = 30
        for metric, value in metrics.items():
            cv2.putText(
                frame, 
                f"{metric}: {value:.2f}", 
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_pos += 30 