import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_helpers import save_image_safely

# Constants
COLORS = {
    'skeleton': (0, 255, 0),  # Green
    'text': (255, 255, 255),  # White
    'background': (0, 0, 0),  # Black
    'highlight': (0, 0, 255)  # Red
}

def draw_skeleton(frame, pose_data, frame_idx=0, confidence_threshold=0.5):
    """Draw skeleton on frame based on pose data"""
    output = frame.copy()
    
    if frame_idx >= len(pose_data):
        return output
    
    landmarks = pose_data[frame_idx].get('landmarks', [])
    connections = pose_data[frame_idx].get('connections', [])
    
    # Draw joints
    for i, point in enumerate(landmarks):
        if point[2] > confidence_threshold:  # Check confidence
            cv2.circle(output, (int(point[0]), int(point[1])), 4, COLORS['skeleton'], -1)
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx < len(landmarks) and end_idx < len(landmarks) and
            landmarks[start_idx][2] > confidence_threshold and
            landmarks[end_idx][2] > confidence_threshold):
            
            start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
            end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
            
            cv2.line(output, start_point, end_point, COLORS['skeleton'], 2)
    
    return output

def create_annotated_video(video_path, pose_data, analysis_results, output_path):
    """Create annotated video with pose overlay and metrics"""
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw skeleton
        if frame_idx < len(pose_data):
            frame = draw_skeleton(frame, pose_data, frame_idx)
        
        # Add metrics overlay
        add_metrics_overlay(frame, analysis_results, frame_idx)
        
        # Write frame
        out.write(frame)
        
        # Update progress
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}/{total_frames}")
    
    # Release resources
    cap.release()
    out.release()
    
    # Generate thumbnail image
    thumbnail_path = output_path.replace('.mp4', '_thumbnail.png')
    cap = cv2.VideoCapture(output_path)
    ret, frame = cap.read()
    if ret:
        save_image_safely(frame, thumbnail_path)
    cap.release()
    
    return output_path

def add_metrics_overlay(frame, analysis_results, frame_idx):
    """Add metrics overlay to frame"""
    # Example metrics to display
    metrics = {}
    
    # Extract appropriate metrics for this frame
    if 'velocity_data' in analysis_results and frame_idx < len(analysis_results['velocity_data']):
        metrics['velocity'] = analysis_results['velocity_data'][frame_idx].get('overall_velocity', 'N/A')
    
    if 'sprint_mechanics' in analysis_results:
        metrics['stride_length'] = analysis_results['sprint_mechanics'].get('avg_stride_length', 'N/A')
        metrics['stride_freq'] = analysis_results['sprint_mechanics'].get('avg_stride_frequency', 'N/A')
    
    # Draw dark rectangle for text background
    cv2.rectangle(frame, (10, 10), (250, 20 + 20 * len(metrics)), COLORS['background'], -1)
    cv2.rectangle(frame, (10, 10), (250, 20 + 20 * len(metrics)), COLORS['skeleton'], 1)
    
    # Add text for each metric
    y_pos = 30
    for label, value in metrics.items():
        text = f"{label}: {value}"
        cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        y_pos += 20
    
    return frame

def generate_analysis_plots(analysis_results, output_dir):
    """Generate analysis plots as PNG files"""
    plots_generated = []
    
    # 1. Velocity over time plot
    if 'velocity_data' in analysis_results:
        try:
            velocity_data = [frame.get('overall_velocity', 0) for frame in analysis_results['velocity_data']]
            frames = range(len(velocity_data))
            
            plt.figure(figsize=(10, 6))
            plt.plot(frames, velocity_data)
            plt.title('Velocity Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Velocity (m/s)')
            plt.grid(True)
            
            output_path = os.path.join(output_dir, 'velocity_plot.png')
            save_image_safely(None, output_path, is_matplotlib=True)
            plots_generated.append(output_path)
        except Exception as e:
            print(f"Error generating velocity plot: {str(e)}")
    
    # 2. Joint angles plot
    if 'joint_angles' in analysis_results:
        try:
            # Example for knee angle
            knee_angles = [frame.get('knee_angle', 0) for frame in analysis_results['joint_angles']]
            frames = range(len(knee_angles))
            
            plt.figure(figsize=(10, 6))
            plt.plot(frames, knee_angles)
            plt.title('Knee Angle Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            plt.grid(True)
            
            output_path = os.path.join(output_dir, 'knee_angle_plot.png')
            save_image_safely(None, output_path, is_matplotlib=True)
            plots_generated.append(output_path)
        except Exception as e:
            print(f"Error generating joint angles plot: {str(e)}")
    
    # Add more plots as needed
    
    return plots_generated
