import cv2
from pathlib import Path
from typing import Dict
import logging


class VideoFrameManager:
    def __init__(self, storage_path: str, target_resolution: tuple, compression_level: int):
        self.storage_path = Path(storage_path)
        self.target_resolution = target_resolution
        self.compression_level = compression_level
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path: str, sequence_name: str, frame_step: int = 1):
        cap = cv2.VideoCapture(video_path)
        metadata = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'resolution': (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        }

        sequence_dir = self.storage_path / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                frame = cv2.resize(frame, self.target_resolution)
                cv2.imwrite(
                    str(sequence_dir / f"frame_{frame_idx:04d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.compression_level]
                )
            frame_idx += 1

        cap.release()
        return type('Metadata', (), metadata)

    def get_frames(self, sequence_name: str):
        sequence_dir = self.storage_path / sequence_name
        for frame_path in sorted(sequence_dir.glob("frame_*.jpg")):
            yield cv2.imread(str(frame_path))


class VideoManager:
    """
    Manages video loading and metadata extraction for the core integration system.
    """
    
    def __init__(self):
        """Initialize the video manager."""
        self.logger = logging.getLogger(__name__)
        self._current_video_path = None
        self._current_video_info = None
    
    def load_video(self, video_path: str) -> Dict:
        """
        Load a video and extract its metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        video_path = str(video_path)
        
        # Check if this is the same video we already loaded
        if self._current_video_path == video_path and self._current_video_info:
            return self._current_video_info
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Release the video capture
            cap.release()
            
            # Create video info dictionary
            video_info = {
                "path": video_path,
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "duration": duration,
                "codec": self._get_video_codec(video_path),
                "file_size": Path(video_path).stat().st_size if Path(video_path).exists() else 0
            }
            
            # Cache the video info
            self._current_video_path = video_path
            self._current_video_info = video_info
            
            self.logger.info(f"Loaded video: {Path(video_path).name} ({width}x{height}, {fps:.2f} FPS, {duration:.2f}s)")
            
            return video_info
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {e}")
            raise ValueError(f"Failed to load video: {e}")
    
    def _get_video_codec(self, video_path: str) -> str:
        """
        Get the video codec information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            String representing the codec or 'unknown'
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                cap.release()
                return codec.strip()
        except Exception:
            pass
        return "unknown"
    
    def get_video_info(self, video_path: str = None) -> Dict:
        """
        Get video information without reloading if already cached.
        
        Args:
            video_path: Optional path to video (uses cached if None)
            
        Returns:
            Dictionary containing video metadata
        """
        if video_path is None:
            if self._current_video_info:
                return self._current_video_info
            else:
                raise ValueError("No video loaded and no path provided")
        
        return self.load_video(video_path)
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate that a video file is readable.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video is valid, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            is_valid = cap.isOpened()
            cap.release()
            return is_valid
        except Exception:
            return False
    
    def extract_frame(self, video_path: str, frame_number: int):
        """
        Extract a specific frame from the video.
        
        Args:
            video_path: Path to the video file
            frame_number: Frame number to extract (0-based)
            
        Returns:
            Frame as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            self.logger.error(f"Error extracting frame {frame_number} from {video_path}: {e}")
            return None
    
    def get_frame_generator(self, video_path: str, start_frame: int = 0, max_frames: int = None):
        """
        Get a generator for video frames.
        
        Args:
            video_path: Path to the video file
            start_frame: Frame to start from (0-based)
            max_frames: Maximum number of frames to yield
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return
            
            # Set start frame position
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_number = start_frame
            frames_yielded = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                yield frame_number, frame
                frame_number += 1
                frames_yielded += 1
                
                # Check if we've reached max frames
                if max_frames is not None and frames_yielded >= max_frames:
                    break
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Error in frame generator for {video_path}: {e}")
    
    def clear_cache(self):
        """Clear the cached video information."""
        self._current_video_path = None
        self._current_video_info = None
