import cv2
from pathlib import Path
from typing import Dict


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
