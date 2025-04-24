import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
from torchvision.models import detection
from torchvision.transforms import functional as F


@dataclass
class DetectedPerson:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float]
    keypoints: np.ndarray = None


class HumanDetector:
    def __init__(self, confidence_threshold: float = 0.5, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.pixel_per_meter = None

    def detect_humans(self, frame: np.ndarray) -> Tuple[List[DetectedPerson], Dict[Tuple[int, int], float]]:
        image_tensor = self._preprocess_image(frame)

        with torch.no_grad():
            predictions = self.model([image_tensor])

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        detected_people = []
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score > self.confidence_threshold:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                detected_people.append(DetectedPerson(
                    bbox=tuple(box),
                    confidence=float(score),
                    center=center
                ))

        distances = self._calculate_distances(detected_people)
        return detected_people, distances

    def calibrate(self, known_distance_pixels: float, known_distance_meters: float):
        self.pixel_per_meter = known_distance_pixels / known_distance_meters

    def _calculate_distances(self, detections: List[DetectedPerson]) -> Dict[Tuple[int, int], float]:
        distances = {}
        for i, person1 in enumerate(detections):
            for j, person2 in enumerate(detections[i + 1:], i + 1):
                pixel_distance = np.sqrt(
                    (person1.center[0] - person2.center[0]) ** 2 +
                    (person1.center[1] - person2.center[1]) ** 2
                )

                meter_distance = pixel_distance / self.pixel_per_meter if self.pixel_per_meter else pixel_distance
                distances[(i, j)] = meter_distance

        return distances

    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)
        return image.to(self.device)

    def draw_detections(self, frame: np.ndarray,
                        detections: List[DetectedPerson],
                        distances: Dict[Tuple[int, int], float]) -> np.ndarray:
        output = frame.copy()

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"Person {i}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (i, j), distance in distances.items():
            start = tuple(map(int, detections[i].center))
            end = tuple(map(int, detections[j].center))

            cv2.line(output, start, end, (255, 0, 0), 2)

            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(output, f"{distance:.2f}m", mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return output
