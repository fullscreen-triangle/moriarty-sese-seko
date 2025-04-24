import cv2
import numpy as np


class AthleteDetector:
    def __init__(self):
        # HSV range for track green
        self.green_lower = np.array([35, 50, 50])
        self.green_upper = np.array([85, 255, 255])

        # Distance thresholds (in pixels)
        self.min_distance = 10  # Minimum distance from green to be considered an athlete
        self.max_distance = 200  # Maximum distance from green to consider

    def detect_athletes(self, frame):
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create green mask
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # Initialize human detector (you can use your preferred method)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Detect all humans
        humans, _ = hog.detectMultiScale(frame)

        athletes = []
        distances = []

        for (x, y, w, h) in humans:
            # Get the bottom center point of the bounding box (feet position)
            foot_point = (x + w // 2, y + h)

            # Calculate distance to nearest green pixel
            distance = self._calculate_distance_to_green(foot_point, green_mask)

            # If person is not standing on green and within reasonable distance
            if self.min_distance < distance < self.max_distance:
                athletes.append((x, y, w, h))
                distances.append(distance)

        # Sort athletes by distance from green (can be used for lane allocation)
        athletes = [x for _, x in sorted(zip(distances, athletes))]
        distances.sort()

        return athletes, distances

    def _calculate_distance_to_green(self, point, green_mask):
        """Calculate minimum distance from point to nearest green pixel"""
        # Create distance transform of inverted green mask
        dist_transform = cv2.distanceTransform(255 - green_mask, cv2.DIST_L2, 3)

        # Get distance at the point
        x, y = point
        return dist_transform[y, x]

    def draw_athletes(self, frame, athletes, distances):
        """Draw bounding boxes and distance information"""
        output = frame.copy()

        for i, ((x, y, w, h), distance) in enumerate(zip(athletes, distances)):
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw distance and probable lane number
            lane = i + 1  # Simple lane assignment based on distance
            label = f"Athlete {lane} (d={distance:.1f}px)"
            cv2.putText(output, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output
