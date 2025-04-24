import cv2
import numpy as np



class SkeletonDrawer:
    def __init__(self):
        self.joint_pairs = [
            (0, 1),  # Head to neck
            (1, 2),  # Neck to left shoulder
            (1, 3),  # Neck to right shoulder
            (2, 4),  # Left shoulder to left hip
            (3, 5),  # Right shoulder to right hip
            (4, 6),  # Left hip to left knee
            (5, 7),  # Right hip to right knee
            (6, 8),  # Left knee to left ankle
            (7, 9)  # Right knee to right ankle
        ]

    def detect_keypoints(self, roi):
        """Estimate keypoints based on body proportions"""
        h, w = roi.shape[:2]

        return [
            (w // 2, h // 8),  # 0: Head
            (w // 2, h // 6),  # 1: Neck
            (w // 4, h // 4),  # 2: Left shoulder
            (3 * w // 4, h // 4),  # 3: Right shoulder
            (w // 3, h // 2),  # 4: Left hip
            (2 * w // 3, h // 2),  # 5: Right hip
            (w // 3, 3 * h // 4),  # 6: Left knee
            (2 * w // 3, 3 * h // 4),  # 7: Right knee
            (w // 3, h - 10),  # 8: Left ankle
            (2 * w // 3, h - 10)  # 9: Right ankle
        ]

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def draw_skeleton(self, frame, detection):
        """Draw skeleton and joint angles on detected athlete"""
        x, y, w, h = detection
        roi = frame[y:y + h, x:x + w].copy()

        # Get keypoints
        keypoints = self.detect_keypoints(roi)

        # Adjust keypoints to frame coordinates
        adjusted_keypoints = [(px + x, py + y) for px, py in keypoints]

        # Draw skeleton lines
        for pair in self.joint_pairs:
            pt1 = tuple(map(int, adjusted_keypoints[pair[0]]))
            pt2 = tuple(map(int, adjusted_keypoints[pair[1]]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw joints
        for point in adjusted_keypoints:
            cv2.circle(frame, tuple(map(int, point)), 4, (0, 0, 255), -1)

        # Calculate and draw important angles
        angles = {
            'Knee': (6, 4, 8),  # Left knee angle
            'Hip': (2, 4, 6),  # Left hip angle
            'Trunk': (1, 4, 6)  # Trunk angle
        }

        for name, (p1, p2, p3) in angles.items():
            angle = self.calculate_angle(
                adjusted_keypoints[p1],
                adjusted_keypoints[p2],
                adjusted_keypoints[p3]
            )

            # Draw angle text
            text_pos = tuple(map(int, adjusted_keypoints[p2]))
            cv2.putText(frame,
                        f"{name}: {angle:.1f}°",
                        (text_pos[0] - 30, text_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)

        return frame
