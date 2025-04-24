import cv2
import numpy as np
from typing import Dict, List

from activity.base import CombatSport


class CombatVisualizer:
    def __init__(self, sport_type: CombatSport):
        self.sport_type = sport_type
        self.colors = {
            'strike': (0, 0, 255),  # Red
            'block': (0, 255, 0),  # Green
            'movement': (255, 0, 0),  # Blue
            'contact': (255, 255, 0)  # Yellow
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_analysis(self,
                      frame: np.ndarray,
                      analysis_data: Dict) -> np.ndarray:
        """
        Draw analysis overlays on frame.
        """
        # Draw skeleton
        frame = self._draw_skeleton(frame, analysis_data['poses'])

        # Draw strikes/techniques
        frame = self._draw_techniques(frame, analysis_data['techniques'])

        # Draw force vectors
        frame = self._draw_force_vectors(frame, analysis_data['forces'])

        # Draw metrics
        frame = self._draw_metrics(frame, analysis_data['metrics'])

        # Draw predictions
        frame = self._draw_predictions(frame, analysis_data['predictions'])

        return frame

    def _draw_skeleton(self,
                       frame: np.ndarray,
                       poses: Dict) -> np.ndarray:
        """
        Draw sport-specific skeleton visualization.
        """
        for person_id, keypoints in poses.items():
            # Draw joints
            for joint in keypoints:
                cv2.circle(frame,
                           (int(joint[0]), int(joint[1])),
                           4,
                           self.colors['movement'],
                           -1)

            # Draw connections
            for connection in self._get_sport_connections():
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                cv2.line(frame,
                         (int(pt1[0]), int(pt1[1])),
                         (int(pt2[0]), int(pt2[1])),
                         self.colors['movement'],
                         2)

        return frame

    def _draw_techniques(self,
                         frame: np.ndarray,
                         techniques: List[Dict]) -> np.ndarray:
        """
        Draw technique-specific visualizations.
        """
        for technique in techniques:
            if technique['type'] == 'strike':
                # Draw strike path
                cv2.polylines(frame,
                              [technique['path']],
                              False,
                              self.colors['strike'],
                              2)

                # Draw impact point
                if technique.get('contact_point'):
                    cv2.circle(frame,
                               technique['contact_point'],
                               8,
                               self.colors['contact'],
                               -1)

            # Draw technique name and metrics
            cv2.putText(frame,
                        f"{technique['name']} ({technique['score']:.2f})",
                        (int(technique['path'][0][0]),
                         int(technique['path'][0][1] - 10)),
                        self.font,
                        0.5,
                        self.colors['strike'],
                        1)

        return frame

    def _draw_force_vectors(self,
                            frame: np.ndarray,
                            forces: List[Dict]) -> np.ndarray:
        """
        Draw force vectors with magnitude visualization.
        """
        for force in forces:
            # Draw arrow
            start_point = tuple(map(int, force['start_point']))
            end_point = tuple(map(int, force['end_point']))

            # Scale arrow length by force magnitude
            magnitude = force['magnitude']
            normalized_magnitude = min(magnitude / 1000, 1.0)  # Normalize to [0,1]

            cv2.arrowedLine(frame,
                            start_point,
                            end_point,
                            self.colors['strike'],
                            thickness=int(2 + normalized_magnitude * 3))

            # Draw force value
            cv2.putText(frame,
                        f"{magnitude:.0f}N",
                        end_point,
                        self.font,
                        0.5,
                        self.colors['strike'],
                        1)

        return frame

    def _draw_metrics(self,
                      frame: np.ndarray,
                      metrics: Dict) -> np.ndarray:
        """
        Draw performance metrics overlay.
        """
        # Draw metrics panel
        panel_height = 120
        panel_width = 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        # Add metrics text
        metrics_text = [
            f"Speed: {metrics['speed']:.1f} m/s",
            f"Power: {metrics['power']:.1f} W",
            f"Efficiency: {metrics['efficiency']:.2f}",
            f"Score: {metrics['score']:.1f}"
        ]

        for i, text in enumerate(metrics_text):
            cv2.putText(panel,
                        text,
                        (10, 30 * (i + 1)),
                        self.font,
                        0.6,
                        (255, 255, 255),
                        1)

        # Overlay panel on frame
        frame[10:10 + panel_height, 10:10 + panel_width] = panel

        return frame
