"""Draw detected pose and hand landmarks."""

import cv2
import mediapipe as mp


class LandmarkVisualizer:
    def __init__(self):
        self.connections = mp.solutions.holistic
        self.drawer = mp.solutions.drawing_utils

    def draw(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        return frame
