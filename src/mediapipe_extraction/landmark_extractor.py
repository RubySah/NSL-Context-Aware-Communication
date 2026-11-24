"""Extract pose and hand landmarks using MediaPipe Tasks."""

import cv2
import numpy as np
import mediapipe as mp

class LandmarkExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        pose = self._pose_array(results)
        left_hand = self._hand_array(results.left_hand_landmarks)
        right_hand = self._hand_array(results.right_hand_landmarks)

        features = np.concatenate([
            pose.flatten(),
            left_hand.flatten(),
            right_hand.flatten(),
        ])

        hand_detected = (
            results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None
        )

        return features.astype(np.float32), hand_detected, results

    @staticmethod
    def _pose_array(results):
        if results.pose_landmarks is None:
            return np.zeros((33, 5), dtype=np.float32)

        return np.array([
            [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)

    @staticmethod
    def _hand_array(landmarks):
        if landmarks is None:
            return np.zeros((21, 3), dtype=np.float32)

        return np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ], dtype=np.float32)

    def close(self):
        self.holistic.close()