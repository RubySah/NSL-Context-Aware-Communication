"""Track hand movement and pauses."""

import numpy as np


class MotionTracker:
    def __init__(self, velocity_threshold):
        self.velocity_threshold = velocity_threshold
        self.previous = None

    def velocity(self, features):
        hands = np.asarray(features[165:], dtype=np.float32)

        if self.previous is None:
            self.previous = hands
            return 0.0

        velocity = float(np.mean(np.abs(hands - self.previous)))
        self.previous = hands

        return velocity

    def is_moving(self, features):
        return self.velocity(features) > self.velocity_threshold

    def reset(self):
        self.previous = None