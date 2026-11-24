"""Manage temporal sign recognition decisions."""

from collections import Counter


class RecognitionManager:
    def __init__(self, config):
        settings = config["recognition"]
        motion = config["motion"]

        self.min_frames = settings["min_frames"]
        self.max_frames = settings["max_frames"]
        self.check_interval = settings["check_interval"]
        self.start_threshold = settings["start_threshold"]
        self.end_threshold = settings["end_threshold"]

        self.word_pause = motion["word_pause_seconds"]
        self.sentence_pause = motion["sentence_pause_seconds"]

        self.history = []
        self.prediction = None
        self.confidence = 0.0
        self.top3 = []

        self.last_check = 0

    def threshold(self, length):
        progress = min(
            max((length - self.min_frames) /
                (self.max_frames - self.min_frames), 0),
            1,
        )

        return (
            self.start_threshold
            - progress * (self.start_threshold - self.end_threshold)
        )

    def should_check(self, length, frame_count):
        return (
            length >= self.min_frames
            and frame_count - self.last_check >= self.check_interval
        )

    def update(self, result, frame_count):
        if result is None:
            return

        self.prediction = result["label"]
        self.confidence = result["confidence"]
        self.top3 = result["top3"]

        self.history.append(self.prediction)
        self.history = self.history[-5:]
        self.last_check = frame_count

    def stable_prediction(self):
        if not self.history:
            return self.prediction

        return Counter(self.history).most_common(1)[0][0]

    def should_commit(self, length):
        if length < self.min_frames or not self.prediction:
            return False

        return self.confidence / 100 >= self.threshold(length)

    def reset(self):
        self.history.clear()
        self.prediction = None
        self.confidence = 0.0
        self.top3.clear()
        self.last_check = 0