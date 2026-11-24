"""Manage the sliding landmark sequence."""

from collections import deque


class SequenceBuffer:
    def __init__(self, max_length):
        self.frames = deque(maxlen=max_length)

    def add(self, features):
        self.frames.append(features)

    def clear(self):
        self.frames.clear()

    def __len__(self):
        return len(self.frames)

    def get(self):
        return list(self.frames)

    def ready(self, min_length):
        return len(self.frames) >= min_length