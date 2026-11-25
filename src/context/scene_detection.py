"""Classify the environment using CLIP in a background thread."""

import threading
import time

import torch

from src.context.clip_model import CLIPModel


SCENE_PROMPTS = [
    "A person inside a hospital.",
    "A person inside a classroom.",
    "A person inside a home living room.",
    "A person on an outdoor street or public place.",
    "A person inside an office.",
    "A person inside a kitchen or dining area.",
    "A person inside a shop or market.",
    "A person inside a library or study room.",
    "A person inside a pharmacy or medical shop.",
    "A person inside a waiting room or reception.",
    "A person inside a restaurant or cafe.",
    "A person inside a community center.",
]


class SceneDetector:
    def __init__(self, config):
        self.interval = config["context"]["scene_interval_seconds"]
        self.clip_model = CLIPModel(
            config["context"]["model"],
            config["device"],
            SCENE_PROMPTS,
        )

        self.current_scene = "Detecting..."
        self.current_score = 0.0

        self._pending = None
        self._last_submit = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # Hand the newest frame to the worker every few seconds; return the latest result.
    def update(self, frame):
        now = time.monotonic()

        if now - self._last_submit >= self.interval:
            with self._lock:
                self._pending = frame.copy()
            self._last_submit = now

        return self.current_scene, self.current_score

    # Background loop: classify whatever frame was submitted last.
    def _worker(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None

            if frame is None:
                time.sleep(0.05)
                continue

            self.current_scene, self.current_score = self._classify(frame)

    @torch.inference_mode()
    def _classify(self, frame):
        image_features = self.clip_model.encode_image(frame)
        similarities = 100.0 * image_features @ self.clip_model.text_features.T
        probabilities = similarities.softmax(dim=-1)[0]
        index = int(probabilities.argmax())

        scene = SCENE_PROMPTS[index].replace("A person ", "").rstrip(".")
        return scene, float(probabilities[index]) * 100

    def close(self):
        self._stop.set()