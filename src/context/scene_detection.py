"""Two-stage scene understanding with CLIP, run in a background thread."""

import json
import os
import re
import threading
import time
import traceback
from collections import deque

import torch

from src.context.clip_model import CLIPModel
from src.context.scene_prompts import SCENE_PROMPTS, SCENE_OBJECTS, OBJECT_TEMPLATE


# "a hospital bed" -> "hospital bed"
def clean_name(text):
    return re.sub(r"^(a|an|the)\s+", "", text.strip().lower())


class SceneDetector:
    def __init__(self, config):
        settings = config["context"]
        self.interval = settings["scene_interval_seconds"]
        self.top_k = settings["object_top_k"]
        self.min_similarity = settings["object_min_similarity"]
        self.json_path = config["paths"]["context_json"]

        self.clip_model = CLIPModel(
            settings["model"],
            config["device"],
            SCENE_PROMPTS,
            SCENE_OBJECTS,
            OBJECT_TEMPLATE,
        )

        self.history = deque(maxlen=settings["smoothing_window"])
        self.context = {
            "environment": "detecting...",
            "environment_confidence": 0.0,
            "visual_context": [],
        }

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

        return self.context["environment"], self.context["environment_confidence"] * 100

    # Background loop: analyze whatever frame was submitted last, then save the JSON.
    def _worker(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None

            if frame is None:
                time.sleep(0.05)
                continue

            try:
                self.context = self._analyze(frame)
                self._save(self.context)
            except Exception:
                traceback.print_exc()

    @torch.inference_mode()
    def _analyze(self, frame):
        features = self.clip_model.encode_image(frame)

        # Stage 1: environment, averaged over the last few detections to avoid flicker.
        self.history.append(self.clip_model.scene_probabilities(features))
        probabilities = torch.stack(list(self.history)).mean(dim=0)
        index = int(probabilities.argmax())
        scene = self.clip_model.scene_names[index]

        # Stage 2: objects, taken only from the chosen scene's own list.
        objects = self.clip_model.top_objects(
            features, scene, self.top_k, self.min_similarity
        )

        return {
            "environment": scene.lower(),
            "environment_confidence": round(float(probabilities[index]), 2),
            "visual_context": [clean_name(name) for name in objects],
        }

    # Write the JSON atomically so a reader never sees a half-written file.
    def _save(self, context):
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.json_path.with_suffix(".tmp")

            with temp.open("w", encoding="utf-8") as file:
                json.dump(context, file, indent=2)

            os.replace(temp, self.json_path)
        except OSError:
            pass   # file busy (e.g. open elsewhere); the next cycle will retry

    def close(self):
        self._stop.set()