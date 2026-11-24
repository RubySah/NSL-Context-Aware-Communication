"""Classify the environment using CLIP."""

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
        self.interval = config["context"]["scene_detection_interval"]
        self.clip_model = CLIPModel(
            config["context"]["model"],
            config["device"],
            SCENE_PROMPTS,
        )

        self.frame_count = 0
        self.current_scene = "Detecting..."
        self.current_score = 0.0

    @torch.inference_mode()
    def update(self, frame):
        self.frame_count += 1

        if self.frame_count % self.interval != 0:
            return self.current_scene, self.current_score

        image_features = self.clip_model.encode_image(frame)

        similarities = (
            100.0 * image_features @ self.clip_model.text_features.T
        )

        probabilities = similarities.softmax(dim=-1)[0]
        index = int(probabilities.argmax())

        self.current_scene = SCENE_PROMPTS[index].replace(
            "A person ", ""
        ).rstrip(".")

        self.current_score = float(probabilities[index]) * 100

        return self.current_scene, self.current_score
