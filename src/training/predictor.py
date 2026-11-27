"""Load the trained Transformer and predict isolated signs from raw landmark clips."""

import json

import numpy as np
import torch

from src.preprocessing.features import NORM_DIM, normalize_sequence
from src.training.model import SignLanguageTransformer

RAW_DIM = 291   # 33 pose x 5 + 2 hands x 21 x 3, as produced by LandmarkExtractor


class SignPredictor:
    def __init__(self, config):
        self.device = config["device"]
        self.fixed_length = config["project"]["fixed_length"]
        self.feature_dim = config["project"]["feature_dim"]
        self.num_classes = config["project"]["num_classes"]

        if self.feature_dim != NORM_DIM:
            raise ValueError(
                f"config feature_dim ({self.feature_dim}) must equal the normalizer output ({NORM_DIM})."
            )

        # Temperature from validation calibration; 1.0 means no calibration.
        self.temperature = float(config.get("calibration", {}).get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("calibration.temperature must be positive.")

        checkpoint = torch.load(
            config["paths"]["checkpoint"],
            map_location=self.device,
            weights_only=True,
        )

        with config["paths"]["class_mapping"].open("r", encoding="utf-8") as file:
            self.class_mapping = {
                name: int(index)
                for name, index in json.load(file).items()
            }

        if len(self.class_mapping) != self.num_classes:
            raise ValueError("Class mapping size does not match num_classes.")

        self.idx_to_class = {
            index: name
            for name, index in self.class_mapping.items()
        }

        if set(self.idx_to_class) != set(range(self.num_classes)):
            raise ValueError("Class indices must be continuous from 0.")

        self.model = SignLanguageTransformer(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            max_seq_len=self.fixed_length,
            **config["model"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    # Takes a raw (frames, 291) clip; normalizes it exactly as the training data was.
    @torch.inference_mode()
    def predict(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.ndim != 2 or sequence.shape[1] != RAW_DIM:
            raise ValueError(
                f"Expected raw landmarks (frames, {RAW_DIM}), got {sequence.shape}."
            )

        if len(sequence) == 0:
            return None

        sequence = normalize_sequence(sequence)            # (frames, 199)

        # Center-crop long clips, same as training.
        if len(sequence) > self.fixed_length:
            start = (len(sequence) - self.fixed_length) // 2
            sequence = sequence[start:start + self.fixed_length]

        actual_length = len(sequence)

        padded = np.zeros(
            (self.fixed_length, self.feature_dim),
            dtype=np.float32,
        )
        padded[:actual_length] = sequence

        inputs = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        lengths = torch.tensor([actual_length], device=self.device)

        outputs = self.model(inputs, lengths) / self.temperature
        probabilities = torch.softmax(outputs, dim=1)

        scores, indices = probabilities.topk(3, dim=1)

        top3 = [
            (self.idx_to_class[index.item()], score.item() * 100)
            for index, score in zip(indices[0], scores[0])
        ]

        return {
            "label": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
        }