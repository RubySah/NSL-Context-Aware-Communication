"""Load the trained Transformer and predict isolated signs."""

import json

import numpy as np
import torch

from src.training.model import SignLanguageTransformer


class SignPredictor:
    def __init__(self, config):
        self.device = config["device"]
        self.fixed_length = config["project"]["fixed_length"]
        self.feature_dim = config["project"]["feature_dim"]
        self.num_classes = config["project"]["num_classes"]

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

        model_config = config["model"]

        self.model = SignLanguageTransformer(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            max_seq_len=self.fixed_length,
            **model_config,
        ).to(self.device)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.ndim != 2 or sequence.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected (frames, {self.feature_dim}), got {sequence.shape}."
            )

        if len(sequence) == 0:
            return None

        sequence = sequence[:self.fixed_length]
        actual_length = len(sequence)

        padded = np.zeros(
            (self.fixed_length, self.feature_dim),
            dtype=np.float32,
        )
        padded[:actual_length] = sequence

        inputs = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        lengths = torch.tensor([actual_length], device=self.device)

        outputs = self.model(inputs, lengths)
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