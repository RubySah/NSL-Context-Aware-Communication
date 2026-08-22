# src/dataset_loading/nsl_dataset.py
"""PyTorch Dataset for variable-length NSL landmark sequences, with padding/masking collate."""

import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NSLLandmarkDataset(Dataset):
    """Loads real (processed) + augmented landmark sequences for one split."""

    def __init__(self, data_root: str, split: str, class_mapping_path: str = None):
        self.data_root = data_root
        self.split = split

        class_mapping_path = class_mapping_path or os.path.join(
            data_root, "metadata", "class_mapping.json"
        )
        with open(class_mapping_path, encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        self.samples = []
        self._load_manifest(os.path.join(data_root, "splits", f"{split}_manifest.csv"),
                            os.path.join(data_root, "processed", split))
        self._load_augmented(os.path.join(data_root, "metadata", "augmentation_summary.csv"),
                             os.path.join(data_root, "augmented", split))

    def _load_manifest(self, manifest_path: str, processed_dir: str) -> None:
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "path": os.path.join(processed_dir, row["class_name"], f"{row['sample_id']}.npy"),
                    "class_name": row["class_name"],
                })

    def _load_augmented(self, summary_path: str, augmented_dir: str) -> None:
        if not os.path.exists(summary_path):
            return
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != self.split:
                    continue
                self.samples.append({
                    "path": os.path.join(augmented_dir, row["class_name"], f"{row['sample_id']}.npy"),
                    "class_name": row["class_name"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        landmarks = np.load(sample["path"])
        label = self.class_mapping[sample["class_name"]]
        return torch.from_numpy(landmarks).float(), label


def collate_variable_length(batch):
    """Pads a batch of (seq_len, 291) tensors to the batch's max length,
    returning (padded_sequences, attention_mask, labels)."""
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)
    feature_dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq
        mask[i, :seq.shape[0]] = True

    return padded, mask, torch.tensor(labels, dtype=torch.long)


def make_dataloader(data_root: str, split: str, batch_size: int = 16,
                     shuffle: bool = None) -> DataLoader:
    dataset = NSLLandmarkDataset(data_root, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else (split == "train"),
        collate_fn=collate_variable_length,
    )


if __name__ == "__main__":
    loader = make_dataloader(data_root="data", split="train", batch_size=4)
    padded, mask, labels = next(iter(loader))
    print(f"padded shape: {tuple(padded.shape)}")
    print(f"mask shape:   {tuple(mask.shape)}")
    print(f"labels:       {labels.tolist()}")
    print(f"lengths from mask: {mask.sum(dim=1).tolist()}")
