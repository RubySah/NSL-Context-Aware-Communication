# src/consolidation/combine_datasets.py
"""Combines Source A + Source B into per-split manifests and processed landmarks."""

import os
import sys
import json
import csv
import yaml
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mediapipe_extraction"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "augmentation"))

from extract_landmarks import process_source_a_sample, process_source_b_sample
from balance_dataset import resample_source_b_sequence

SPLITS = ("train", "val", "test")
EXCLUDED_CLASSES = {"Thank You"}


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path} (run this script from the project root)"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    profile = cfg["active_profile"]
    source_a_root = cfg["source_a"][profile]["root"]
    source_b_root = cfg["source_b"][profile]["root"]

    for name, root in [("source_a", source_a_root), ("source_b", source_b_root)]:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{name} root does not exist: {root}")

    return {"source_a_root": source_a_root, "source_b_root": source_b_root}


def discover_classes(source_a_root: str) -> dict:
    """Class mapping is derived from Source A's train folder, excluding EXCLUDED_CLASSES."""
    train_dir = os.path.join(source_a_root, "train")
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and d not in EXCLUDED_CLASSES
    )
    return {name: idx for idx, name in enumerate(classes)}


def process_class_split(source_root: str, split: str, class_name: str,
                         source_tag: str, out_dir: str) -> list:
    """Processes every sample folder for one (source, split, class), skipping
    samples already processed on disk so the run is resumable."""
    class_dir = os.path.join(source_root, split, class_name)
    if not os.path.isdir(class_dir):
        print(f"WARNING: missing folder, skipping -> {class_dir}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for sample_name in sorted(os.listdir(class_dir)):
        sample_path = os.path.join(class_dir, sample_name)
        if not os.path.isdir(sample_path):
            continue

        sample_id = f"{source_tag}_{class_name}_{sample_name}"
        out_path = os.path.join(out_dir, f"{sample_id}.npy")

        if os.path.exists(out_path):
            landmarks = np.load(out_path)
        else:
            if source_tag == "A":
                landmarks, _ = process_source_a_sample(sample_path)
            else:
                landmarks, _ = process_source_b_sample(sample_path)
                landmarks = resample_source_b_sequence(landmarks)
            np.save(out_path, landmarks)

        rows.append({
            "sample_id": sample_id,
            "class_name": class_name,
            "source_dataset": source_tag,
            "source_path": sample_path,
            "split": split,
            "signer_id": "",
            "processed_sequence_length": landmarks.shape[0],
            "is_augmented": False,
        })

    return rows


def combine(config_path: str, data_root: str) -> None:
    cfg = load_config(config_path)
    class_mapping = discover_classes(cfg["source_a_root"])

    metadata_dir = os.path.join(data_root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    for split in SPLITS:
        rows = []
        for class_name in class_mapping:
            out_dir = os.path.join(data_root, "processed", split, class_name)
            rows += process_class_split(cfg["source_a_root"], split, class_name, "A", out_dir)
            rows += process_class_split(cfg["source_b_root"], split, class_name, "B", out_dir)

        manifest_path = os.path.join(data_root, "splits", f"{split}_manifest.csv")
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

        print(f"{split}: {len(rows)} samples written to {manifest_path}")


if __name__ == "__main__":
    combine(config_path="configs/dataset_paths.yaml", data_root="data")