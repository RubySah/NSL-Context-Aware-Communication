# src/augmentation/generate_augmented.py
"""Generates augmented landmark files to meet the per-split balancing targets."""

import os
import csv
import numpy as np
from collections import defaultdict

from balance_dataset import plan_class_balancing, SplitCounts
from augment_transforms import AUGMENTATIONS

SPLITS = ("train", "val", "test")
AUGMENTATIONS_ORDER = sorted(AUGMENTATIONS.keys())  # fixed order so rng.choice is deterministic


def load_manifest(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def real_counts_by_class(rows: list) -> dict:
    counts = defaultdict(int)
    for row in rows:
        counts[row["class_name"]] += 1
    return counts


def generate_for_split(split: str, data_root: str, rng: np.random.Generator) -> list:
    manifest_path = os.path.join(data_root, "splits", f"{split}_manifest.csv")
    rows = load_manifest(manifest_path)
    counts = real_counts_by_class(rows)

    samples_by_class = defaultdict(list)
    for row in rows:
        samples_by_class[row["class_name"]].append(row)

    summary_rows = []
    for class_name, count in counts.items():
        split_counts = {"train": 0, "val": 0, "test": 0}
        split_counts[split] = count
        plan = plan_class_balancing(class_name, SplitCounts(
            train=split_counts["train"], val=split_counts["val"], test=split_counts["test"]
        ))
        n_needed = getattr(plan.augment, "val" if split == "val" else split)

        if n_needed == 0:
            continue

        # Written directly into data/processed, merged with real samples in the
        # same per-class folder -- not a separate data/augmented tree.
        out_dir = os.path.join(data_root, "processed", split, class_name)
        os.makedirs(out_dir, exist_ok=True)
        real_samples = samples_by_class[class_name]

        for i in range(n_needed):
            parent = real_samples[i % len(real_samples)]
            parent_path = os.path.join(
                data_root, "processed", split, class_name, f"{parent['sample_id']}.npy"
            )

            seed = rng.integers(0, 2**31 - 1)
            local_rng = np.random.default_rng(seed)
            aug_type = local_rng.choice(list(AUGMENTATIONS_ORDER))

            aug_id = f"aug_{class_name}_{i}"
            out_path = os.path.join(out_dir, f"{aug_id}.npy")

            if os.path.exists(out_path):
                augmented_seq = np.load(out_path)
            else:
                seq = np.load(parent_path)
                augmented_seq = AUGMENTATIONS[aug_type](seq, local_rng)
                np.save(out_path, augmented_seq)

            summary_rows.append({
                "sample_id": aug_id,
                "class_name": class_name,
                "split": split,
                "parent_sample_id": parent["sample_id"],
                "augmentation_type": aug_type,
                "augmentation_seed": int(seed),
                "processed_sequence_length": augmented_seq.shape[0],
                "is_augmented": True,
            })

    return summary_rows


def main(data_root: str = "data", seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    all_rows = []
    for split in SPLITS:
        rows = generate_for_split(split, data_root, rng)
        all_rows += rows
        print(f"{split}: generated {len(rows)} augmented samples")

    summary_path = os.path.join(data_root, "metadata", "augmentation_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()