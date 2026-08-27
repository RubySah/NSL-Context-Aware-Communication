# src/augmentation/rebalance_splits.py
"""Moves original samples from train into val/test where needed to hit the
70:15:15-ish final ratio (train~175, val~37-38, test~37-38).
"""

import os
import csv
import shutil
from collections import defaultdict

import numpy as np

from balance_dataset import VAL_TARGET_MIN, VAL_TARGET_MAX, TEST_TARGET_MIN, TEST_TARGET_MAX

SEED = 42
VAL_TARGET = (VAL_TARGET_MIN + VAL_TARGET_MAX) // 2   # 37
TEST_TARGET = (TEST_TARGET_MIN + TEST_TARGET_MAX) // 2  # 37


def load_manifest(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(path: str, rows: list, fieldnames: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _by_class(rows: list) -> dict:
    d = defaultdict(list)
    for r in rows:
        d[r["class_name"]].append(r)
    return d


def _move_npy(data_root: str, class_name: str, sample_id: str, from_split: str, to_split: str) -> bool:
    src = os.path.join(data_root, "processed", from_split, class_name, f"{sample_id}.npy")
    dst_dir = os.path.join(data_root, "processed", to_split, class_name)
    dst = os.path.join(dst_dir, f"{sample_id}.npy")
    if not os.path.exists(src):
        print(f"WARNING: expected original file missing, could not move: {src}")
        return False
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(src, dst)
    return True


def rebalance_val_test_from_train(data_root: str = "data", seed: int = SEED) -> list:
    rng = np.random.default_rng(seed)
    splits_dir = os.path.join(data_root, "splits")
    train_path = os.path.join(splits_dir, "train_manifest.csv")
    val_path = os.path.join(splits_dir, "val_manifest.csv")
    test_path = os.path.join(splits_dir, "test_manifest.csv")

    train_rows = load_manifest(train_path)
    val_rows = load_manifest(val_path)
    test_rows = load_manifest(test_path)

    fieldnames = list((train_rows or val_rows or test_rows)[0].keys())

    train_by_class = _by_class(train_rows)
    val_by_class = _by_class(val_rows)
    test_by_class = _by_class(test_rows)

    all_classes = sorted(set(train_by_class) | set(val_by_class) | set(test_by_class))
    moved_log = []

    for class_name in all_classes:
        val_have = len(val_by_class[class_name])
        test_have = len(test_by_class[class_name])
        val_needed = max(0, VAL_TARGET - val_have)
        test_needed = max(0, TEST_TARGET - test_have)
        total_needed = val_needed + test_needed

        if total_needed == 0:
            continue

        train_pool = train_by_class[class_name]
        available = len(train_pool)

        if available == 0:
            print(f"WARNING: {class_name} has no train originals to donate; "
                  f"val short by {val_needed}, test short by {test_needed}")
            continue

        n_to_move = min(total_needed, available)
        if n_to_move < total_needed:
            print(f"WARNING: {class_name} can only donate {n_to_move}/{total_needed} "
                  f"train originals toward val/test targets (val short "
                  f"{max(0, val_needed - n_to_move)}, test short "
                  f"{max(0, test_needed - (n_to_move - val_needed))} after donation)")

        chosen_idx = rng.choice(available, size=n_to_move, replace=False)
        chosen = [train_pool[i] for i in sorted(chosen_idx)]
        chosen_ids = {r["sample_id"] for r in chosen}

        # Remove chosen rows from this class's train pool.
        train_by_class[class_name] = [r for r in train_pool if r["sample_id"] not in chosen_ids]

        n_val_move = min(val_needed, len(chosen))
        val_moved, test_moved = chosen[:n_val_move], chosen[n_val_move:n_val_move + test_needed]

        for r in val_moved:
            if _move_npy(data_root, class_name, r["sample_id"], "train", "val"):
                r = dict(r)
                r["split"] = "val"
                val_by_class[class_name].append(r)
                moved_log.append((class_name, r["sample_id"], "train", "val"))

        for r in test_moved:
            if _move_npy(data_root, class_name, r["sample_id"], "train", "test"):
                r = dict(r)
                r["split"] = "test"
                test_by_class[class_name].append(r)
                moved_log.append((class_name, r["sample_id"], "train", "test"))

    new_train = [r for rows in train_by_class.values() for r in rows]
    new_val = [r for rows in val_by_class.values() for r in rows]
    new_test = [r for rows in test_by_class.values() for r in rows]

    save_manifest(train_path, new_train, fieldnames)
    save_manifest(val_path, new_val, fieldnames)
    save_manifest(test_path, new_test, fieldnames)

    print(f"rebalance: moved {len(moved_log)} original samples from train "
          f"into val/test across {len(all_classes)} classes")
    return moved_log


if __name__ == "__main__":
    rebalance_val_test_from_train()
