# src/preprocessing/normalize_dataset.py
"""Converts raw 291-dim landmark clips in data/processed into body-relative features in data/final_dataset_norm."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))   # project root, so `src` imports work as a script

from src.preprocessing.features import normalize_sequence, NORM_DIM

PROJECT_ROOT = Path(
    r"D:\Ruby\projects\Major Project\NSL_Context_Aware_Communication"
)

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final_dataset_norm"

RAW_DIM = 291
SPLITS = ("train", "val", "test")


def normalize_file(src, dst):
    """Normalizes one clip; returns an error message, or None on success."""

    seq = np.load(src, allow_pickle=False)

    if seq.ndim != 2 or seq.shape[1] != RAW_DIM or len(seq) == 0:
        return f"bad shape {seq.shape}"

    out = normalize_sequence(seq)

    if out.shape[1] != NORM_DIM or not np.isfinite(out).all():
        return "non-finite or wrong-width output"

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)

    return None


def normalize_dataset():
    """Normalizes every split, keeping the split/class/file layout."""

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*.npy")):
        raise FileExistsError(
            f"{OUTPUT_DIR} already has clips. Delete it first so stale files are not mixed in."
        )

    problems = []

    print(f"{'Split':<8}{'Found':>8}{'Written':>10}{'Skipped':>10}")
    print("-" * 36)

    for split in SPLITS:
        files = sorted((INPUT_DIR / split).glob("*/*.npy"))
        written = 0

        for src in files:
            error = normalize_file(src, OUTPUT_DIR / src.relative_to(INPUT_DIR))

            if error:
                problems.append((src, error))
            else:
                written += 1

        print(f"{split:<8}{len(files):>8}{written:>10}{len(files) - written:>10}")

    for src, error in problems[:20]:
        print(f"  skipped {src.relative_to(INPUT_DIR)}: {error}")

    info = {"feature_dim": NORM_DIM, "raw_dim": RAW_DIM, "normalizer": "normalize_sequence"}

    with (OUTPUT_DIR / "feature_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print(f"\nFeature dim: {RAW_DIM} -> {NORM_DIM}. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    normalize_dataset()