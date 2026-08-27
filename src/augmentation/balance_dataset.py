# src/augmentation/balance_dataset.py
"""Per-split augmentation planning.
"""
from dataclasses import dataclass, asdict
import csv
import numpy as np

TRAIN_TARGET_MIN = 170
TRAIN_TARGET_MAX = 180
TRAIN_TARGET = 175  # preferred aim point within [MIN, MAX]

# val/test raised from 25-30 -> 37-38 so the FINAL per-class ratio
# (train=175, val=38, test=37, total=250) approximates 70:15:15
# (69.5%/15.2%/14.8% w/ val=38, or symmetric 37/38 split -- see note below).
# Since val/test are original-only (never augmented), hitting this target
# requires enough originals set aside at split time -- see rebalance_splits.py,
# which moves spare train originals into val/test when a class is short.
VAL_TARGET_MIN = 37
VAL_TARGET_MAX = 38

TEST_TARGET_MIN = 37
TEST_TARGET_MAX = 38


@dataclass
class SplitCounts:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


@dataclass
class ClassBalancePlan:
    class_name: str
    real: SplitCounts
    augment: SplitCounts
    final: SplitCounts
    needs_augmentation: bool
    note: str


def plan_class_balancing(class_name: str, real: SplitCounts) -> ClassBalancePlan:
    """Plans augmentation counts for a class.

    Only `train` is ever augmented. If a class already has >= TRAIN_TARGET_MIN
    original training samples, no augmentation is added (and nothing is ever
    removed -- originals are always preserved). Val and test are passed
    through untouched: they contain only original data, per the "no
    augmentation on validation or test" requirement.
    """
    if real.train >= TRAIN_TARGET_MIN:
        aug_train = 0
    else:
        aug_train = TRAIN_TARGET - real.train

    # Val/test are never augmented -- originals only.
    aug_val = 0
    aug_test = 0

    augment = SplitCounts(aug_train, aug_val, aug_test)
    final = SplitCounts(
        train=real.train + aug_train,
        val=real.val,
        test=real.test,
    )

    # Soft/contextual checks -- this function may be called with only one
    # split's real count populated (see generate_augmented.py, which calls
    # it once per split), so we only validate the split(s) actually present
    # rather than asserting a fixed global total.
    warnings = []
    if real.train > 0 and not (TRAIN_TARGET_MIN <= final.train <= TRAIN_TARGET_MAX + 5):
        warnings.append(
            f"train final={final.train} outside expected "
            f"[{TRAIN_TARGET_MIN}, {TRAIN_TARGET_MAX}] range"
        )
    if real.val > 0 and not (VAL_TARGET_MIN <= real.val <= VAL_TARGET_MAX):
        warnings.append(f"val real={real.val} outside expected [{VAL_TARGET_MIN}, {VAL_TARGET_MAX}] range")
    if real.test > 0 and not (TEST_TARGET_MIN <= real.test <= TEST_TARGET_MAX):
        warnings.append(f"test real={real.test} outside expected [{TEST_TARGET_MIN}, {TEST_TARGET_MAX}] range")

    needs_augmentation = augment.total > 0
    base_note = (
        f"real_total={real.total}, already meets train target"
        if not needs_augmentation
        else f"real_total={real.total}, augmenting train {real.train} -> {final.train}"
    )
    note = base_note if not warnings else base_note + " | " + "; ".join(warnings)

    return ClassBalancePlan(class_name, real, augment, final, needs_augmentation, note)


def plan_balancing(class_counts: dict) -> list:
    return [
        plan_class_balancing(name, SplitCounts(**counts))
        for name, counts in class_counts.items()
    ]


def save_augmentation_plan(plans: list, out_path: str) -> None:
    fieldnames = [
        "class_name",
        "real_train", "real_val", "real_test", "real_total",
        "augment_train", "augment_val", "augment_test", "augment_total",
        "final_train", "final_val", "final_test", "final_total",
        "needs_augmentation", "note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in plans:
            writer.writerow({
                "class_name": p.class_name,
                "real_train": p.real.train, "real_val": p.real.val,
                "real_test": p.real.test, "real_total": p.real.total,
                "augment_train": p.augment.train, "augment_val": p.augment.val,
                "augment_test": p.augment.test, "augment_total": p.augment.total,
                "final_train": p.final.train, "final_val": p.final.val,
                "final_test": p.final.test, "final_total": p.final.total,
                "needs_augmentation": p.needs_augmentation,
                "note": p.note,
            })


def resample_sequence(sequence, target_length: int):
    """Linear interpolation of a landmark sequence along the time axis.

    General-purpose helper (not Source-B-specific). Used by
    augment_transforms.py for the temporal speed-variation augmentation,
    which shortens/lengthens an individual training sample slightly.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    original_length = seq.shape[0]
    if original_length == target_length:
        return seq

    tail_shape = seq.shape[1:]
    flat = seq.reshape(original_length, -1)

    src_idx = np.linspace(0, original_length - 1, num=original_length)
    dst_idx = np.linspace(0, original_length - 1, num=target_length)

    resampled = np.empty((target_length, flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        resampled[:, i] = np.interp(dst_idx, src_idx, flat[:, i])

    return resampled.reshape((target_length,) + tail_shape)


if __name__ == "__main__":
    sample_class_counts = {
        "Play": {"train": 84, "val": 27, "test": 26},
        "Fever": {"train": 122, "val": 28, "test": 26},
        "Hospital": {"train": 175, "val": 26, "test": 27},
        "Book": {"train": 200, "val": 25, "test": 25},
    }
    for cn, counts in sample_class_counts.items():
        p = plan_class_balancing(cn, SplitCounts(**counts))
        print(cn, asdict(p))