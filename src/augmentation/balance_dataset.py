# src/augmentation/balance_dataset.py
"""Per-split augmentation planning and Source B temporal resampling."""
from dataclasses import dataclass, asdict
import csv
import numpy as np

MIN_TARGET = 280
MAX_TARGET = 300
TARGET_TOTAL = 290
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
SOURCE_B_TARGET_LENGTH = 40  # single fixed target, not a range


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


def _desired_split_counts(total: int) -> SplitCounts:
    """Each split's target is rounded independently from SPLIT_RATIO —
    no remainder-absorption into train, so the ratio holds per split."""
    return SplitCounts(
        train=round(total * SPLIT_RATIO["train"]),
        val=round(total * SPLIT_RATIO["val"]),
        test=round(total * SPLIT_RATIO["test"]),
    )


def plan_class_balancing(class_name: str, real: SplitCounts) -> ClassBalancePlan:
    desired = _desired_split_counts(TARGET_TOTAL)

    aug_train = max(0, desired.train - real.train)
    aug_val = max(0, desired.val - real.val)
    aug_test = max(0, desired.test - real.test)

    augment = SplitCounts(aug_train, aug_val, aug_test)
    final = SplitCounts(
        train=real.train + aug_train,
        val=real.val + aug_val,
        test=real.test + aug_test,
    )

    if not (MIN_TARGET <= final.total <= MAX_TARGET):
        raise AssertionError(
            f"{class_name}: final total {final.total} outside "
            f"[{MIN_TARGET}, {MAX_TARGET}]"
        )

    needs_augmentation = augment.total > 0
    note = (
        f"real_total={real.total}, already meets target"
        if not needs_augmentation
        else f"real_total={real.total}, augmenting toward "
             f"{desired.train}/{desired.val}/{desired.test}"
    )

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
    """Linear interpolation of a landmark sequence along the time axis."""
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


def resample_source_b_sequence(sequence, target_length: int = SOURCE_B_TARGET_LENGTH):
    """Resamples Source B's 25-frame landmark sequences to ~40 frames,
    once at preprocessing time. Source A is left at its natural length —
    both feed the Transformer as variable-length, padded/masked at batch time."""
    return resample_sequence(sequence, target_length=target_length)


if __name__ == "__main__":
    sample_class_counts = {
        "Play": {"train": 84, "val": 18, "test": 18},
        "Fever": {"train": 122, "val": 26, "test": 26},
        "Hospital": {"train": 125, "val": 26, "test": 27},
        "Book": {"train": 100, "val": 0, "test": 0},
    }
    for cn, counts in sample_class_counts.items():
        p = plan_class_balancing(cn, SplitCounts(**counts))
        print(cn, asdict(p))

    fake_25 = np.random.rand(25, 291).astype("float32")
    resampled = resample_source_b_sequence(fake_25)
    print(f"\nSource B resample: 25 -> {resampled.shape[0]} frames, "
          f"feature dim preserved: {resampled.shape[1] == fake_25.shape[1]}")