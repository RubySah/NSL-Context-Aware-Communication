"""Generates augmented landmark samples for training only."""

import numpy as np

from pathlib import Path

from augment_transforms import apply_random_augmentation


PROJECT_ROOT = Path(
    r"D:\Ruby\projects\Major Project\NSL_Context_Aware_Communication"
)

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"

TARGET_PER_CLASS = 182
RANDOM_SEED = 42


def get_original_files(class_dir):
    """Returns original landmark files, excluding augmented samples."""

    return sorted(
        path for path in class_dir.glob("*.npy")
        if not path.stem.startswith("aug_")
    )


def augment_class(class_dir, rng):
    """Augments a class until it reaches the target sample count."""

    originals = get_original_files(class_dir)
    existing = list(class_dir.glob("*.npy"))

    needed = TARGET_PER_CLASS - len(existing)

    if needed <= 0 or not originals:
        return len(originals), 0, len(existing)

    generated = 0
    index = 1

    while generated < needed:
        source = rng.choice(originals)
        seq = np.load(source, allow_pickle=False)

        augmented, _ = apply_random_augmentation(seq, rng)

        output = class_dir / f"aug_{source.stem}_{index:05d}.npy"
        index += 1

        if output.exists():
            continue

        np.save(output, augmented.astype(np.float32))
        generated += 1

    final_count = len(list(class_dir.glob("*.npy")))

    return len(originals), generated, final_count


def generate_augmented_dataset():
    """Balances training classes through landmark augmentation."""

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(TRAIN_DIR)

    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    for class_dir in sorted(TRAIN_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        original, generated, final = augment_class(class_dir, rng)

        results.append((class_dir.name, original, generated, final))

    print("\nTRAINING AUGMENTATION SUMMARY")
    print("-" * 65)
    print(f"{'Class':<22}{'Original':>10}{'Generated':>12}{'Final':>10}")
    print("-" * 65)

    for name, original, generated, final in results:
        print(f"{name:<22}{original:>10}{generated:>12}{final:>10}")

    print("-" * 65)
    print(f"Total classes: {len(results)}")
    print(f"Total generated: {sum(r[2] for r in results)}")
    print(f"Total training samples: {sum(r[3] for r in results)}")


if __name__ == "__main__":
    generate_augmented_dataset()
