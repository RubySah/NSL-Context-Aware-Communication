"""
Balance NSL landmark dataset by filling validation and test splits.

Rules:
1. Preserve every existing file in val and test.
2. Fill val and test to TARGET_PER_CLASS using random samples from train.
3. Move selected files from train (no duplication).
4. Do not remove excess files if a class already exceeds the target.
5. Ignore missing classes and empty folders.
6. No augmentation is performed by this script.

Run first with DRY_RUN = True.
Change it to False after checking the proposed operations.
"""

import random
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Ruby\projects\Major Project\NSL_Context_Aware_Communication"
)

DATASET_ROOT = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"

TARGET_PER_CLASS = 39

# Keep True for the first run.
# Change to False after reviewing the dry-run report.
DRY_RUN = False

# Fixed seed makes random selection reproducible.
RANDOM_SEED = 42

# Landmark file extensions.
# Add another extension here if your dataset uses it.
VALID_EXTENSIONS = {
    ".npy",
    ".npz",
    ".pt",
    ".pth",
}

# ============================================================
# HELPERS
# ============================================================

def get_landmark_files(folder):
    """Return supported landmark files recursively."""

    if not folder.exists():
        return []

    return sorted(
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def get_class_folders(split_dir):
    """Return existing class directories."""

    if not split_dir.exists():
        return {}

    return {
        folder.name: folder
        for folder in split_dir.iterdir()
        if folder.is_dir()
    }


def move_selected_files(files, destination, source_root):
    """Move selected files while preserving relative paths."""

    moved = 0

    for source_file in files:
        relative_path = source_file.relative_to(source_root)
        destination_file = destination / relative_path

        destination_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if destination_file.exists():
            print(f"WARNING: Destination exists, skipping: {destination_file}")
            continue

        if not DRY_RUN:
            shutil.move(
                str(source_file),
                str(destination_file)
            )

        moved += 1

    return moved


# ============================================================
# MAIN BALANCING FUNCTION
# ============================================================

def balance_dataset():

    random.seed(RANDOM_SEED)

    print("=" * 75)
    print("NSL DATASET BALANCING")
    print("=" * 75)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Target per validation/test class: {TARGET_PER_CLASS}")
    print(f"Dry run: {DRY_RUN}")
    print(f"Random seed: {RANDOM_SEED}")

    # Validate directories
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Required directory not found: {directory}"
            )

    train_classes = get_class_folders(TRAIN_DIR)
    val_classes = get_class_folders(VAL_DIR)
    test_classes = get_class_folders(TEST_DIR)

    # Use only classes that actually exist in the dataset.
    all_classes = sorted(
        set(train_classes)
        | set(val_classes)
        | set(test_classes)
    )

    if not all_classes:
        print("No class folders found.")
        return

    print(f"\nClasses found: {len(all_classes)}")

    # Statistics
    report = []
    total_moved_to_val = 0
    total_moved_to_test = 0

    insufficient_classes = []

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_name in all_classes:

        print("\n" + "-" * 75)
        print(f"CLASS: {class_name}")
        print("-" * 75)

        train_class = train_classes.get(
            class_name,
            TRAIN_DIR / class_name
        )

        val_class = val_classes.get(
            class_name,
            VAL_DIR / class_name
        )

        test_class = test_classes.get(
            class_name,
            TEST_DIR / class_name
        )

        train_files = get_landmark_files(train_class)
        val_files = get_landmark_files(val_class)
        test_files = get_landmark_files(test_class)

        original_train = len(train_files)
        original_val = len(val_files)
        original_test = len(test_files)

        print(f"Before balancing:")
        print(f"  Train: {original_train}")
        print(f"  Val:   {original_val}")
        print(f"  Test:  {original_test}")

        # Calculate the number of samples required.
        val_needed = max(
            0,
            TARGET_PER_CLASS - original_val
        )

        test_needed = max(
            0,
            TARGET_PER_CLASS - original_test
        )

        total_needed = val_needed + test_needed

        # Check available training samples.
        if len(train_files) < total_needed:

            insufficient_classes.append(class_name)

            print("\nWARNING: Not enough training samples!")
            print(f"  Required: {total_needed}")
            print(f"  Available: {len(train_files)}")

            # Do not partially modify this class.
            report.append({
                "class": class_name,
                "train_before": original_train,
                "val_before": original_val,
                "test_before": original_test,
                "val_moved": 0,
                "test_moved": 0,
                "status": "INSUFFICIENT TRAIN SAMPLES",
            })

            continue

        # Randomly select files without replacement.
        random.shuffle(train_files)

        selected_for_val = train_files[:val_needed]

        selected_for_test = train_files[
            val_needed:val_needed + test_needed
        ]

        # Move validation samples.
        moved_val = move_selected_files(
            selected_for_val,
            val_class,
            train_class
        )

        # Move test samples.
        moved_test = move_selected_files(
            selected_for_test,
            test_class,
            train_class
        )

        total_moved_to_val += moved_val
        total_moved_to_test += moved_test

        # Calculate resulting counts.
        final_train = original_train - moved_val - moved_test
        final_val = original_val + moved_val
        final_test = original_test + moved_test

        print("\nSelected:")
        print(f"  To Val:  {moved_val}")
        print(f"  To Test: {moved_test}")

        print("\nExpected after balancing:")
        print(f"  Train: {final_train}")
        print(f"  Val:   {final_val}")
        print(f"  Test:  {final_test}")

        report.append({
            "class": class_name,
            "train_before": original_train,
            "val_before": original_val,
            "test_before": original_test,
            "val_moved": moved_val,
            "test_moved": moved_test,
            "train_after": final_train,
            "val_after": final_val,
            "test_after": final_test,
            "status": "OK",
        })

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("BALANCING SUMMARY")
    print("=" * 75)

    print(f"Classes processed: {len(report)}")
    print(f"Files moved to validation: {total_moved_to_val}")
    print(f"Files moved to testing: {total_moved_to_test}")
    print(f"Total files moved: {total_moved_to_val + total_moved_to_test}")

    if insufficient_classes:
        print("\nClasses with insufficient training samples:")
        for class_name in insufficient_classes:
            print(f"  - {class_name}")

    print("\n" + "=" * 75)

    if DRY_RUN:
        print("DRY RUN COMPLETE - NO FILES WERE MOVED.")
        print("Review the output, then set DRY_RUN = False.")
    else:
        print("BALANCING COMPLETE.")
        print("Existing validation and test files were preserved.")

    print("=" * 75)

    # Print final class-wise table.
    print("\nCLASS-WISE SUMMARY")
    print("-" * 75)
    print(
        f"{'Class':<22}"
        f"{'Train':>9}"
        f"{'Val':>9}"
        f"{'Test':>9}"
        f"{'Status':>25}"
    )
    print("-" * 75)

    for item in report:
        train_count = item.get("train_after", item["train_before"])
        val_count = item.get(
            "val_after",
            item["val_before"]
        )
        test_count = item.get(
            "test_after",
            item["test_before"]
        )

        print(
            f"{item['class']:<22}"
            f"{train_count:>9}"
            f"{val_count:>9}"
            f"{test_count:>9}"
            f"{item['status']:>25}"
        )

    print("-" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    balance_dataset()
