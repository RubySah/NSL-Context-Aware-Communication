"""Train the NSL sign language Transformer on landmark sequences."""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import SignLanguageTransformer

# ── Config ───────────────────────────────────────────────────
DATA_PATH      = r"D:\Ruby\projects\Major Project\NSL_Context_Aware_Communication\data\final_dataset"
CLASS_MAP_PATH = r"D:\Ruby\projects\Major Project\NSL_Context_Aware_Communication\data\metadata\class_mapping.json"
CHECKPOINT     = r"models\gesture_transformer\best\best_model2.pth"
HISTORY_PATH   = r"models\gesture_transformer\best\train_history2.json"

FIXED_LEN   = 100
FEAT_DIM    = 291
BATCH_SIZE  = 32
EPOCHS      = 180
LR          = 0.0001
NUM_CLASSES = 52
PATIENCE    = 15
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")

# ── Class mapping ────────────────────────────────────────────
with open(CLASS_MAP_PATH, encoding="utf-8") as f:
    sign_to_idx = json.load(f)

idx_to_sign = {v: k for k, v in sign_to_idx.items()}

# ── Column indices of every landmark's x, y, z ───────────────
# Layout: 33 pose landmarks x 5 dims, then 21 left hand x 3, then 21 right hand x 3.
POSE_X  = np.arange(33) * 5
LHAND_X = 165 + np.arange(21) * 3
RHAND_X = 228 + np.arange(21) * 3
ALL_X = np.concatenate([POSE_X, LHAND_X, RHAND_X])
ALL_Y = ALL_X + 1
ALL_Z = ALL_X + 2


# ── Augmentation ─────────────────────────────────────────────
def augment(seq):
    """Speed change, scale/shift around frame center, and noise on detected landmarks only."""
    # Speed variation via nearest-frame resampling (no blending into zero-filled gaps)
    n = len(seq)
    new_n = int(np.clip(round(n * random.uniform(0.85, 1.15)), 2, FIXED_LEN))
    out = seq[np.linspace(0, n - 1, new_n).round().astype(int)].copy()

    # A landmark counts as detected if any of x, y, z is nonzero
    present = (out[:, ALL_X] != 0) | (out[:, ALL_Y] != 0) | (out[:, ALL_Z] != 0)

    # Scale around the frame center plus a small constant shift
    scale = random.uniform(0.95, 1.05)
    dx, dy = np.random.uniform(-0.02, 0.02, 2)
    for cols, shift in ((ALL_X, dx), (ALL_Y, dy)):
        moved = (out[:, cols] - 0.5) * scale + 0.5 + shift
        out[:, cols] = np.where(present, moved, 0.0)

    # Gaussian noise on detected x, y, z only (visibility/presence untouched)
    for cols in (ALL_X, ALL_Y, ALL_Z):
        noise = np.random.normal(0, 0.01, present.shape)
        out[:, cols] += (noise * present).astype(np.float32)

    return out


# ── Dataset ──────────────────────────────────────────────────
class NSLDataset(Dataset):
    """Caches all sequences in RAM; pads to FIXED_LEN and returns (seq, length, label)."""

    def __init__(self, split_name):
        self.train = split_name == "train"
        self.sequences, self.labels = [], []
        split_path = os.path.join(DATA_PATH, split_name)
        skipped = 0

        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            if class_name not in sign_to_idx or not os.path.isdir(class_path):
                continue

            for fname in sorted(os.listdir(class_path)):
                if not fname.endswith(".npy"):
                    continue
                try:
                    seq = np.load(os.path.join(class_path, fname)).astype(np.float32)
                except Exception as e:
                    print(f"  skipped {class_name}/{fname}: {e}")
                    skipped += 1
                    continue

                if seq.ndim != 2 or seq.shape[1] != FEAT_DIM or len(seq) == 0:
                    print(f"  skipped {class_name}/{fname}: bad shape {seq.shape}")
                    skipped += 1
                    continue

                # Center-crop sequences longer than FIXED_LEN
                if len(seq) > FIXED_LEN:
                    start = (len(seq) - FIXED_LEN) // 2
                    seq = seq[start:start + FIXED_LEN]

                self.sequences.append(seq)
                self.labels.append(sign_to_idx[class_name])

        print(f"  {split_name}: {len(self.sequences)} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.train:
            seq = augment(seq)

        length = min(len(seq), FIXED_LEN)
        padded = np.zeros((FIXED_LEN, FEAT_DIM), dtype=np.float32)
        padded[:length] = seq[:length]  # left-aligned, matches the model's mask

        return torch.from_numpy(padded), length, self.labels[idx]


# ── Loaders ──────────────────────────────────────────────────
print("\nLoading datasets...")
train_dataset = NSLDataset("train")
val_dataset   = NSLDataset("val")
test_dataset  = NSLDataset("test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Model, loss, optimizer ───────────────────────────────────
model = SignLanguageTransformer(
    input_dim=FEAT_DIM,
    num_classes=NUM_CLASSES,
    max_seq_len=FIXED_LEN,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── Evaluation ───────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader):
    """Returns sample-weighted loss, accuracy, predictions and labels."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for seqs, lengths, labels in loader:
        seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        outputs = model(seqs, lengths)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(labels), float((preds == labels).mean()), preds, labels


# ── Training loop ────────────────────────────────────────────
best_val_acc, best_val_loss = 0.0, float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("\nStarting training...")
print("=" * 70)

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, lengths, labels in train_loader:
        seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(seqs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss, train_acc = total_loss / total, correct / total
    val_loss, val_acc, _, _ = evaluate(val_loader)
    scheduler.step(val_loss)

    # One rule for both saving and early stopping: higher val acc, ties broken by lower val loss
    improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
    if improved:
        best_val_acc, best_val_loss = val_acc, val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")
    else:
        patience_counter += 1

    for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
        history[key].append(value)

    print(f"Epoch {epoch + 1:3d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if patience_counter >= PATIENCE:
        print(f"\n⏹ Early stopping at epoch {epoch + 1} "
              f"(no improvement for {PATIENCE} epochs)")
        break

with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f)

print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.4f}")

# ── Final test evaluation with the best checkpoint ───────────
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
test_loss, test_acc, preds, labels = evaluate(test_loader)

per_class = []
for i in range(NUM_CLASSES):
    mask = labels == i
    if mask.any():
        per_class.append((idx_to_sign[i], float((preds[mask] == i).mean()), int(mask.sum())))
per_class.sort(key=lambda item: item[1])

print("\n" + "=" * 70)
print(f"TEST  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | "
      f"Macro per-class Acc: {np.mean([a for _, a, _ in per_class]):.4f}")
print("\nWeakest 10 classes:")
for name, acc, n in per_class[:10]:
    print(f"  {name:<18}{acc:>7.2%}  (n={n})")