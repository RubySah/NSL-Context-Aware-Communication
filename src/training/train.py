# src/training (NSL gesture recognition — model training)

import os, json, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SignLanguageTransformer

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR = "/content/data/processed"
TRAIN_DIR, VAL_DIR, TEST_DIR = (os.path.join(DATA_DIR, s) for s in ("train", "val", "test"))
CLASS_MAPPING_PATH = os.path.join(DATA_DIR, "class_mapping.json")

SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/Major Project/models"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "final_model.pth")
HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.json")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "last_checkpoint.pth")
RESULTS_PATH = os.path.join(SAVE_DIR, "test_results.json")

# ── Hyperparameters ───────────────────────────────────────────
FIXED_LEN, FEAT_DIM, NUM_CLASSES = 100, 291, 52
BATCH_SIZE, MAX_EPOCHS = 32, 180
LEARNING_RATE, WEIGHT_DECAY, DROPOUT, LABEL_SMOOTHING = 0.0001, 0.001, 0.35, 0.1
SCHEDULER_PATIENCE, SCHEDULER_FACTOR = 5, 0.5
EARLY_STOPPING_PATIENCE, GRADIENT_CLIP = 15, 1.0
NUM_WORKERS, SEED = 2, 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("NSL SIGN LANGUAGE TRANSFORMER TRAINING")
print("=" * 70)
print(f"Device       : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
print(f"Data         : {DATA_DIR}\nBatch size   : {BATCH_SIZE}\nMax epochs   : {MAX_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}\nWeight decay : {WEIGHT_DECAY}\nDropout      : {DROPOUT}")
print(f"Label smooth : {LABEL_SMOOTHING}\nFixed length : {FIXED_LEN}")
print("=" * 70)

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Class mapping ─────────────────────────────────────────────
if not os.path.exists(CLASS_MAPPING_PATH):
    raise FileNotFoundError(f"class_mapping.json not found:\n{CLASS_MAPPING_PATH}")

with open(CLASS_MAPPING_PATH, "r") as f:
    sign_to_idx = {sign: int(idx) for sign, idx in json.load(f).items()}
idx_to_sign = {idx: sign for sign, idx in sign_to_idx.items()}

print(f"\nClasses found in class_mapping.json: {len(sign_to_idx)}")
if len(sign_to_idx) != NUM_CLASSES:
    raise ValueError(f"Expected {NUM_CLASSES} classes, but got {len(sign_to_idx)}.")


# ── Dataset ───────────────────────────────────────────────────
class NSLDataset(Dataset):
    """Loads landmark sequences from per-class .npy files, padding/truncating to a fixed length."""

    def __init__(self, root_dir, sign_to_idx, fixed_len=100):
        self.fixed_len = fixed_len
        self.samples = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found:\n{root_dir}")

        for sign_name in sorted(os.listdir(root_dir)):
            sign_dir = os.path.join(root_dir, sign_name)
            if not os.path.isdir(sign_dir):
                continue
            if sign_name not in sign_to_idx:
                print(f"Warning: '{sign_name}' not present in class_mapping.json. Skipping.")
                continue
            label = sign_to_idx[sign_name]
            self.samples += [
                (os.path.join(sign_dir, fn), label)
                for fn in sorted(os.listdir(sign_dir)) if fn.lower().endswith(".npy")
            ]

        if not self.samples:
            raise RuntimeError(f"No .npy samples found in:\n{root_dir}")
        print(f"Loaded {len(self.samples):,} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        sequence = np.load(filepath).astype(np.float32)

        if sequence.ndim != 2:
            raise ValueError(f"Invalid shape in {filepath}: {sequence.shape}. Expected (frames, 291).")
        if sequence.shape[1] != FEAT_DIM:
            raise ValueError(f"Invalid feature dimension in {filepath}: {sequence.shape}. Expected {FEAT_DIM}.")

        actual_length = min(sequence.shape[0], self.fixed_len)
        padded = np.zeros((self.fixed_len, FEAT_DIM), dtype=np.float32)
        padded[:actual_length] = sequence[:actual_length]

        return torch.from_numpy(padded), torch.tensor(actual_length, dtype=torch.long), torch.tensor(label, dtype=torch.long)


print("\nLoading datasets...")
train_dataset = NSLDataset(TRAIN_DIR, sign_to_idx, FIXED_LEN)
val_dataset = NSLDataset(VAL_DIR, sign_to_idx, FIXED_LEN)
test_dataset = NSLDataset(TEST_DIR, sign_to_idx, FIXED_LEN)
print(f"\nDataset sizes:\nTrain: {len(train_dataset):,}\nVal  : {len(val_dataset):,}\nTest : {len(test_dataset):,}")

loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

# ── Model, loss, optimizer, scheduler ─────────────────────────
model = SignLanguageTransformer(
    input_dim=FEAT_DIM, d_model=160, num_heads=4, num_layers=4, d_ff=256,
    num_classes=NUM_CLASSES, max_seq_len=FIXED_LEN, dropout=DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel:\n{model}\n\nTotal parameters    : {total_params:,}\nTrainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

history = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "learning_rate")}


# ── Train / eval loops ────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    """Runs one pass over `loader`; trains if `optimizer` is given, else evaluates."""
    model.train(optimizer is not None)
    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for sequences, lengths, labels in loader:
            sequences, lengths, labels = (t.to(DEVICE, non_blocking=True) for t in (sequences, lengths, labels))

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def save_checkpoint(path, **extra):
    torch.save({"model_state_dict": model.state_dict(), "class_mapping": sign_to_idx, **extra}, path)


# ── Training loop ─────────────────────────────────────────────
print("\n" + "=" * 70 + "\nSTARTING TRAINING\n" + "=" * 70)

best_val_loss, best_epoch, early_stop_counter, best_model_state = float("inf"), 0, 0, None

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = run_epoch(model, val_loader, criterion)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    for key, val in zip(history, (epoch, train_loss, val_loss, train_accuracy, val_accuracy, current_lr)):
        history[key].append(float(val))

    print(
        f"Epoch [{epoch:03d}/{MAX_EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}% | LR: {current_lr:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss, best_epoch, early_stop_counter = val_loss, epoch, 0
        best_model_state = copy.deepcopy(model.state_dict())
        save_checkpoint(
            BEST_MODEL_PATH, epoch=epoch, optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(), val_loss=val_loss, val_accuracy=val_accuracy,
            config=dict(fixed_len=FIXED_LEN, feature_dim=FEAT_DIM, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, label_smoothing=LABEL_SMOOTHING,
                        dropout=DROPOUT, gradient_clip=GRADIENT_CLIP),
        )
        print(f"  Best model saved (Val Loss: {val_loss:.4f})")
    else:
        early_stop_counter += 1

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    save_checkpoint(
        CHECKPOINT_PATH, epoch=epoch, optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), best_val_loss=best_val_loss, best_epoch=best_epoch, history=history,
    )

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}.\nBest epoch: {best_epoch}\nBest validation loss: {best_val_loss:.4f}")
        break

# ── Restore and save best/final models ────────────────────────
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nRestored best model from epoch {best_epoch}")

save_checkpoint(
    FINAL_MODEL_PATH,
    config=dict(fixed_len=FIXED_LEN, feature_dim=FEAT_DIM, num_classes=NUM_CLASSES,
                d_model=160, num_heads=4, num_layers=4, d_ff=256, dropout=DROPOUT),
)
print(f"Final model saved to:\n{FINAL_MODEL_PATH}")

# ── Test set evaluation ───────────────────────────────────────
print("\n" + "=" * 70 + "\nEVALUATING BEST MODEL ON TEST SET\n" + "=" * 70)
test_loss, test_accuracy = run_epoch(model, test_loader, criterion)
print(f"\nTest Loss    : {test_loss:.4f}\nTest Accuracy: {test_accuracy * 100:.2f}%")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "best_epoch": best_epoch, "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "num_train_samples": len(train_dataset), "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset), "num_classes": NUM_CLASSES,
    }, f, indent=4)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 70 + "\nTRAINING COMPLETE\n" + "=" * 70)
print(f"Best Epoch       : {best_epoch}\nBest Val Loss    : {best_val_loss:.4f}")
print(f"Test Loss        : {test_loss:.4f}\nTest Accuracy    : {test_accuracy * 100:.2f}%")
print(f"\nSaved files:\nBest model       : {BEST_MODEL_PATH}\nFinal model      : {FINAL_MODEL_PATH}")
print(f"Training history : {HISTORY_PATH}\nTest results     : {RESULTS_PATH}\nCheckpoint       : {CHECKPOINT_PATH}")
print("=" * 70)