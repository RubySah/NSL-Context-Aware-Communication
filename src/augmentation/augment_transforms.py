# src/augmentation/augment_transforms.py
"""Landmark-level augmentation transforms for 291-dim NSL sequences."""

import numpy as np

POSE_LANDMARKS, POSE_DIMS = 33, 5
HAND_LANDMARKS, HAND_DIMS = 21, 3

POSE_END = POSE_LANDMARKS * POSE_DIMS
LHAND_END = POSE_END + HAND_LANDMARKS * HAND_DIMS
RHAND_END = LHAND_END + HAND_LANDMARKS * HAND_DIMS

POSE_X = np.arange(POSE_LANDMARKS) * POSE_DIMS
LHAND_X = POSE_END + np.arange(HAND_LANDMARKS) * HAND_DIMS
RHAND_X = LHAND_END + np.arange(HAND_LANDMARKS) * HAND_DIMS

ALL_X = np.concatenate([POSE_X, LHAND_X, RHAND_X])
ALL_Y = ALL_X + 1
ALL_Z = ALL_X + 2

LEFT_HAND = slice(POSE_END, LHAND_END)
RIGHT_HAND = slice(LHAND_END, RHAND_END)

ASPECT = 4 / 3   # frame width / height, so rotation is not skewed

# Scale and shift are not used: body-relative normalization cancels them.


def detected(seq):
    """Mask (frames, 75) of landmarks that were detected (any of x, y, z nonzero)."""

    return (seq[:, ALL_X] != 0) | (seq[:, ALL_Y] != 0) | (seq[:, ALL_Z] != 0)


def resample_sequence(seq, target_length):
    """Nearest-frame resampling; interpolating would blend hands with zero-filled gaps."""

    idx = np.linspace(0, seq.shape[0] - 1, target_length).round().astype(int)
    return seq[idx].copy()


def temporal_speed_variation(seq, rng, factor_range=(0.75, 1.25)):
    """Resamples to simulate variations in signing speed."""

    factor = rng.uniform(*factor_range)
    new_len = max(8, round(seq.shape[0] * factor))
    return resample_sequence(seq, target_length=new_len)


def temporal_crop(seq, rng, max_trim=0.15):
    """Trims up to 15% from the start and/or end, like an imperfect live cut."""

    n = seq.shape[0]
    cut = int(n * rng.uniform(0.0, max_trim))
    head = int(rng.integers(0, cut + 1))

    return seq[head:n - (cut - head)].copy()


def temporal_frame_dropout(seq, rng, drop_prob=0.1):
    """Drops random frames while preserving their order."""

    keep_mask = rng.random(seq.shape[0]) > drop_prob

    if keep_mask.sum() < 2:
        keep_mask[rng.choice(seq.shape[0], 2, replace=False)] = True

    return seq[keep_mask]


def landmark_noise(seq, rng, sigma=0.005):
    """Adds Gaussian noise to the x/y/z of detected landmarks only."""

    out = seq.copy()
    present = detected(out)

    for cols in (ALL_X, ALL_Y, ALL_Z):
        out[:, cols] += rng.normal(0, sigma, size=present.shape) * present

    return out


def landmark_rotation(seq, rng, max_angle_deg=10.0):
    """Rotates detected x/y coordinates around the frame center."""

    out = seq.copy()
    present = detected(out)

    theta = np.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x = (out[:, ALL_X] - 0.5) * ASPECT
    y = out[:, ALL_Y] - 0.5

    out[:, ALL_X] = np.where(present, (x * cos_t - y * sin_t) / ASPECT + 0.5, 0.0)
    out[:, ALL_Y] = np.where(present, x * sin_t + y * cos_t + 0.5, 0.0)

    return out


def hand_dropout(seq, rng):
    """Zeroes one hand for a short run of frames, like a missed detection."""

    out = seq.copy()
    n = out.shape[0]

    hand = LEFT_HAND if rng.random() < 0.5 else RIGHT_HAND
    run = min(n, int(rng.integers(3, max(4, n // 4 + 1))))
    start = int(rng.integers(0, n - run + 1))
    out[start:start + run, hand] = 0.0

    return out


AUGMENTATIONS = {
    "speed": temporal_speed_variation,
    "crop": temporal_crop,
    "dropout": temporal_frame_dropout,
    "noise": landmark_noise,
    "rotation": landmark_rotation,
    "hand_dropout": hand_dropout,
}


def apply_random_augmentation(seq, rng, name=None, n_ops=2):
    """Applies one named transform, or n_ops distinct random ones chained."""

    names = [name] if name else list(rng.choice(list(AUGMENTATIONS), size=n_ops, replace=False))

    for op in names:
        seq = AUGMENTATIONS[op](seq, rng)

    return seq.astype(np.float32), "+".join(names)