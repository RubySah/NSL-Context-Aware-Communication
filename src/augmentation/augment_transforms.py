"""Landmark-level augmentation transforms for 291-dim NSL sequences."""

import numpy as np

POSE_LANDMARKS, POSE_DIMS = 33, 5
HAND_LANDMARKS, HAND_DIMS = 21, 3

POSE_END = POSE_LANDMARKS * POSE_DIMS
LHAND_END = POSE_END + HAND_LANDMARKS * HAND_DIMS

POSE_X = np.arange(POSE_LANDMARKS) * POSE_DIMS
LHAND_X = POSE_END + np.arange(HAND_LANDMARKS) * HAND_DIMS
RHAND_X = LHAND_END + np.arange(HAND_LANDMARKS) * HAND_DIMS

ALL_X = np.concatenate([POSE_X, LHAND_X, RHAND_X])
ALL_Y = ALL_X + 1
ALL_Z = ALL_X + 2
ALL_XYZ = np.sort(np.concatenate([ALL_X, ALL_Y, ALL_Z]))


def resample_sequence(seq, target_length):
    """Resamples a sequence using linear interpolation."""

    positions = np.linspace(0, seq.shape[0] - 1, target_length)

    return np.stack([
        np.interp(
            positions,
            np.arange(seq.shape[0]),
            seq[:, i]
        )
        for i in range(seq.shape[1])
    ], axis=1).astype(np.float32)


def temporal_speed_variation(seq, rng, factor_range=(0.9, 1.1)):
    """Resamples to simulate variations in signing speed."""

    factor = rng.uniform(*factor_range)
    new_len = max(2, round(seq.shape[0] * factor))
    return resample_sequence(seq, target_length=new_len)


def temporal_frame_dropout(seq, rng, drop_prob=0.1):
    """Drops random frames while preserving their order."""

    keep_mask = rng.random(seq.shape[0]) > drop_prob

    if keep_mask.sum() < 2:
        keep_mask[rng.choice(seq.shape[0], 2, replace=False)] = True

    return seq[keep_mask]


def landmark_noise(seq, rng, sigma=0.01):
    """Adds Gaussian noise to x/y/z coordinates only."""

    out = seq.copy()
    out[:, ALL_XYZ] += rng.normal(
        0, sigma, size=(seq.shape[0], ALL_XYZ.size)
    )
    return out


def landmark_translation(seq, rng, max_shift=0.02):
    """Applies a constant random x/y shift across the sequence."""

    out = seq.copy()
    dx, dy = rng.uniform(-max_shift, max_shift, size=2)
    out[:, ALL_X] += dx
    out[:, ALL_Y] += dy
    return out


def landmark_scale(seq, rng, scale_range=(0.95, 1.05)):
    """Scales x/y coordinates around the normalized frame center."""

    out = seq.copy()
    s = rng.uniform(*scale_range)
    out[:, ALL_X] = (out[:, ALL_X] - 0.5) * s + 0.5
    out[:, ALL_Y] = (out[:, ALL_Y] - 0.5) * s + 0.5
    return out


def landmark_rotation(seq, rng, max_angle_deg=5.0):
    """Rotates x/y coordinates around the frame center."""

    out = seq.copy()
    theta = np.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x, y = out[:, ALL_X] - 0.5, out[:, ALL_Y] - 0.5

    out[:, ALL_X] = x * cos_t - y * sin_t + 0.5
    out[:, ALL_Y] = x * sin_t + y * cos_t + 0.5

    return out


AUGMENTATIONS = {
    "speed": temporal_speed_variation,
    "dropout": temporal_frame_dropout,
    "noise": landmark_noise,
    "translation": landmark_translation,
    "scale": landmark_scale,
    "rotation": landmark_rotation,
}


def apply_random_augmentation(seq, rng, name=None):
    """Applies one named or randomly selected augmentation."""

    name = name or rng.choice(list(AUGMENTATIONS))
    return AUGMENTATIONS[name](seq, rng), name
