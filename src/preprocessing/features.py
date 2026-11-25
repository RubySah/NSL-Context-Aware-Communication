# src/preprocessing/features.py
"""Body-relative feature normalization shared by training and live prediction."""

import numpy as np

UPPER_POSE = list(range(23))                 # face points 0-10, shoulders, elbows, wrists 11-16, pose hand points 17-22
L_SH, R_SH = 11, 12
DELTA_GAIN = 10.0                            # per-frame wrist motion is tiny, so scale it up
NORM_DIM = len(UPPER_POSE) * 3 + 2 * 62 + 2 + 4   # 199


# Shoulder-center origin and shoulder-width scale, from frames where both shoulders are visible.
def _anchor(pose):
    ok = (pose[:, L_SH, 3] > 0.5) & (pose[:, R_SH, 3] > 0.5)
    if not ok.any():
        return np.array([0.5, 0.5, 0.0], np.float32), 0.25
    left, right = pose[ok, L_SH, :3], pose[ok, R_SH, :3]
    origin = np.median((left + right) / 2, axis=0)
    width = np.median(np.linalg.norm(left[:, :2] - right[:, :2], axis=1))
    return origin, max(float(width), 1e-3)


# Body-relative wrist position plus wrist-relative hand shape; zeros when the hand is missing.
def _hand_features(hand, origin, scale):
    present = hand.reshape(len(hand), -1).any(axis=1)
    wrist = hand[:, 0, :]
    palm = np.linalg.norm(hand[:, 9, :2] - wrist[:, :2], axis=1).clip(min=1e-3)
    shape = (hand[:, 1:, :] - wrist[:, None, :]) / palm[:, None, None]
    pos = (wrist[:, :2] - origin[:2]) / scale
    feats = np.concatenate([pos, shape.reshape(len(hand), -1)], axis=1)
    feats[~present] = 0.0
    return feats.astype(np.float32), present, pos * present[:, None]


# (T, 291) raw landmarks -> (T, NORM_DIM) body-relative features.
def normalize_sequence(seq):
    seq = np.asarray(seq, np.float32)
    pose = seq[:, :165].reshape(-1, 33, 5)
    origin, scale = _anchor(pose)

    body = ((pose[:, UPPER_POSE, :3] - origin) / scale).reshape(len(seq), -1)
    body[~pose.reshape(len(seq), -1).any(axis=1)] = 0.0

    left, l_ok, l_pos = _hand_features(seq[:, 165:228].reshape(-1, 21, 3), origin, scale)
    right, r_ok, r_pos = _hand_features(seq[:, 228:].reshape(-1, 21, 3), origin, scale)

    # Wrist velocity, zeroed where a hand has just appeared or disappeared.
    wrists = np.concatenate([l_pos, r_pos], axis=1)
    delta = np.diff(wrists, axis=0, prepend=wrists[:1]) * DELTA_GAIN
    flags = np.stack([l_ok, r_ok], axis=1)
    both = flags & np.roll(flags, 1, axis=0)
    both[0] = False
    delta *= np.repeat(both, 2, axis=1)

    out = np.concatenate([body, left, right, flags.astype(np.float32), delta], axis=1)
    return out.astype(np.float32)