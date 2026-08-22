# src/mediapipe_extraction/extract_landmarks.py
"""MediaPipe Holistic extraction for Source A and Source B frame folders."""

import os
import re
import glob
import numpy as np
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
POSE_DIMS = 5   # x, y, z, visibility, presence
HAND_DIMS = 3   # x, y, z

FEATURE_DIM = (
    NUM_POSE_LANDMARKS * POSE_DIMS
    + NUM_HAND_LANDMARKS * HAND_DIMS
    + NUM_HAND_LANDMARKS * HAND_DIMS
)  # 165 + 63 + 63 = 291


def load_frames_from_video(video_path: str):
    """Fallback loader if raw video is ever used instead of pre-extracted frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    return frames, {"fps": fps, "frame_count": len(frames), "source": "video"}


def _numeric_sort_key(path: str):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else path


def load_frames_from_folder(folder_path: str, extensions=(".jpg", ".jpeg", ".png")):
    files = [
        f for f in glob.glob(os.path.join(folder_path, "*"))
        if f.lower().endswith(extensions)
    ]
    files.sort(key=_numeric_sort_key)

    if not files:
        raise IOError(f"No frame images found in: {folder_path}")

    frames = []
    for f in files:
        frame_bgr = cv2.imread(f)
        if frame_bgr is None:
            raise IOError(f"Could not read frame image: {f}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    return frames, {"fps": None, "frame_count": len(frames), "source": "folder"}


def _pose_array(results):
    if results.pose_landmarks is None:
        return np.zeros((NUM_POSE_LANDMARKS, POSE_DIMS), dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility, lm.presence]
         for lm in results.pose_landmarks.landmark],
        dtype=np.float32,
    )


def _hand_array(hand_landmarks):
    if hand_landmarks is None:
        return np.zeros((NUM_HAND_LANDMARKS, HAND_DIMS), dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )


def extract_landmark_sequence(frames, holistic=None) -> np.ndarray:
    """Runs Holistic over ordered RGB frames using one shared instance
    (preserves MediaPipe's internal tracking/ROI reuse). Missing pose/hand
    detections are zero-filled to keep sequence length intact. Face is unused."""
    owns_holistic = holistic is None
    if owns_holistic:
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    try:
        per_frame_features = []
        for frame_rgb in frames:
            results = holistic.process(frame_rgb)
            pose = _pose_array(results)
            left_hand = _hand_array(results.left_hand_landmarks)
            right_hand = _hand_array(results.right_hand_landmarks)
            per_frame_features.append(
                np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
            )
        return np.stack(per_frame_features, axis=0)
    finally:
        if owns_holistic:
            holistic.close()


def process_source_a_sample(folder_path: str) -> tuple:
    """Source A: variable-length frame folder -> landmark sequence, kept variable-length."""
    frames, meta = load_frames_from_folder(folder_path)
    return extract_landmark_sequence(frames), meta


def process_source_a_sample_from_video(video_path: str) -> tuple:
    frames, meta = load_frames_from_video(video_path)
    return extract_landmark_sequence(frames), meta


def process_source_b_sample(folder_path: str) -> tuple:
    """Source B: 25-frame folder -> landmark sequence (caller resamples to ~40)."""
    frames, meta = load_frames_from_folder(folder_path)
    return extract_landmark_sequence(frames), meta


if __name__ == "__main__":
    fake_a = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(40)]
    fake_b = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(25)]

    seq_a = extract_landmark_sequence(fake_a)
    seq_b = extract_landmark_sequence(fake_b)

    print(f"Source A: {len(fake_a)} frames -> {seq_a.shape}")
    print(f"Source B: {len(fake_b)} frames -> {seq_b.shape}")
    print(f"Feature dim {FEATURE_DIM} matches: "
          f"{seq_a.shape[1] == FEATURE_DIM and seq_b.shape[1] == FEATURE_DIM}")