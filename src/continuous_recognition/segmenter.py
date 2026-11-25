"""Cut the live landmark stream into one clip per sign."""

import time
from collections import deque

import numpy as np

LEFT_HAND = slice(165, 228)
RIGHT_HAND = slice(228, 291)
WRIST_Y = (166, 229)         # y of landmark 0 in the left and right hand blocks
HIP_Y = (116, 121)           # pose landmarks 23 and 24
HIP_VISIBILITY = (118, 123)


class SignSegmenter:
    # Idle -> signing when raised hands start moving; signing -> idle after a pause.
    # Returns one complete clip per sign, like the whole-sign clips used in training.

    def __init__(self, config):
        rec, mot = config["recognition"], config["motion"]
        self.min_frames = rec["min_frames"]
        self.max_frames = rec["max_frames"]
        self.tail_frames = rec["tail_frames"]
        self.preroll = deque(maxlen=rec["preroll_frames"])

        self.velocity_threshold = mot["velocity_threshold"]
        self.start_frames = mot["start_frames"]
        self.start_window = mot.get("start_window", 5)
        self.use_hip_line = mot.get("use_hip_line", False)
        self.smoothing = mot["smoothing"]
        self.require_raised = mot["require_raised_hands"]
        self.rest_line = mot["rest_line"]
        self.word_pause = mot["word_pause_seconds"]

        self.discarded = 0
        self.reset()

    # Full reset, used at startup and when the user clears everything.
    def reset(self):
        self._clear_segment()
        self.prev = None
        self.speed = 0.0
        self.seen = [False, False]      # debug: hands detected in the last frame
        self.raised = [False, False]    # debug: hands counted as raised
        self.last_motion_time = time.monotonic()

    # Drop the segment in progress but keep motion tracking state.
    def _clear_segment(self):
        self.signing = False
        self.frames = []
        self.preroll.clear()
        self.recent = deque(maxlen=self.start_window)   # moving flags of the last few frames
        self.last_motion_idx = 0

    @property
    def length(self):
        return len(self.frames)

    # Seconds since any raised hand last moved.
    def idle_for(self, now):
        return now - self.last_motion_time

    # Frames collected so far, for the live preview.
    def current(self):
        return np.asarray(self.frames, dtype=np.float32)

    # Which hands are visible and above the resting line.
    def _raised_hands(self, f):
        hips_visible = f[HIP_VISIBILITY[0]] >= 0.5 and f[HIP_VISIBILITY[1]] >= 0.5
        use_hips = self.use_hip_line and hips_visible
        rest_y = (f[HIP_Y[0]] + f[HIP_Y[1]]) / 2 if use_hips else self.rest_line

        flags, seen = [], []
        for hand, wrist_y in ((LEFT_HAND, WRIST_Y[0]), (RIGHT_HAND, WRIST_Y[1])):
            present = bool(f[hand].any())
            seen.append(present)
            flags.append(present and (not self.require_raised or f[wrist_y] < rest_y))
        self.seen = seen
        return flags

    # Smoothed x/y speed of raised hands; ignores hands that just appeared.
    def _speed(self, f, raised):
        raw = 0.0
        if self.prev is not None:
            for hand, is_raised in zip((LEFT_HAND, RIGHT_HAND), raised):
                if is_raised and self.prev[hand].any():
                    now_xy = f[hand].reshape(21, 3)[:, :2]
                    old_xy = self.prev[hand].reshape(21, 3)[:, :2]
                    raw = max(raw, float(np.abs(now_xy - old_xy).mean()))

        self.speed = self.smoothing * raw + (1 - self.smoothing) * self.speed
        return self.speed

    # Feed one frame; returns a finished clip (frames, 291) or None.
    def update(self, features, now):
        raised = self._raised_hands(features)
        self.raised = raised
        moving = any(raised) and self._speed(features, raised) > self.velocity_threshold
        self.prev = features

        if moving:
            self.last_motion_time = now

        if not self.signing:
            self.preroll.append(features)
            self.recent.append(moving)
            # Start when enough of the last few frames were moving; one bad frame won't reset it.
            if sum(self.recent) >= self.start_frames:
                self.signing = True
                self.frames = list(self.preroll)
                self.last_motion_idx = len(self.frames) - 1
            return None

        self.frames.append(features)
        if moving:
            self.last_motion_idx = len(self.frames) - 1

        paused = now - self.last_motion_time >= self.word_pause
        if paused or len(self.frames) >= self.max_frames:
            return self._finish()
        return None

    # Trim trailing stillness, drop clips that are too short, return the rest.
    def _finish(self):
        end = min(len(self.frames), self.last_motion_idx + 1 + self.tail_frames)
        segment = np.asarray(self.frames[:end], dtype=np.float32)
        self._clear_segment()

        if len(segment) < self.min_frames:
            self.discarded += 1
            return None
        return segment


# Accept a segment prediction only if it is confident and clearly ahead of the runner-up.
def accept_prediction(result, min_confidence, min_margin):
    if result is None:
        return None, "no prediction"

    top = result["top3"]
    confidence = top[0][1] / 100
    margin = confidence - top[1][1] / 100 if len(top) > 1 else confidence

    if confidence < min_confidence:
        return None, f"low confidence {confidence:.2f} ({top[0][0]})"
    if margin < min_margin:
        return None, f"ambiguous {top[0][0]}/{top[1][0]} margin {margin:.2f}"
    return top[0][0], f"{confidence:.2f}"