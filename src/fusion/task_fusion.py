# src/fusion/task_fusion.py
"""Task-oriented, confidence-based fusion of gesture and scene evidence."""

import json
import os
import time
from collections import Counter

from src.fusion.task_domains import SIGN_DOMAINS, SCENE_DOMAINS, OBJECT_DOMAINS

class TaskFusion:
    def __init__(self, config):
        settings = config["fusion"]
        self.high = settings["high_confidence"]
        self.low = settings["low_confidence"]
        self.min_scene = settings["min_scene_confidence"]
        self.json_path = config["paths"]["fusion_json"]

    def _level(self, confidence):
        if confidence >= self.high:
            return "high"
        return "medium" if confidence >= self.low else "low"

    def _word_record(self, entry):
        return {
            "word": entry["word"],
            "confidence": round(entry["confidence"], 2),
            "level": self._level(entry["confidence"]),
            "alternatives": [
                {"word": w, "confidence": round(c, 2)} for w, c in entry["alternatives"]
            ],
            "domain": SIGN_DOMAINS.get(entry["word"], "general"),
        }

    # Organize one finished sentence plus the current scene; no intent decision here.
    def fuse(self, entries, context):
        if not entries:
            return None

        words = [self._word_record(e) for e in entries]
        scores = [e["confidence"] for e in entries]
        level = self._level(min(scores))      # the weakest word sets the sentence level

        scene = context["environment"]
        scene_conf = context["environment_confidence"]
        scene_domain = SCENE_DOMAINS.get(scene)
        reliable = scene_domain is not None and scene_conf >= self.min_scene

        domains = Counter(w["domain"] for w in words if w["domain"] != "general")
        gesture_domain = domains.most_common(1)[0][0] if domains else "general"
        objects = list(context.get("visual_context", [])) if reliable else []
        object_domains = Counter(OBJECT_DOMAINS[o] for o in objects if o in OBJECT_DOMAINS)

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gesture": {
                "gloss": [w["word"] for w in words],
                "words": words,
                "mean_confidence": round(sum(scores) / len(scores), 2),
                "min_confidence": round(min(scores), 2),
                "level": level,
            },
            "context": {
                "environment": scene,
                "confidence": round(scene_conf, 2),
                "domain": scene_domain,
                "reliable": reliable,
            },
            "task": {
                "gesture_domain": gesture_domain,
                "domain_counts": dict(domains),
                "context_matches_gesture": (scene_domain == gesture_domain) if reliable else None,
                "context_needed": level == "low",
            },
        }

    # Write the JSON atomically so a reader never sees a half-written file.
    def save(self, fused):
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.json_path.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as file:
                json.dump(fused, file, indent=2)
            os.replace(temp, self.json_path)
        except OSError:
            pass