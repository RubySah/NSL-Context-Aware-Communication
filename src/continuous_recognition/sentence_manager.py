# src/continuous_recognition/sentence_manager.py
"""Maintain and log recognized sign sequences."""

from pathlib import Path


class SentenceManager:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.entries = []      # {"word", "confidence", "alternatives"} per committed sign
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def words(self):
        return [entry["word"] for entry in self.entries]

    # Each finished clip commits once, so repeated words are allowed.
    def commit(self, word, confidence=0.0, alternatives=None):
        if not word:
            return False

        self.entries.append({
            "word": word,
            "confidence": confidence,
            "alternatives": alternatives or [],
        })
        return True

    # Log the sentence and return its entries for fusion.
    def finalize(self):
        if not self.entries:
            return None

        entries, self.entries = self.entries, []

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(" ".join(e["word"] for e in entries) + "\n")

        return entries

    def reset(self):
        self.entries.clear()

    def display(self, limit=5):
        return " ".join(self.words[-limit:])