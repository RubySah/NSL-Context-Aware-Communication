"""Maintain and log recognized sign sequences."""

from pathlib import Path


class SentenceManager:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.words = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Each finished clip commits once, so repeated words are allowed.
    def commit(self, word):
        if not word:
            return False

        self.words.append(word)
        return True

    def finalize(self):
        if not self.words:
            return None

        sentence = " ".join(self.words)

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(sentence + "\n")

        self.words.clear()
        return sentence

    def reset(self):
        self.words.clear()

    def display(self, limit=5):
        return " ".join(self.words[-limit:])