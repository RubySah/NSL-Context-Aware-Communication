"""Real-time Nepali Sign Language recognition."""

import time

import cv2

from src.utils.config import load_config
from src.mediapipe_extraction.landmark_extractor import LandmarkExtractor
from src.mediapipe_extraction.landmark_visualizer import LandmarkVisualizer
from src.training.predictor import SignPredictor
from src.context.scene_detection import SceneDetector
from src.continuous_recognition.segmenter import SignSegmenter, accept_prediction
from src.continuous_recognition.sentence_manager import SentenceManager


# Draw one line of overlay text.
def put_text(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    config = load_config()
    rec = config["recognition"]

    extractor = LandmarkExtractor()
    visualizer = LandmarkVisualizer()
    predictor = SignPredictor(config)
    scene_detector = SceneDetector(config)
    segmenter = SignSegmenter(config)
    sentence = SentenceManager(config["paths"]["sentence_log"])

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        extractor.close()
        scene_detector.close()
        raise RuntimeError("Could not open camera.")

    frame_count = 0
    preview = ("--", 0.0)          # live guess while signing
    last_result = "waiting for a sign"

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_count += 1
            now = time.monotonic()

            scene, scene_score = scene_detector.update(frame)
            features, _, results = extractor.extract(frame)
            frame = visualizer.draw(frame, results)

            clip = segmenter.update(features, now)

            # Live preview on the partial clip: shown on screen, never committed.
            if (
                segmenter.signing
                and segmenter.length >= rec["min_frames"]
                and frame_count % rec["preview_interval"] == 0
            ):
                guess = predictor.predict(segmenter.current())
                if guess:
                    preview = (guess["label"], guess["confidence"])

            # A finished clip is classified once and committed only if clearly confident.
            if clip is not None:
                result = predictor.predict(clip)
                word, info = accept_prediction(
                    result, rec["accept_confidence"], rec["min_margin"]
                )

                top3 = ", ".join(f"{n} {s:.0f}%" for n, s in result["top3"]) if result else "-"
                print(f"clip of {len(clip)} frames -> {top3} -> "
                      f"{'COMMIT ' + word if word else 'rejected: ' + info}")

                if word:
                    sentence.commit(word)
                    last_result = f"{word} ({info})"
                else:
                    last_result = f"rejected: {info}"
                preview = ("--", 0.0)

            # A long pause with no motion ends the sentence.
            if (
                sentence.words
                and not segmenter.signing
                and segmenter.idle_for(now) >= config["motion"]["sentence_pause_seconds"]
            ):
                sentence.finalize()

            state = f"SIGNING ({segmenter.length} frames)" if segmenter.signing else "IDLE"
            state_color = (0, 165, 255) if segmenter.signing else (200, 200, 200)

            put_text(frame, f"State: {state}", 35, state_color, 0.7)
            put_text(frame, f"Guess: {preview[0]} ({preview[1]:.0f}%)", 70, (0, 255, 0))
            put_text(frame, f"Last: {last_result}", 105)
            put_text(frame, f"Scene: {scene} ({scene_score:.0f}%)", 140)
            put_text(frame, f"Sentence: {sentence.display()}", frame.shape[0] - 30,
                     (0, 255, 255), 0.65)

            cv2.imshow("NSL Real-Time Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                sentence.finalize()
                break

            if key == ord(" "):
                segmenter.reset()
                sentence.reset()
                preview = ("--", 0.0)
                last_result = "cleared"

    finally:
        camera.release()
        extractor.close()
        scene_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()