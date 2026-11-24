"""Real-time Nepali Sign Language recognition."""

import time

import cv2

from src.utils.config import load_config
from src.mediapipe_extraction.landmark_extractor import LandmarkExtractor
from src.mediapipe_extraction.landmark_visualizer import LandmarkVisualizer
from src.training.predictor import SignPredictor
from src.context.scene_detection import SceneDetector
from src.continuous_recognition.buffer import SequenceBuffer
from src.continuous_recognition.motion import MotionTracker
from src.continuous_recognition.sentence_manager import SentenceManager
from src.continuous_recognition.recognition import RecognitionManager


def main():
    config = load_config()

    extractor = LandmarkExtractor()
    visualizer = LandmarkVisualizer()
    predictor = SignPredictor(config)
    scene_detector = SceneDetector(config)

    settings = config["recognition"]

    buffer = SequenceBuffer(settings["max_frames"])
    motion = MotionTracker(
        config["motion"]["velocity_threshold"]
    )
    sentence = SentenceManager(
        config["paths"]["sentence_log"]
    )
    recognition = RecognitionManager(config)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        extractor.close()
        raise RuntimeError("Could not open camera.")

    frame_count = 0
    last_motion_time = time.monotonic()
    last_commit_time = 0.0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            frame_count += 1
            now = time.monotonic()

            scene, scene_score = scene_detector.update(frame)

            timestamp_ms = int(now * 1000)

            features, hand_detected, results = extractor.extract(frame)

            frame = visualizer.draw(frame, results)

            if hand_detected:
                buffer.add(features)

                if motion.is_moving(features):
                    last_motion_time = now
            else:
                motion.reset()

            stillness = now - last_motion_time

            if (
                len(buffer) >= settings["min_frames"]
                and recognition.should_check(
                    len(buffer), frame_count
                )
            ):
                result = predictor.predict(buffer.get())
                recognition.update(result, frame_count)

            if (
                recognition.should_commit(len(buffer))
                and (
                    stillness >= config["motion"]["word_pause_seconds"]
                    or len(buffer) >= settings["max_frames"]
                    or now - last_commit_time >= 1.0
                )
            ):
                word = recognition.stable_prediction()

                if sentence.commit(word):
                    last_commit_time = now

                buffer.clear()
                motion.reset()
                recognition.reset()

            if (
                sentence.words
                and stillness >= config["motion"]["sentence_pause_seconds"]
            ):
                sentence.finalize()
                buffer.clear()
                motion.reset()
                recognition.reset()

            cv2.putText(
                frame,
                f"Sign: {recognition.prediction or '--'}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {recognition.confidence:.1f}%",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Scene: {scene} ({scene_score:.1f}%)",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Sentence: {sentence.display()}",
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Buffer: {len(buffer)}/{settings['max_frames']}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("NSL Real-Time Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                sentence.finalize()
                break

            if key == ord(" "):
                buffer.clear()
                motion.reset()
                recognition.reset()
                sentence.reset()
                last_motion_time = time.monotonic()

    finally:
        camera.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
