"""CropGuard live program: camera -> trained model -> pest status -> spray decision.

Run from the repository root:
    python -m src.camera.live_detection
    python -m src.camera.live_detection --source 1
    python -m src.camera.live_detection --source http://192.168.1.5:8080/video
    python -m src.camera.live_detection --source test.mp4

Keys: Q quits; S saves the current frame under results/camera_test/.
Simulation only: no pump, relay, or hardware is controlled.
"""
import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.decision.spray_decision import REQUIRED_FRAMES, SprayDecision
from src.detection.detector import load_model
from src.detection.inference import detect_frame


GREEN, RED, WHITE = (0, 170, 0), (0, 0, 255), (255, 255, 255)


def open_source(source):
    """Open a camera index, stream URL, or video file."""
    if str(source).isdigit():
        index = int(source)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture = cv2.VideoCapture(index)
        return capture
    return cv2.VideoCapture(source)


class State:
    """Latest model result shared between the camera and model threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
        self.highest = 0.0
        self.decision = "DON'T SPRAY"
        self.engine = SprayDecision()
        self.processed = 0

    def update(self, model, frame, confidence_threshold):
        detections, highest = detect_frame(model, frame, confidence_threshold)
        with self.lock:
            self.detections = detections
            self.highest = highest
            self.decision = self.engine.update(bool(detections), highest)
            self.processed += 1

    def snapshot(self):
        with self.lock:
            return (
                list(self.detections),
                self.highest,
                self.decision,
                self.engine.consecutive_detections,
            )


def draw_overlay(frame, detections, highest, decision, confirmed, fps):
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Pest {detection.confidence:.2f}",
            (x1, max(y1 - 8, 95)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    spray = decision == "SPRAY"
    frame_width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (frame_width, 85), RED if spray else GREEN, -1)
    status = f"PEST DETECTED ({highest:.2f})" if detections else "NO PEST"
    cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)
    cv2.putText(frame, decision, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 3)

    frame_height = frame.shape[0]
    info = f"Confirmed {confirmed}/{REQUIRED_FRAMES}   FPS {fps:.0f}   Q=quit  S=save"
    cv2.putText(
        frame,
        info,
        (12, frame_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2,
    )
    return frame


def run(source="0", conf=0.60, model=None, show=True, threaded=True, max_frames=None):
    """Run camera or video inference and return a small run summary."""
    if model is None:
        model = load_model(require_trained=True)
        model(np.zeros((64, 64, 3), np.uint8), verbose=False)

    capture = open_source(source)
    if not capture.isOpened():
        print(
            f"ERROR: could not open source {source!r}. "
            "Close other apps using the camera, or try --source 1."
        )
        return None

    is_file = not str(source).isdigit() and not str(source).startswith(("http", "rtsp"))
    frame_rate = capture.get(cv2.CAP_PROP_FPS) or 15
    frame_delay = 1.0 / frame_rate if is_file else 0.0

    state = State()
    latest = {"frame": None, "stop": False}
    new_frame = threading.Event()

    def worker():
        while not latest["stop"]:
            if not new_frame.wait(timeout=0.2):
                continue
            new_frame.clear()
            frame = latest["frame"]
            if frame is not None:
                state.update(model, frame.copy(), conf)

    thread = None
    if threaded:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    print("CropGuard live started. Press Q in the video window to quit.")
    seen = []
    shown = 0
    last = time.time()
    fps = 0.0
    try:
        while True:
            if frame_delay:
                time.sleep(frame_delay)
            ok, frame = capture.read()
            if not ok:
                break

            if threaded:
                latest["frame"] = frame
                new_frame.set()
            else:
                state.update(model, frame, conf)

            detections, highest, decision, confirmed = state.snapshot()
            seen.append(decision)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            shown += 1

            if show:
                output_frame = draw_overlay(frame, detections, highest, decision, confirmed, fps)
                cv2.imshow("CropGuard - Live Detection", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    folder = Path("results/camera_test")
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"frame_{int(time.time())}.jpg"
                    saved = cv2.imwrite(str(path), output_frame)
                    print(f"Saved {path}" if saved else "ERROR: save failed")

            if max_frames and shown >= max_frames:
                break
    finally:
        latest["stop"] = True
        new_frame.set()
        if thread is not None:
            thread.join()
        capture.release()
        cv2.destroyAllWindows()

    return {"frames": shown, "model_runs": state.processed, "decisions": seen}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropGuard live pest detection")
    parser.add_argument("--source", default="0", help="camera index, URL, or video file")
    parser.add_argument("--conf", type=float, default=0.60, help="box confidence threshold")
    args = parser.parse_args()
    run(args.source, args.conf)