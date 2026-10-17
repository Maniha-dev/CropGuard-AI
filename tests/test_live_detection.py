from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.camera import live_detection


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def test_state_sprays_after_three_confirmations_and_resets(monkeypatch):
    detection = SimpleNamespace(box=(1, 1, 20, 20), confidence=0.9)
    outcomes = iter([([detection], 0.9), ([detection], 0.9), ([detection], 0.9), ([], 0.0)])
    monkeypatch.setattr(live_detection, "detect_frame", lambda *_args: next(outcomes))
    state = live_detection.State()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    for _ in range(2):
        state.update(None, frame, 0.60)
        assert state.snapshot()[2] == "DON'T SPRAY"

    state.update(None, frame, 0.60)
    assert state.snapshot()[2:] == ("SPRAY", 3)

    state.update(None, frame, 0.60)
    assert state.snapshot()[2:] == ("DON'T SPRAY", 0)


@pytest.mark.parametrize("threaded", [False, True])
def test_run_processes_video_frames_with_fake_model(monkeypatch, threaded):
    capture = FakeCapture([np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)])
    detection = SimpleNamespace(box=(1, 1, 20, 20), confidence=0.9)
    calls = 0

    def fake_detect_frame(_model, _frame, _confidence):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return [detection], 0.9
        return [], 0.0

    monkeypatch.setattr(live_detection, "open_source", lambda _source: capture)
    monkeypatch.setattr(live_detection, "detect_frame", fake_detect_frame)

    result = live_detection.run(
        source="test.mp4",
        model=object(),
        show=False,
        threaded=threaded,
        max_frames=4,
    )

    assert result["frames"] == 4
    assert 1 <= result["model_runs"] <= 4
    assert all(decision in ("SPRAY", "DON'T SPRAY") for decision in result["decisions"])
    assert capture.released