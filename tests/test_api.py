import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.server import app


def _png_bytes():
    ok, buf = cv2.imencode(".png", np.full((240, 320, 3), 120, np.uint8))
    assert ok
    return io.BytesIO(buf.tobytes())


def test_health_and_detect_on_blank_image():
    with TestClient(app) as client:  # runs lifespan -> loads best.pt
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["classes"] == ["Helicoverpa-Armigera"]

        r = client.post("/detect", files={"image": ("f.png", _png_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["image"] == {"width": 320, "height": 240}
        assert body["pest_detected"] is False and body["detections"] == []


def test_detect_rejects_non_image():
    with TestClient(app) as client:
        r = client.post("/detect", files={"image": ("x.txt", io.BytesIO(b"not an image"), "text/plain")})
        assert r.status_code == 400


def test_detect_rejects_bad_confidence():
    with TestClient(app) as client:
        r = client.post("/detect?confidence=2", files={"image": ("f.png", _png_bytes(), "image/png")})
        assert r.status_code == 422


# ---- spray decision per browser session (uses a fake model: we have no pest photo) ----
from types import SimpleNamespace  # noqa: E402


class _Box:
    def __init__(self, conf):
        self.conf, self.cls, self.xyxy = [conf], [0], [[10, 10, 60, 60]]


class _SwitchModel:
    """Sees a pest only while .pest is True."""

    names = {0: "Helicoverpa-Armigera"}
    task = "segment"

    def __init__(self):
        self.pest = True

    def __call__(self, frame, verbose=False):
        return [SimpleNamespace(boxes=[_Box(0.9)] if self.pest else [])]


def _post(client, session):
    return client.post(
        f"/detect?session={session}",
        files={"image": ("f.png", _png_bytes(), "image/png")},
    ).json()


def test_spray_after_three_pest_frames_then_reset_and_sessions_are_separate():
    with TestClient(app) as client:
        fake = _SwitchModel()
        app.state.model = fake  # replace the real model after startup

        a = [_post(client, "tab-a") for _ in range(3)]
        assert [r["decision"] for r in a] == ["DON'T SPRAY", "DON'T SPRAY", "SPRAY"]
        assert a[2]["confirmed_frames"] == 3 and a[2]["required_frames"] == 3
        assert a[2]["detections"][0]["class_name"] == "Helicoverpa-Armigera"

        # another browser tab starts from zero
        assert _post(client, "tab-b")["decision"] == "DON'T SPRAY"

        # pest disappears -> counter resets
        fake.pest = False
        r = _post(client, "tab-a")
        assert r["decision"] == "DON'T SPRAY" and r["confirmed_frames"] == 0


def test_bad_session_id_rejected():
    with TestClient(app) as client:
        r = client.post("/detect?session=../x", files={"image": ("f.png", _png_bytes(), "image/png")})
        assert r.status_code == 422


def test_home_page_is_served():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200 and "CropGuard" in r.text
