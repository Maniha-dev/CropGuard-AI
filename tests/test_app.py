from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("app_name", ["app.py", "app_live.py"])
def test_app_starts_without_exceptions(app_name):
    app_path = Path(__file__).resolve().parents[1] / app_name
    with patch("src.detection.detector.load_model", return_value=object()), patch(
        "streamlit_webrtc.webrtc_streamer", return_value=None
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception