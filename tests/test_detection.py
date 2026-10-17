from pathlib import Path

from src.detection.detector import PRETRAINED_MODEL, PROJECT_ROOT, TRAINED_MODEL, get_model_path


def test_project_paths_are_relative_to_repository():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert PRETRAINED_MODEL == PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt"
    assert TRAINED_MODEL == PROJECT_ROOT / "models" / "trained" / "best.pt"


def test_model_selection_finds_pretrained_model():
    assert get_model_path().exists()