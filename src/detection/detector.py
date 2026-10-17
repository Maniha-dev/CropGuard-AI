from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINED_MODEL = PROJECT_ROOT / "models" / "trained" / "best.pt"
PRETRAINED_MODEL = PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt"


def get_model_path(require_trained=False):
    if TRAINED_MODEL.exists():
        return TRAINED_MODEL
    if require_trained:
        raise FileNotFoundError(f"Trained model not found: {TRAINED_MODEL}")
    if PRETRAINED_MODEL.exists():
        return PRETRAINED_MODEL
    raise FileNotFoundError(f"YOLOv8 model not found: {PRETRAINED_MODEL}")


def load_model(require_trained=False):
    return YOLO(get_model_path(require_trained=require_trained))