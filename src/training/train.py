from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    model = YOLO(PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt")

    model.train(
    data=PROJECT_ROOT / "dataset" / "data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    patience=5,
    workers=4,
    project=PROJECT_ROOT / "results" / "training",
    name="yolov8n_baseline",
    )

    print("Training completed.")


if __name__ == "__main__":
    main()