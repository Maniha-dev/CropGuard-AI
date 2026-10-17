from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "best.pt"
TEST_IMAGES = PROJECT_ROOT / "dataset" / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "predictions"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    model.predict(
        source=TEST_IMAGES,
        conf=0.25,
        save=True,
        project=OUTPUT_DIR,
        name="test_predictions",
    )

    print("================================")
    print("CropGuard Test Completed")
    print("================================")
    print(f"Results saved to: {OUTPUT_DIR / 'test_predictions'}")


if __name__ == "__main__":
    main()