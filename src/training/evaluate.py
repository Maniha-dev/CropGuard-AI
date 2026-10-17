from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "best.pt"
DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
    f"Trained model not found: {MODEL_PATH}. "
    "Make sure best.pt is present in models/trained/."
)

    model = YOLO(MODEL_PATH)
    metrics = model.val(
        data=DATA_PATH,
        split="test",
        imgsz=640,
        project=PROJECT_ROOT / "results" / "evaluation",
        name="cropguard",
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print("\n================================")
    print("       CropGuard Evaluation")
    print("================================")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")

    print("================================")


if __name__ == "__main__":
    main()