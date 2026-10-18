"""Find out WHY the model does or does not see pests.

Usage (from the project root):
    python -m src.tools.diagnose dataset/test/images
    python -m src.tools.diagnose results/camera_test          # frames saved from the camera
    python -m src.tools.diagnose my_photo.jpg --web-width 480

For every image it prints the highest pest confidence found (at a very low threshold),
once on the ORIGINAL image and once after the same shrink + JPEG compression that the
live web page applies. Annotated copies go to results/diagnose/.

How to read the result:
  * original is high, web-like is much lower -> the live page's compression hurts (raise --web-width)
  * both are low on real photos            -> the model does not know this kind of picture
                                              (domain gap): collect + label more real images
  * both are high on dataset images but low on camera frames -> same: domain gap
"""
import argparse
from pathlib import Path

import cv2

from src.detection.detector import load_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOW_CONFIDENCE = 0.05   # look at even weak detections
SEEN = 0.25             # ultralytics default: "the model noticed something"
SPRAY_CAPABLE = 0.60    # CropGuard's SPRAY rule needs at least this
WEB_WIDTH = 640         # must match MAX_WIDTH in web/index.html
WEB_QUALITY = 85        # must match the JPEG quality in web/index.html


def web_like(image, width=WEB_WIDTH, quality=WEB_QUALITY):
    """Reproduce what the live page uploads: shrink to `width`, then JPEG-compress."""
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else image


def find_boxes(model, image):
    """Return [(confidence, (x1, y1, x2, y2)), ...] at a very low threshold."""
    found = []
    for result in model(image, conf=LOW_CONFIDENCE, verbose=False):
        if result.boxes is None:
            continue
        for box in result.boxes:
            found.append((float(box.conf[0]), tuple(map(int, box.xyxy[0]))))
    return sorted(found, reverse=True)


def best_confidence(boxes):
    return boxes[0][0] if boxes else 0.0


def list_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def annotate(image, boxes):
    out = image.copy()
    for confidence, (x1, y1, x2, y2) in boxes:
        color = (0, 200, 0) if confidence >= SPRAY_CAPABLE else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{confidence:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def summarize(bests):
    n = len(bests)
    return (
        sum(b >= SEEN for b in bests),
        sum(b >= SPRAY_CAPABLE for b in bests),
        sum(bests) / n if n else 0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="CropGuard model diagnosis")
    parser.add_argument("source", help="an image or a folder of images")
    parser.add_argument("--web-width", type=int, default=WEB_WIDTH)
    parser.add_argument("--limit", type=int, default=200, help="max images to check")
    parser.add_argument("--out", default="results/diagnose")
    args = parser.parse_args()

    images = list_images(args.source)[: args.limit]
    if not images:
        print(f"No images found in {args.source!r}.")
        return 1

    model = load_model(require_trained=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_best, web_best, rows = [], [], []
    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip (cannot read): {path}")
            continue
        boxes_original = find_boxes(model, image)
        boxes_web = find_boxes(model, web_like(image, args.web_width))
        original_best.append(best_confidence(boxes_original))
        web_best.append(best_confidence(boxes_web))
        rows.append((path.name, original_best[-1], web_best[-1]))
        if index < 30:
            cv2.imwrite(str(out_dir / f"{path.stem}_boxes.jpg"), annotate(image, boxes_original))

    print(f"\n{'image':40} {'original':>9} {'web-like':>9}   (highest confidence, 0.00 = nothing found)")
    for name, original, web in rows[:25]:
        print(f"{name[:40]:40} {original:9.2f} {web:9.2f}")
    if len(rows) > 25:
        print(f"... and {len(rows) - 25} more")

    n = len(original_best)
    seen_o, spray_o, avg_o = summarize(original_best)
    seen_w, spray_w, avg_w = summarize(web_best)
    print(f"\nImages checked: {n}")
    print(f"{'':28}{'original':>10}{'web-like':>10}")
    print(f"{'model noticed (>=0.25)':28}{seen_o:>10}{seen_w:>10}")
    print(f"{'SPRAY-capable (>=0.60)':28}{spray_o:>10}{spray_w:>10}")
    print(f"{'average best confidence':28}{avg_o:>10.2f}{avg_w:>10.2f}")
    print(f"\nAnnotated images: {out_dir.resolve()}")
    print("Orange box = weak (below 0.60), green box = strong (0.60 or more).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
