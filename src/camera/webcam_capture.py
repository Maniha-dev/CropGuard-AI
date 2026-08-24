import cv2
from pathlib import Path

# Get CropGuard project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Create the results folder
SAVE_DIR = PROJECT_ROOT / "results" / "camera_test"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PATH = SAVE_DIR / "test_frame.jpg"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow("CropGuard Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        success = cv2.imwrite(str(SAVE_PATH), frame)

        if success:
            print("Frame saved successfully:")
            print(SAVE_PATH)
        else:
            print("ERROR: Could not save frame")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()