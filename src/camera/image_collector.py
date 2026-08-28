import cv2
import os

SAVE_DIR = "dataset/collected"

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

image_count = 0

print("Camera started")
print("Press S to save an image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow("CropGuard Image Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        image_count += 1

        filename = os.path.join(
            SAVE_DIR,
            f"image_{image_count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total images captured: {image_count}")