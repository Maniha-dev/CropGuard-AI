import cv2

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
        cv2.imwrite(
            "results/camera_test/test_frame.jpg",
            frame
        )
        print("Frame saved")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()