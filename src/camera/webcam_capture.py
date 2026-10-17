import cv2

from src.detection.detector import load_model
from src.detection.inference import detect_frame
from src.decision.spray_decision import SprayDecision


def main():
    # Load trained model
    model = load_model(require_trained=True)

    # Create spray decision system
    decision_system = SprayDecision()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print("CropGuard camera started")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Detect pest
        detections, highest_confidence = detect_frame(
            model,
            frame,
            confidence_threshold=0.60
        )

        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Pest {detection.confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Make spray decision
        pest_detected = len(detections) > 0

        decision = decision_system.update(
            pest_detected,
            highest_confidence
        )

        # Display detection status
        if pest_detected:
            status = f"PEST DETECTED | Confidence: {highest_confidence:.2f}"
        else:
            status = "NO PEST DETECTED"

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Display spray decision
        cv2.putText(
            frame,
            f"Decision: {decision}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255) if decision == "SPRAY" else (255, 255, 255),
            2
        )

        cv2.imshow("CropGuard Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite(
                "results/camera_test/test_frame.jpg",
                frame
            )
            print("Frame saved")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()