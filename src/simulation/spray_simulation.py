import cv2

from src.decision.spray_decision import SprayDecision, REQUIRED_FRAMES
from src.detection.detector import load_model
from src.detection.inference import detect_frame


def main():
    model = load_model()
    decision_engine = SprayDecision()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    print("================================")
    print(" CropGuard Spray Simulation")
    print(" Press Q to quit")
    print("================================")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read camera frame.")
                break

            detections, highest_confidence = detect_frame(model, frame)
            decision = decision_engine.update(bool(detections), highest_confidence)

            for detection in detections:
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Helicoverpa {detection.confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Decision: {decision}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if decision == "SPRAY" else (0, 0, 255),
                3,
            )
            cv2.putText(frame, f"Confidence: {highest_confidence:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Confirmed frames: {decision_engine.consecutive_detections}/{REQUIRED_FRAMES}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("CropGuard - Spray Simulation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()