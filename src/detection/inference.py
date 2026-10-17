from dataclasses import dataclass


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


def detect_frame(model, frame, confidence_threshold=0.60):
    detections = []
    highest_confidence = 0.0

    for result in model(frame, verbose=False):
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            highest_confidence = max(highest_confidence, confidence)

            if confidence >= confidence_threshold:
                coordinates = tuple(map(int, box.xyxy[0]))

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        box=coordinates,
                    )
                )

    return detections, highest_confidence