CONFIDENCE_THRESHOLD = 0.60
REQUIRED_FRAMES = 3


class SprayDecision:

    def __init__(self):
        self.consecutive_detections = 0

    def update(self, pest_detected, confidence):

        if pest_detected and confidence >= CONFIDENCE_THRESHOLD:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        if self.consecutive_detections >= REQUIRED_FRAMES:
            return "SPRAY"

        return "DON'T SPRAY"