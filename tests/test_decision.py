from src.decision.spray_decision import SprayDecision


def test_low_confidence_does_not_spray():
    decision = SprayDecision()
    assert decision.update(True, 0.59) == "DON'T SPRAY"


def test_one_or_two_detections_do_not_spray():
    decision = SprayDecision()
    assert decision.update(True, 0.60) == "DON'T SPRAY"
    assert decision.update(True, 0.90) == "DON'T SPRAY"


def test_three_consecutive_detections_spray():
    decision = SprayDecision()
    decision.update(True, 0.60)
    decision.update(True, 0.75)
    assert decision.update(True, 0.99) == "SPRAY"


def test_missing_detection_resets_counter():
    decision = SprayDecision()
    decision.update(True, 0.90)
    decision.update(True, 0.90)
    assert decision.update(False, 0.0) == "DON'T SPRAY"
    assert decision.update(True, 0.90) == "DON'T SPRAY"