from types import SimpleNamespace

import numpy as np

from src.tools.diagnose import best_confidence, find_boxes, summarize, web_like


class _Box:
    def __init__(self, conf, xyxy):
        self.conf, self.xyxy = [conf], [xyxy]


class _Model:
    def __call__(self, image, conf=0.25, verbose=False):
        return [SimpleNamespace(boxes=[_Box(0.3, [1, 2, 3, 4]), _Box(0.8, [5, 6, 7, 8])])]


def test_find_boxes_sorted_highest_first():
    boxes = find_boxes(_Model(), np.zeros((10, 10, 3), np.uint8))
    assert [round(c, 1) for c, _ in boxes] == [0.8, 0.3]
    assert best_confidence(boxes) == 0.8 and best_confidence([]) == 0.0


def test_summarize_counts():
    assert summarize([0.0, 0.3, 0.7, 0.9]) == (3, 2, 0.475)


def test_web_like_shrinks_wide_images_only():
    big = np.zeros((900, 1600, 3), np.uint8)
    assert web_like(big, 640).shape[:2] == (360, 640)
    small = np.zeros((100, 200, 3), np.uint8)
    assert web_like(small, 640).shape[:2] == (100, 200)
