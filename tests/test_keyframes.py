import cv2
import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.keyframes import KeyframeSelector


def _frame(shift: int = 0) -> np.ndarray:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in range(20, 300, 30):
        cv2.circle(image, (x + shift, 100 + (x % 50)), 8, (255, 255, 255), -1)
    cv2.rectangle(image, (50 + shift, 150), (250 + shift, 210), (180, 180, 180), 2)
    return image


def test_identical_frames_one_keyframe() -> None:
    selector = KeyframeSelector(VisionConfig())
    assert selector.consider(_frame(), 0, 0.0) is not None
    assert selector.consider(_frame(), 1, 0.1) is None


def test_shifted_frame_new_keyframe() -> None:
    selector = KeyframeSelector(VisionConfig())
    assert selector.consider(_frame(), 0, 0.0) is not None
    assert selector.consider(_frame(80), 1, 0.1) is not None
