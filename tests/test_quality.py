import cv2
import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.quality import assess


def test_blurred_vs_sharp_synthetic() -> None:
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (280, 200), (190, 190, 190), 4)
    sharp = assess(frame, VisionConfig())
    blurred = assess(cv2.GaussianBlur(frame, (31, 31), 0), VisionConfig())
    assert sharp.blur_variance > blurred.blur_variance
    assert sharp.accepted
    assert not blurred.accepted
