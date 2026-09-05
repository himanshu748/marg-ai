import cv2
import numpy as np

from .config import VisionConfig
from .models import QualityResult


def assess(frame: np.ndarray, config: VisionConfig) -> QualityResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = np.logical_or(gray < 8, gray > 247)
    clipped_fraction = float(np.mean(clipped))
    accepted = (
        blur_variance >= config.blur_min_laplacian_var
        and clipped_fraction <= config.exposure_clip_pct_max
    )
    return QualityResult(
        accepted=accepted,
        blur_variance=blur_variance,
        clipped_fraction=clipped_fraction,
    )
