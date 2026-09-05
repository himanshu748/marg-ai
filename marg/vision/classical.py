import cv2
import numpy as np

from .models import Detection


def pothole_support(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    x, y, width, height = [int(round(value)) for value in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame.shape[1], max(x0 + 1, x + width))
    y1 = min(frame.shape[0], max(y0 + 1, y + height))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    road_start = min(frame.shape[0] - 1, max(y1, int(frame.shape[0] * 0.62)))
    road_band = l_channel[road_start:, :]
    road_median = float(np.median(road_band)) if road_band.size else float(np.median(l_channel))
    dark_ratio = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0] < road_median - 25))
    variance = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    band_variance = float(cv2.Laplacian(road_band, cv2.CV_64F).var()) if road_band.size else variance
    texture_ratio = min(1.0, variance / max(1.0, band_variance))
    return float(np.clip(0.5 * dark_ratio + 0.5 * texture_ratio, 0.0, 1.0))


def fuse_detection(frame: np.ndarray, detection: Detection) -> Detection:
    support = pothole_support(frame, detection.bbox) if detection.class_name == "D40" else 0.0
    detection.fused_conf = (
        0.7 * detection.confidence + 0.3 * support
        if detection.class_name == "D40"
        else detection.confidence
    )
    return detection
