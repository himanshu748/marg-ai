from __future__ import annotations

import logging
import threading
from itertools import product
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_FACE_MODEL = _ROOT / "models" / "face_detection_yunet_2023mar.onnx"
_PLATE_MODEL = _ROOT / "models" / "license_plate_detection_lpd_yunet_2023mar.onnx"


class PrivacyDetectionError(RuntimeError):
    """Privacy checks could not complete; callers must not publish the image."""


class SensitiveRegionDetector:
    def __init__(self) -> None:
        self._face = None
        self._plate = None
        self._priors = self._make_priors()
        self._lock = threading.RLock()

    def ensure_ready(self) -> None:
        """Require both privacy models before producing any shareable artifacts."""
        with self._lock:
            for kind, path in (("face", _FACE_MODEL), ("license-plate", _PLATE_MODEL)):
                if not path.is_file():
                    raise PrivacyDetectionError(f"Privacy {kind} model is missing; redacted export is unavailable")
            try:
                if self._face is None:
                    self._face = cv2.FaceDetectorYN.create(str(_FACE_MODEL), "", (320, 320), 0.6, 0.3, 5000)
                if self._plate is None:
                    self._plate = cv2.dnn.readNet(str(_PLATE_MODEL))
            except cv2.error as error:
                raise PrivacyDetectionError("Privacy models could not load; redacted export is unavailable") from error

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if image is None or image.size == 0:
            raise PrivacyDetectionError("Cannot check privacy on an empty image")
        # OpenCV DNN networks have mutable input state and are shared by API jobs.
        with self._lock:
            self.ensure_ready()
            try:
                return _merge_overlapping(self._detect_faces(image) + self._detect_plates(image))
            except (cv2.error, ValueError, TypeError, IndexError) as error:
                LOGGER.warning("Privacy inference failed; refusing unverified export", exc_info=True)
                raise PrivacyDetectionError("Privacy detection failed; redacted export is unavailable") from error

    def _detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        self._face.setInputSize((width, height))
        _, detections = self._face.detect(image)
        if detections is None:
            return []
        return [_clip_box(row[:4], width, height) for row in detections]

    def _detect_plates(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        windows = [(0, 0, width, height)]
        lower_y = round(height * 0.4)
        upper_y = round(height * 0.6)
        for y0, y1 in ((lower_y, height), (0, upper_y)):
            for x0 in (0, width // 4, width // 2):
                windows.append((x0, y0, min(width, x0 + width // 2), y1))
        candidates: list[tuple[list[float], float]] = []
        for x0, y0, x1, y1 in windows:
            if x1 <= x0 or y1 <= y0:
                continue
            native = cv2.resize(
                image[y0:y1, x0:x1],
                (320, 240),
                interpolation=cv2.INTER_AREA,
            )
            self._plate.setInput(cv2.dnn.blobFromImage(native))
            loc, conf, iou = self._plate.forward(["loc", "conf", "iou"])
            if not all(np.isfinite(value).all() for value in (loc, conf, iou)):
                raise ValueError("Non-finite privacy detector output")
            scores = np.sqrt(
                np.clip(conf[:, 1], 0.0, 1.0) * np.clip(iou[:, 0], 0.0, 1.0)
            )
            quads = np.hstack(
                (
                    (
                        self._priors[:, 0:2]
                        + loc[:, 4:6] * 0.1 * self._priors[:, 2:4]
                    )
                    * (320, 240),
                    (
                        self._priors[:, 0:2]
                        + loc[:, 6:8] * 0.1 * self._priors[:, 2:4]
                    )
                    * (320, 240),
                    (
                        self._priors[:, 0:2]
                        + loc[:, 10:12] * 0.1 * self._priors[:, 2:4]
                    )
                    * (320, 240),
                    (
                        self._priors[:, 0:2]
                        + loc[:, 12:14] * 0.1 * self._priors[:, 2:4]
                    )
                    * (320, 240),
                )
            )
            scale_x = (x1 - x0) / 320
            scale_y = (y1 - y0) / 240
            for quad, score in zip(quads, scores):
                points = quad.reshape(4, 2)
                points[:, 0] = points[:, 0] * scale_x + x0
                points[:, 1] = points[:, 1] * scale_y + y0
                left, top = points.min(axis=0)
                right, bottom = points.max(axis=0)
                candidates.append(
                    (
                        [float(left), float(top), float(right - left), float(bottom - top)],
                        float(score),
                    )
                )
        if not candidates:
            return []
        boxes = [box for box, _ in candidates]
        scores = [score for _, score in candidates]
        keep = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )
        return [
            _clip_box(boxes[int(index)], width, height)
            for index in np.asarray(keep).reshape(-1)
        ]

    @staticmethod
    def _make_priors() -> np.ndarray:
        min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        steps = [8, 16, 32, 64]
        feature_maps = [[30, 40], [15, 20], [7, 10], [3, 5]]
        priors = []
        for level, (height, width) in enumerate(feature_maps):
            for row, column in product(range(height), range(width)):
                for min_size in min_sizes[level]:
                    priors.append([(column + 0.5) * steps[level] / 320, (row + 0.5) * steps[level] / 240, min_size / 320, min_size / 240])
        return np.asarray(priors, dtype=np.float32)

_DETECTOR = SensitiveRegionDetector()


def ensure_privacy_ready() -> None:
    _DETECTOR.ensure_ready()


def privacy_status() -> dict[str, object]:
    """Model loading readiness, not a claim that every face or plate is detected."""
    try:
        ensure_privacy_ready()
    except PrivacyDetectionError as error:
        return {"available": False, "detail": str(error)}
    return {"available": True, "detail": "Face and plate models loaded; automated redaction requires review"}


def detect_sensitive_regions(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    return _DETECTOR.detect(img)


def redact(
    img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    protect: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, int]:
    """Pixelate sensitive boxes, including any overlap with damage detections.

    ``protect`` remains accepted for older callers, but cannot exempt sensitive
    pixels. Pixels outside the sensitive boxes are always preserved.
    """
    output = img.copy()
    redacted = 0
    for box in boxes:
        x0, y0, x1, y1 = _expanded_bounds(box, output.shape[1], output.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        region = output[y0:y1, x0:x1]
        small_width, small_height = max(1, region.shape[1] // 24), max(1, region.shape[0] // 24)
        pixelated = cv2.resize(
            cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA),
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        pixelated = cv2.blur(
            pixelated,
            (max(3, small_width * 2 + 1), max(3, small_height * 2 + 1)),
        )
        mean_color = np.full_like(pixelated, cv2.mean(region)[:3])
        pixelated = cv2.addWeighted(pixelated, 0.25, mean_color, 0.75, 0)
        mask = np.zeros(region.shape[:2], dtype=bool)
        bx, by, bw, bh = box
        bx0, by0, bx1, by1 = max(x0, round(bx)), max(y0, round(by)), min(x1, round(bx + bw)), min(y1, round(by + bh))
        if bx1 <= bx0 or by1 <= by0:
            continue
        mask[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = True
        if np.any(mask):
            region[mask] = pixelated[mask]
            redacted += 1
    return output, redacted


def _expanded_bounds(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = box
    pad_x, pad_y = round(box_width * 0.10), round(box_height * 0.10)
    return max(0, x - pad_x), max(0, y - pad_y), min(width, x + box_width + pad_x), min(height, y + box_height + pad_y)


def _clip_box(values: tuple[float, float, float, float] | np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = [float(value) for value in values]
    x0, y0 = max(0, round(x)), max(0, round(y))
    x1, y1 = min(width, round(x + box_width)), min(height, round(y + box_height))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _merge_overlapping(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for candidate in boxes:
        if candidate[2] <= 0 or candidate[3] <= 0:
            continue
        current = candidate
        changed = True
        while changed:
            changed = False
            for index, existing in enumerate(merged):
                if _overlaps(current, existing):
                    current = _union(current, existing)
                    merged.pop(index)
                    changed = True
                    break
        merged.append(current)
    return merged


def _overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + ah, by + bh)


def _union(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0
