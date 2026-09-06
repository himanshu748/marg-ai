from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CASCADE_NAMES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_profileface.xml",
    "haarcascade_russian_plate_number.xml",
)


def detect_sensitive_regions(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces and license plates using the cascades shipped with OpenCV."""
    if img is None or img.size == 0:
        return []
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes: list[tuple[int, int, int, int]] = []
    cascade_factory = getattr(cv2, "CascadeClassifier", None)
    if cascade_factory is None:
        return boxes
    for name in _CASCADE_NAMES:
        cascade = cascade_factory(str(Path(cv2.data.haarcascades) / name))
        if cascade.empty():
            continue
        found = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        boxes.extend(tuple(int(value) for value in box) for box in found)
    return _merge_overlapping(boxes)


def redact(
    img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    protect: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, int]:
    """Pixelate sensitive boxes without modifying protected damage boxes."""
    output = img.copy()
    protected = protect or []
    redacted = 0
    for box in boxes:
        x0, y0, x1, y1 = _expanded_bounds(box, output.shape[1], output.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        region = output[y0:y1, x0:x1]
        if region.size == 0:
            continue
        small_width = max(1, region.shape[1] // 12)
        small_height = max(1, region.shape[0] // 12)
        pixelated = cv2.resize(
            cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA),
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        bx, by, bw, bh = box
        bx0 = max(x0, round(bx))
        by0 = max(y0, round(by))
        bx1 = min(x1, round(bx + bw))
        by1 = min(y1, round(by + bh))
        if bx1 <= bx0 or by1 <= by0:
            continue
        mask[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = 1
        for protected_box in protected:
            px, py, pw, ph = protected_box
            px0 = max(x0, round(px))
            py0 = max(y0, round(py))
            px1 = min(x1, round(px + pw))
            py1 = min(y1, round(py + ph))
            if px1 > px0 and py1 > py0:
                mask[py0 - y0 : py1 - y0, px0 - x0 : px1 - x0] = 0
        if np.any(mask):
            region[mask.astype(bool)] = pixelated[mask.astype(bool)]
            redacted += 1
    return output, redacted


def _expanded_bounds(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = box
    pad_x = round(box_width * 0.10)
    pad_y = round(box_height * 0.10)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(width, x + box_width + pad_x),
        min(height, y + box_height + pad_y),
    )


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for candidate in boxes:
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


def _overlaps(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return (
        max(ax, bx) < min(ax + aw, bx + bw)
        and max(ay, by) < min(ay + ah, by + bh)
    )


def _union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0
