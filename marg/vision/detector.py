from pathlib import Path

import cv2
import numpy as np

from .config import VisionConfig
from .models import Detection


class DNNDetector:
    def __init__(self, model_path: str | Path, config: VisionConfig) -> None:
        self.config = config
        self.model_path = str(model_path)
        self.net = cv2.dnn.readNetFromONNX(self.model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        input_image, scale, pad_x, pad_y = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            input_image,
            scalefactor=1.0 / 255.0,
            size=(self.config.dnn_input_size, self.config.dnn_input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        rows = self._rows(output)
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        raw: list[tuple[list[int], float, int]] = []
        for row in rows:
            if row.shape[0] < 5 + len(self.config.classes):
                continue
            class_scores = row[4 : 4 + len(self.config.classes)]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.config.conf_thresh:
                continue
            cx, cy, width, height = [float(value) for value in row[:4]]
            x = int((cx - width / 2.0 - pad_x) / scale)
            y = int((cy - height / 2.0 - pad_y) / scale)
            w = int(width / scale)
            h = int(height / scale)
            x = max(0, min(frame.shape[1] - 1, x))
            y = max(0, min(frame.shape[0] - 1, y))
            w = max(1, min(frame.shape[1] - x, w))
            h = max(1, min(frame.shape[0] - y, h))
            boxes.append([x, y, w, h])
            scores.append(confidence)
            class_ids.append(class_id)
            raw.append(([x, y, w, h], confidence, class_id))
        if not boxes:
            return []
        kept = cv2.dnn.NMSBoxes(boxes, scores, self.config.conf_thresh, self.config.nms_iou)
        indices = np.asarray(kept).reshape(-1).tolist() if len(kept) else []
        return [
            Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=class_id,
                class_name=self.config.classes[class_id],
                confidence=confidence,
                fused_conf=confidence,
            )
            for index in indices
            for box, confidence, class_id in [raw[index]]
        ]

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        size = self.config.dnn_input_size
        scale = min(size / frame.shape[1], size / frame.shape[0])
        width = max(1, int(round(frame.shape[1] * scale)))
        height = max(1, int(round(frame.shape[0] * scale)))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - width) / 2.0
        pad_y = (size - height) / 2.0
        canvas[int(pad_y) : int(pad_y) + height, int(pad_x) : int(pad_x) + width] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _rows(output: np.ndarray) -> np.ndarray:
        squeezed = np.squeeze(output)
        if squeezed.ndim != 2:
            raise ValueError(f"Unexpected DNN output shape: {output.shape}")
        return squeezed.T if squeezed.shape[0] < squeezed.shape[1] else squeezed
