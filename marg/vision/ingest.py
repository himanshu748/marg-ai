import time
from collections.abc import Iterator
from pathlib import Path

import cv2

from .config import VisionConfig
from .models import FramePacket


class VideoSource:
    def __init__(self, video_path: str | Path, config: VisionConfig) -> None:
        self.video_path = str(video_path)
        self.config = config
        self.last_decode_s = 0.0

    def __iter__(self) -> Iterator[FramePacket]:
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        index = 0
        try:
            while True:
                started = time.perf_counter()
                ok, frame = capture.read()
                self.last_decode_s = time.perf_counter() - started
                if not ok:
                    break
                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 0 else index / max(fps, 30.0)
                if self.config.target_width > 0 and frame.shape[1] > self.config.target_width:
                    scale = self.config.target_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (self.config.target_width, max(1, int(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                yield FramePacket(index=index, timestamp_s=timestamp_s, frame=frame)
                index += 1
        finally:
            capture.release()
