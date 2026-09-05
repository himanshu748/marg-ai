import cv2
import numpy as np

from .config import VisionConfig
from .models import Keyframe


class KeyframeSelector:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._orb = cv2.ORB_create(nfeatures=config.orb_nfeatures)
        self._last_gray: np.ndarray | None = None
        self._last_timestamp = -1.0
        self._next_id = 0

    def consider(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> Keyframe | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_gray is None:
            return self._accept(gray, frame, frame_index, timestamp_s)
        elapsed = timestamp_s - self._last_timestamp
        new_ratio = self._new_ratio(self._last_gray, gray)
        if elapsed > self.config.keyframe_max_elapsed_s or new_ratio >= self.config.keyframe_min_new_ratio:
            return self._accept(gray, frame, frame_index, timestamp_s)
        return None

    def _accept(
        self,
        gray: np.ndarray,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> Keyframe:
        keyframe = Keyframe(
            keyframe_id=self._next_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            frame=frame.copy(),
        )
        self._next_id += 1
        self._last_gray = gray
        self._last_timestamp = timestamp_s
        return keyframe

    def _new_ratio(self, previous: np.ndarray, current: np.ndarray) -> float:
        keypoints_a, descriptors_a = self._orb.detectAndCompute(previous, None)
        keypoints_b, descriptors_b = self._orb.detectAndCompute(current, None)
        if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 4 or len(keypoints_b) < 4:
            return 1.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return 1.0
        points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good])
        points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 5.0)
        if mask is None:
            return 1.0
        inlier_ratio = float(np.mean(mask.ravel().astype(np.float32)))
        return 1.0 - inlier_ratio
