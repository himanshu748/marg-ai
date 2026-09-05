from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import VisionConfig
from .models import Detection, Observation, TrackUpdate


@dataclass(slots=True)
class _Track:
    instance_id: int
    bbox: tuple[float, float, float, float]
    classes: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    fused_conf: float = 0.0
    keyframe_ids: list[int] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    best_observation: Observation | None = None
    unseen_keyframes: int = 0
    lat: float = 0.0
    lon: float = 0.0


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class InstanceTracker:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._tracks: dict[int, _Track] = {}
        self._closed: dict[int, _Track] = {}
        self._next_id = 0

    def update(
        self,
        detections: list[Detection],
        keyframe_id: int | None,
        flow: np.ndarray | None = None,
        lat: float = 0.0,
        lon: float = 0.0,
        frame_idx: int = 0,
        t_s: float = 0.0,
    ) -> list[TrackUpdate]:
        predicted = {track_id: self._warp(track.bbox, flow) for track_id, track in self._tracks.items()}
        available = set(predicted)
        updates: list[TrackUpdate] = []
        assigned: set[int] = set()
        for detection in detections:
            best_id = -1
            best_iou = self.config.tracker_iou
            for track_id in available - assigned:
                overlap = iou(predicted[track_id], detection.bbox)
                if overlap > best_iou:
                    best_iou, best_id = overlap, track_id
            if best_id < 0:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = _Track(instance_id=best_id, bbox=detection.bbox)
            track = self._tracks[best_id]
            track.bbox = detection.bbox
            observation = Observation(
                frame_idx=frame_idx,
                t_s=t_s,
                bbox=[float(value) for value in detection.bbox],
                class_name=detection.class_name,
                conf=detection.confidence,
                fused_conf=detection.fused_conf,
            )
            best_observation = None
            if track.best_observation is None or observation.fused_conf > track.best_observation.fused_conf:
                track.best_observation = observation
                best_observation = observation
            track.observations.append(observation)
            track.classes.append(detection.class_name)
            track.confidences.append(detection.confidence)
            track.fused_conf = max(track.fused_conf, detection.fused_conf)
            track.lat, track.lon = lat, lon
            track.unseen_keyframes = 0
            if keyframe_id is not None and (not track.keyframe_ids or track.keyframe_ids[-1] != keyframe_id):
                track.keyframe_ids.append(keyframe_id)
            assigned.add(best_id)
            updates.append(
                TrackUpdate(
                    instance_id=best_id,
                    detection=detection,
                    observation=observation,
                    best_observation=best_observation,
                )
            )
        if keyframe_id is not None:
            for track_id, track in list(self._tracks.items()):
                if track_id not in assigned:
                    track.unseen_keyframes += 1
                    if track.unseen_keyframes >= self.config.tracker_max_unseen_keyframes:
                        self._closed[track_id] = self._tracks.pop(track_id)
        return updates

    def finalize(self) -> list[_Track]:
        return list(self._closed.values()) + list(self._tracks.values())

    @staticmethod
    def _warp(bbox: tuple[float, float, float, float], flow: np.ndarray | None) -> tuple[float, float, float, float]:
        if flow is None or flow.size == 0:
            return bbox
        x, y, width, height = bbox
        h, w = flow.shape[:2]
        points = np.float32(
            [
                [x, y],
                [x + width, y],
                [x, y + height],
                [x + width, y + height],
            ]
        )
        points_small = points
        offsets = np.float32(
            [flow[int(np.clip(point[1], 0, h - 1)), int(np.clip(point[0], 0, w - 1))] for point in points_small]
        )
        warped = points_small + offsets
        x0, y0 = np.min(warped, axis=0)
        x1, y1 = np.max(warped, axis=0)
        return (float(x0), float(y0), float(max(1.0, x1 - x0)), float(max(1.0, y1 - y0)))

    @staticmethod
    def summarize(track: _Track) -> tuple[str, float, list[float]]:
        class_name = Counter(track.classes).most_common(1)[0][0]
        return class_name, max(track.confidences, default=0.0), [
            observation.conf for observation in track.observations
        ]

    @staticmethod
    def classes(track: _Track) -> list[str]:
        return [observation.class_name for observation in track.observations]
