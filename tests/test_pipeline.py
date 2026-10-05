from pathlib import Path

import cv2
import numpy as np
import pytest

from marg.vision import pipeline
from marg.vision.config import VisionConfig
from marg.vision.models import Detection, QualityResult
from marg.vision.pipeline import run
from marg.vision.severity import bbox_area_m2


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx").exists(),
    reason="ONNX detector model is not present in this checkout",
)
def test_pipeline_end_to_end(tmp_path: Path) -> None:
    video = Path(__file__).parent / "fixtures" / "clip3s.mp4"
    config = VisionConfig(target_width=320, model_path=Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx")
    result = run(video, out_dir=tmp_path / "out", config=config)
    assert result.frames > 0
    assert result.frame_width == 320
    assert result.frame_height is not None and result.frame_height > 0
    saved = type(result).model_validate_json((tmp_path / "out" / "result.json").read_text())
    assert (saved.frame_width, saved.frame_height) == (result.frame_width, result.frame_height)
    assert result.geo_source == "synthetic"
    assert (tmp_path / "out" / "result.json").exists()
    output_dir = tmp_path / "out"
    assert all(
        instance.evidence_path is not None
        and not Path(instance.evidence_path).is_absolute()
        and (output_dir / instance.evidence_path).exists()
        for instance in result.instances
    )
    assert all(not Path(path).is_absolute() for path in result.keyframe_paths)
    assert all(not Path(path).is_absolute() for path in result.crop_paths)


def patch_synthetic_video(monkeypatch, frames, detections=None):
    class Capture:
        def __init__(self, path):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 30 if prop == cv2.CAP_PROP_FPS else (self.index - 1) * 1000 / 30

        def read(self):
            if self.index == len(frames):
                return False, None
            frame = frames[self.index].copy()
            self.index += 1
            return True, frame

        def release(self):
            pass

    class Detector:
        def __init__(self, *args):
            self.index = 0

        def detect(self, frame):
            height, width = frame.shape[:2]
            result = (
                detections[self.index] if detections is not None else [
                    Detection((0.4 * width, 0.75 * height, 0.1 * width, 0.05 * height), 3, "D40", 0.9, 0.9)
                ]
            )
            self.index += 1
            return result

    class Selector:
        def __init__(self, config):
            pass

        def consider(self, *args):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(pipeline, "DNNDetector", Detector)
    monkeypatch.setattr(pipeline, "KeyframeSelector", Selector)
    monkeypatch.setattr(pipeline, "assess", lambda *args: QualityResult(True, 100, 0))
    monkeypatch.setattr(pipeline, "fuse_detection", lambda frame, detection: detection)
    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", lambda previous, current, *args: np.zeros((*current.shape, 2), np.float32))


@pytest.mark.parametrize(
    "width,height,target_width,expected_size",
    [
        (320, 180, 320, (320, 180)),
        (320, 240, 960, (320, 240)),  # Sources smaller than the target are not enlarged.
        (640, 480, 320, (320, 240)),
        (320, 320, 0, (320, 320)),  # Resizing disabled.
        (180, 320, -1, (180, 320)),
    ],
)
def test_normalized_damage_area_is_stable_across_frame_sizes(monkeypatch, tmp_path, width, height, target_width, expected_size):
    patch_synthetic_video(monkeypatch, [np.zeros((height, width, 3), np.uint8)])
    config = VisionConfig(target_width=target_width, frame_stride=1, redact=False)
    result = run("synthetic.mp4", out_dir=tmp_path, config=config)
    assert (result.frame_width, result.frame_height) == expected_size
    assert len(result.instances) == 1
    # The normalized box represents the same calibrated road footprint,
    # regardless of resolution, aspect ratio or whether resizing is enabled.
    reference = bbox_area_m2((128, 135, 32, 9), 320, 180, config)
    area = result.instances[0].area_m2
    assert np.isfinite(area) and area > 0
    assert area == pytest.approx(reference, rel=1e-5)


def test_each_instance_uses_its_best_frame_dimensions(monkeypatch, tmp_path):
    frames = [np.zeros((240, 320, 3), np.uint8), np.zeros((480, 640, 3), np.uint8)]
    first_box = (128.0, 180.0, 32.0, 12.0)
    second_box = (384.0, 360.0, 64.0, 24.0)
    patch_synthetic_video(monkeypatch, frames, [
        [Detection(first_box, 3, "D40", 0.9, 0.9)],
        [Detection(first_box, 3, "D40", 0.8, 0.8), Detection(second_box, 3, "D40", 0.9, 0.9)],
    ])
    config = VisionConfig(target_width=0, frame_stride=1, redact=False)
    result = run("synthetic.mp4", out_dir=tmp_path, config=config)
    first, second = result.instances
    assert (result.frame_width, result.frame_height) == (640, 480)
    assert (first.best_frame_idx, second.best_frame_idx) == (0, 1)
    assert first.area_m2 == pytest.approx(bbox_area_m2(first_box, 320, 240, config))
    assert second.area_m2 == pytest.approx(bbox_area_m2(second_box, 640, 480, config))
    assert first.area_m2 != pytest.approx(bbox_area_m2(first_box, 640, 480, config))
