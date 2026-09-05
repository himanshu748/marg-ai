from pathlib import Path

import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector


def test_detector_loads_and_runs() -> None:
    model = Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx"
    detector = DNNDetector(model, VisionConfig())
    detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
    assert isinstance(detections, list)
