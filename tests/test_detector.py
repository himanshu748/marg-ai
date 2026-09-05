from pathlib import Path

import cv2
import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector


def test_detector_loads_and_runs() -> None:
    model = Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx"
    detector = DNNDetector(model, VisionConfig())
    detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
    assert isinstance(detections, list)


def test_bengaluru_pothole_detects_d40() -> None:
    model = Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx"
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "pothole_bengaluru.jpg"))
    assert image is not None
    detector = DNNDetector(model, VisionConfig())
    detections = detector.detect(image)
    assert any(
        detection.class_name == "D40" and detection.confidence > 0.5
        for detection in detections
    )
