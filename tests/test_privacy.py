from pathlib import Path

import cv2
import numpy as np
import pytest

from marg.vision.config import VisionConfig
from marg.vision.pipeline import run
from marg.vision.privacy import detect_sensitive_regions, redact


def test_redact_pixelates_box_and_preserves_outside() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(160, 220, 3), dtype=np.uint8)
    original = image.copy()
    output, count = redact(image, [(50, 40, 80, 60)])
    assert count == 1
    assert np.var(output[40:100, 50:130].astype(np.float32)) < np.var(
        original[40:100, 50:130].astype(np.float32)
    ) * 0.5
    mask = np.ones(image.shape[:2], dtype=bool)
    mask[40:100, 50:130] = False
    assert np.array_equal(output[mask], original[mask])


def test_redact_protects_damage_region() -> None:
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, size=(160, 220, 3), dtype=np.uint8)
    original = image.copy()
    output, count = redact(
        image,
        [(50, 40, 80, 60)],
        protect=[(80, 55, 35, 30)],
    )
    assert count == 1
    assert np.array_equal(output[55:85, 80:115], original[55:85, 80:115])


def test_sensitive_region_detection_on_bengaluru_fixture() -> None:
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "pothole_bengaluru.jpg"))
    regions = detect_sensitive_regions(image)
    assert isinstance(regions, list)


@pytest.mark.parametrize(
    ("filename", "minimum"),
    [("privacy_face_cc0.jpg", 1), ("privacy_plate_cc0.png", 1)],
)
def test_sensitive_fixture_detection_and_redaction(filename: str, minimum: int) -> None:
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / filename))
    regions = detect_sensitive_regions(image)
    assert len(regions) >= minimum
    for x, y, width, height in regions:
        output, count = redact(image, [(
            x,
            y,
            width,
            height,
        )])
        original_region = image[y : y + height, x : x + width]
        redacted_region = output[y : y + height, x : x + width]
        assert count == 1
        assert np.var(redacted_region.astype(np.float32)) < np.var(
            original_region.astype(np.float32)
        ) * 0.5


def test_sensitive_dashcam_fixture_detects_red_van_plate() -> None:
    image = cv2.imread(
        str(Path(__file__).parent / "fixtures" / "dashcam_plates_CC_BY_SA_4.0.jpg")
    )
    regions = detect_sensitive_regions(image)
    expected = (400, 400, 140, 90)
    ex, ey, ew, eh = expected
    assert any(
        ex <= x + width / 2 <= ex + ew and ey <= y + height / 2 <= ey + eh
        for x, y, width, height in regions
    )


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx").exists(),
    reason="ONNX detector model is not present in this checkout",
)
def test_pipeline_records_redactions(tmp_path: Path) -> None:
    video = Path(__file__).parent / "fixtures" / "clip3s.mp4"
    config = VisionConfig(
        target_width=320,
        model_path=Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx",
        redact=True,
    )
    run(video, out_dir=tmp_path / "out", config=config)
    result = (tmp_path / "out" / "result.json").read_text(encoding="utf-8")
    import json

    payload = json.loads(result)
    assert isinstance(payload["redactions"], int)
    assert payload["redactions"] >= 0
