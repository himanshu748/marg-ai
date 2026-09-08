from pathlib import Path

import cv2
import numpy as np
import pytest

from marg.vision import privacy
from marg.vision.config import VisionConfig
from marg.vision.pipeline import _save_crop, run
from marg.vision.privacy import (
    PrivacyDetectionError,
    SensitiveRegionDetector,
    detect_sensitive_regions,
    redact,
)


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


@pytest.mark.parametrize("protected", [[(80, 55, 35, 30)], [(0, 0, 220, 160)]])
def test_sensitive_pixels_win_over_damage_protection(protected) -> None:
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, size=(160, 220, 3), dtype=np.uint8)
    original = image.copy()
    output, count = redact(
        image,
        [(50, 40, 80, 60)],
        protect=protected,
    )
    assert count == 1
    assert not np.array_equal(output[55:85, 80:115], original[55:85, 80:115])
    expected, _ = redact(image, [(50, 40, 80, 60)])
    assert np.array_equal(output, expected)
    assert np.array_equal(image, original)


@pytest.mark.parametrize("model_name", ["_FACE_MODEL", "_PLATE_MODEL"])
def test_missing_model_blocks_export(monkeypatch, tmp_path: Path, model_name: str) -> None:
    monkeypatch.setattr(privacy, model_name, tmp_path / "missing.onnx")
    with pytest.raises(PrivacyDetectionError, match="model is missing"):
        SensitiveRegionDetector().detect(np.zeros((160, 220, 3), dtype=np.uint8))
    assert privacy.privacy_status()["available"] is False


def test_corrupt_model_blocks_export(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "corrupt.onnx"
    model.write_bytes(b"not an ONNX model")
    monkeypatch.setattr(privacy, "_FACE_MODEL", model)
    with pytest.raises(PrivacyDetectionError, match="could not load"):
        SensitiveRegionDetector().ensure_ready()


@pytest.mark.parametrize("detector_method", ["_detect_faces", "_detect_plates"])
def test_inference_failure_cannot_be_treated_as_no_sensitive_regions(monkeypatch, detector_method: str) -> None:
    detector = SensitiveRegionDetector()
    monkeypatch.setattr(detector, "ensure_ready", lambda: None)
    monkeypatch.setattr(detector, "_detect_faces", lambda image: [])
    monkeypatch.setattr(detector, "_detect_plates", lambda image: [])

    def fail(image):
        raise cv2.error("inference unavailable")

    monkeypatch.setattr(detector, detector_method, fail)
    with pytest.raises(PrivacyDetectionError, match="Privacy detection failed"):
        detector.detect(np.zeros((160, 220, 3), dtype=np.uint8))


def test_damage_crop_masks_sensitive_overlap_using_full_frame_coordinates(tmp_path: Path) -> None:
    image = np.random.default_rng(9).integers(0, 256, size=(160, 220, 3), dtype=np.uint8)
    target = tmp_path / "crop.jpg"
    count = _save_crop(image, (80, 55, 35, 30), target, redact_output=True, sensitive_regions=[(50, 40, 80, 60)])
    output = cv2.imread(str(target))
    assert count == 1
    assert output.shape == (30, 35, 3)
    assert np.var(output.astype(np.float32)) < np.var(image[55:85, 80:115].astype(np.float32)) * 0.5


def test_redacted_crop_not_written_when_privacy_detector_fails(monkeypatch, tmp_path: Path) -> None:
    def fail(image):
        raise PrivacyDetectionError("Privacy detection failed")

    monkeypatch.setattr("marg.vision.pipeline.detect_sensitive_regions", fail)
    target = tmp_path / "crop.jpg"
    with pytest.raises(PrivacyDetectionError):
        _save_crop(np.zeros((160, 220, 3), dtype=np.uint8), (10, 10, 30, 30), target, redact_output=True)
    assert not target.exists()


def test_pipeline_checks_privacy_before_replacing_existing_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(privacy, "_FACE_MODEL", tmp_path / "missing.onnx")
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "result.json"
    existing.write_text('{"existing": true}')
    with pytest.raises(PrivacyDetectionError, match="model is missing"):
        run("unused-video.mp4", out_dir=output)
    assert existing.read_text() == '{"existing": true}'


def test_failed_restart_does_not_expose_previous_completion(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("marg.vision.pipeline.ensure_privacy_ready", lambda: None)

    def fail(*args):
        raise RuntimeError("damage model unavailable")

    monkeypatch.setattr("marg.vision.pipeline.DNNDetector", fail)
    output = tmp_path / "out"
    output.mkdir()
    (output / "result.json").write_text('{"existing": true}')
    with pytest.raises(RuntimeError, match="damage model unavailable"):
        run("unused-video.mp4", out_dir=output)
    assert not (output / "result.json").exists()


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
