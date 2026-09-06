from pathlib import Path

import pytest

from marg.vision.config import VisionConfig
from marg.vision.pipeline import run

pytestmark = pytest.mark.skipif(
    not (Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx").exists(),
    reason="ONNX detector model is not present in this checkout",
)


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    video = Path(__file__).parent / "fixtures" / "clip3s.mp4"
    config = VisionConfig(target_width=320, model_path=Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx")
    result = run(video, out_dir=tmp_path / "out", config=config)
    assert result.frames > 0
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
