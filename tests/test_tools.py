from pathlib import Path

from marg.agent.tools import ToolContext, ToolSet
from marg.store.local import LocalStore
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.models import Segment, SurveyInstance, SurveyResult


def test_inspect_roi_on_bengaluru_fixture(tmp_path: Path) -> None:
    root = Path(__file__).parent / "fixtures"
    result = SurveyResult(
        survey_id="tool-test",
        video="synthetic",
        geo_source="synthetic",
        frames=1,
        processed_frames=1,
        keyframes=1,
        detections={"D00": 0, "D10": 0, "D20": 0, "D40": 1},
        instances=[
            SurveyInstance(
                id=1,
                bbox=[572.0, 186.0, 352.0, 118.0],
                class_name="D40",
                confidence=0.4,
                fused_conf=0.4,
                severity=4,
                area_m2=0.3,
                lat=28.6,
                lon=77.2,
                keyframe_ids=[0],
                frame_confs=[0.4],
                frame_classes=["D40"],
            )
        ],
        segments=[Segment(id=0, instance_ids=[1], lat=28.6, lon=77.2, radius_m=25)],
        keyframe_paths=["fixtures/pothole_bengaluru.jpg"],
    )
    context = ToolContext(
        result=result,
        store=LocalStore(tmp_path / "store"),
        detector=DNNDetector(Path(__file__).parents[1] / "models" / "rdd_yolov8s.onnx", VisionConfig()),
        keyframe_dir=root,
    )
    output = ToolSet(context).inspect_roi(1)
    assert output["confirmed"] is True
    assert output["detections"]
    assert float(output["best_match_conf"]) >= 0.5
    assert float(output["iou"]) >= 0.3
    assert (root.parent / "agent_crops" / str(output["crop_path"])).exists()
