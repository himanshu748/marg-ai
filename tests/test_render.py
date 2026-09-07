from pathlib import Path

import cv2

from marg.vision.config import VisionConfig
from marg.vision.models import Observation, SurveyInstance, SurveyResult
from marg.vision.render import render_annotated_video


def test_render_annotated_video_writes_frames(tmp_path: Path) -> None:
    video = Path(__file__).parent / "fixtures" / "clip3s.mp4"
    observation = Observation(
        frame_idx=0,
        t_s=0.0,
        bbox=[20.0, 20.0, 80.0, 60.0],
        class_name="D40",
        conf=0.8,
        fused_conf=0.8,
    )
    later = observation.model_copy(update={"frame_idx": 2, "t_s": 0.1})
    instance = SurveyInstance(
        id=1,
        bbox=observation.bbox,
        class_name="D40",
        confidence=0.8,
        fused_conf=0.8,
        severity=5,
        area_m2=2.0,
        lat=28.6,
        lon=77.2,
        observations=[observation, later],
    )
    result = SurveyResult(
        survey_id="test",
        video=str(video),
        geo_source="synthetic",
        frames=10,
        processed_frames=2,
        keyframes=1,
        detections={"D40": 2},
        instances=[instance],
    )
    output = tmp_path / "annotated.mp4"
    assert render_annotated_video(video, result, output, VisionConfig(), redact=False) > 0
    assert output.exists()
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    capture.release()
