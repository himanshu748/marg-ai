import json
import shutil
import time
from pathlib import Path

from fastapi.testclient import TestClient

from marg.api.app import create_app
from marg.vision.models import Segment, SurveyInstance, SurveyResult


def _write_fixture(root: Path) -> None:
    survey = root / "demo"
    (survey / "keyframes").mkdir(parents=True)
    (survey / "crops").mkdir()
    (survey / "evidence").mkdir()
    fixture = Path(__file__).parent / "fixtures" / "pothole_bengaluru.jpg"
    shutil.copy2(fixture, survey / "keyframes" / "kf_00000.jpg")
    shutil.copy2(fixture, survey / "crops" / "inst_0000.jpg")
    shutil.copy2(fixture, survey / "evidence" / "inst_0000.jpg")
    result = SurveyResult(
        survey_id="demo",
        video="synthetic",
        geo_source="synthetic",
        frames=10,
        processed_frames=4,
        keyframes=1,
        detections={"D00": 0, "D10": 0, "D20": 0, "D40": 1},
        instances=[
            SurveyInstance(
                id=0,
                bbox=[450, 100, 600, 300],
                class_name="D40",
                confidence=0.4,
                fused_conf=0.4,
                severity=5,
                area_m2=0.5,
                lat=28.6139,
                lon=77.209,
                keyframe_ids=[0],
                frame_confs=[0.4],
                frame_classes=["D40"],
                evidence_path=str(survey / "evidence" / "inst_0000.jpg"),
            )
        ],
        segments=[Segment(id=0, instance_ids=[0], lat=28.6139, lon=77.209, radius_m=25)],
        keyframe_paths=[str(survey / "keyframes" / "kf_00000.jpg")],
        crop_paths=[str(survey / "crops" / "inst_0000.jpg")],
        metrics={"runtime_s": 1.25},
    )
    (survey / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


def test_api_dashboard_endpoints(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))

    surveys = client.get("/api/surveys")
    assert surveys.status_code == 200
    assert surveys.json()[0]["id"] == "demo"
    assert client.get("/api/surveys/demo").json()["survey_id"] == "demo"
    assert client.get("/api/surveys/demo/agent").status_code == 404
    assert client.get("/api/surveys/demo/keyframes/../result.json", follow_redirects=False).status_code in {400, 404}
    assert client.get("/api/surveys/demo/keyframes/kf_00000.jpg").status_code == 200
    assert client.get("/api/surveys/demo/evidence/inst_0000.jpg").status_code == 200

    started = client.post("/api/surveys/demo/run_agent?llm=mock")
    assert started.status_code == 200
    deadline = time.monotonic() + 20
    agent = None
    while time.monotonic() < deadline:
        response = client.get("/api/surveys/demo/agent")
        if response.status_code == 200:
            agent = response.json()
            break
        time.sleep(0.1)
    assert agent is not None
    assert agent["work_orders"]
    trace = client.get("/api/surveys/demo/trace")
    assert trace.status_code == 200
    assert trace.json()
    inspect_entries = [
        entry for entry in trace.json() if entry.get("tool") == "inspect_roi"
    ]
    assert inspect_entries
    crop_name = inspect_entries[0]["output"]["crop_path"]
    assert client.get(f"/api/surveys/demo/agent_crops/{crop_name}").status_code == 200

    work_order_id = agent["work_orders"][0]["work_order_id"]
    decision = client.post(
        f"/api/surveys/demo/work_orders/{work_order_id}/decision",
        json={"decision": "approve", "note": "Reviewed in dashboard"},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"
    decisions = json.loads((tmp_path / "demo" / "agent" / "decisions.json").read_text())
    assert decisions[work_order_id]["decision"] == "approve"
