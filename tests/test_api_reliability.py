import importlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marg.api.app import create_app
from marg.api.runtime import JobFailure, JobRuntime, QueueFull
from marg.vision.models import SurveyResult

api = importlib.import_module("marg.api.app")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in (
        "MARG_READ_ONLY",
        "MARG_API_TOKEN",
        "MARG_COOKIE_SECURE",
        "MARG_S3_BUCKET",
        "MARG_SQS_QUEUE_URL",
        "MARG_BEDROCK_ENABLED",
        "MARG_MAX_VIDEO_BYTES",
        "MARG_JOB_CAPACITY",
    ):
        monkeypatch.delenv(name, raising=False)


def write_survey(root: Path, survey_id="demo") -> Path:
    directory = root / survey_id
    directory.mkdir(parents=True)
    result = SurveyResult(
        survey_id=survey_id,
        video="clip.mp4",
        geo_source="synthetic",
        frames=10,
        processed_frames=4,
        keyframes=1,
        detections={"D40": 0},
    )
    (directory / "result.json").write_text(result.model_dump_json())
    agent = directory / "agent"
    agent.mkdir()
    (agent / "agent_result.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "status": "completed",
                "audit": {"passed": True},
                "trace": [],
                "work_orders": [
                    {"work_order_id": f"wo-{index}", "status": "pending_approval"}
                    for index in range(3)
                ],
            }
        )
    )
    return directory


def wait_job(client, url, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(url).json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Job did not finish: {job}")


def test_local_capabilities_bedrock_disabled_and_static(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        capabilities = client.get("/api/capabilities").json()
        assert capabilities["storage"] == "local"
        assert capabilities["uploads"]["enabled"]
        assert capabilities["agent"]["modes"] == ["mock"]
        assert capabilities["agent"]["bedrock"]["readiness"] == "disabled"
        assert (
            client.post("/api/surveys/missing/run_agent?llm=bedrock").status_code == 403
        )
        assert client.get("/").status_code == 200
        assert client.get("/static/index.html").status_code == 200
        assert client.get("/static/../app.py").status_code == 404


def test_token_session_protects_reads_media_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MARG_API_TOKEN", "test-workspace-secret")
    directory = write_survey(tmp_path)
    (directory / "evidence").mkdir()
    (directory / "evidence" / "one.jpg").write_bytes(b"image")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/session").json()["authenticated"] is False
        for path in ("/api/surveys", "/api/surveys/demo/evidence/one.jpg"):
            assert client.get(path).status_code == 401
        assert (
            client.post(
                "/api/session", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        login = client.post(
            "/api/session", headers={"Authorization": "Bearer test-workspace-secret"}
        )
        assert login.status_code == 200
        assert "HttpOnly" in login.headers["set-cookie"]
        assert "SameSite=strict" in login.headers["set-cookie"]
        assert client.get("/api/surveys/demo/evidence/one.jpg").status_code == 200
        assert (
            client.post(
                "/api/surveys/demo/work_orders/wo-0/decision",
                json={"decision": "approve"},
                headers={"Origin": "https://unrelated.example"},
            ).status_code
            == 403
        )
        assert client.delete("/api/session").status_code == 200
        assert client.get("/api/surveys").status_code == 401


def test_read_only_covers_all_mutations(monkeypatch, tmp_path):
    monkeypatch.setenv("MARG_READ_ONLY", "1")
    directory = write_survey(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/surveys/demo/run_agent").status_code == 403
        assert (
            client.post(
                "/api/surveys/demo/work_orders/wo-0/decision",
                json={"decision": "approve"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/surveys/upload",
                files={"video": ("clip.mp4", b"video", "video/mp4")},
            ).status_code
            == 403
        )
        assert client.get("/api/surveys/demo/agent").status_code == 200
        assert not (directory / "agent" / "decisions.json").exists()
        assert not (tmp_path / ".marg").exists()


def test_duplicate_agent_and_failure_are_observable(monkeypatch, tmp_path):
    write_survey(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def fail(*args):
        entered.set()
        assert release.wait(5)
        raise JobFailure("Review provider unavailable. Retry deterministic review.")

    monkeypatch.setattr(api, "_run_agent", fail)
    with TestClient(create_app(tmp_path)) as client:
        first = client.post("/api/surveys/demo/run_agent").json()
        assert entered.wait(5)
        duplicate = client.post("/api/surveys/demo/run_agent")
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"]["job_id"] == first["job_id"]
        assert (
            client.post(
                "/api/surveys/demo/work_orders/wo-0/decision",
                json={"decision": "approve"},
            ).status_code
            == 409
        )
        status = client.get("/api/surveys/demo/status").json()
        assert status["agent_job"]["job_id"] == first["job_id"]
        # Reviewing again must not hide an already completed vision survey.
        assert client.get("/api/surveys").json()[0]["status"] == "done"
        release.set()
        finished = wait_job(client, first["status_url"])
        assert finished["status"] == "failed"
        assert "provider unavailable" in finished["error"]
        assert all(
            finished[key]
            for key in ("created_at", "started_at", "finished_at", "updated_at")
        )
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(first["status_url"]).json()["status"] == "failed"


def test_agent_queue_bound(monkeypatch, tmp_path):
    monkeypatch.setenv("MARG_JOB_CAPACITY", "1")
    write_survey(tmp_path, "first")
    write_survey(tmp_path, "second")
    release = threading.Event()
    monkeypatch.setattr(
        api, "_run_agent", lambda *args: release.wait(5) and {"status": "completed"}
    )
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/surveys/first/run_agent").status_code == 200
        response = client.post("/api/surveys/second/run_agent")
        assert response.status_code == 429
        assert response.headers["retry-after"] == "5"
        release.set()


def test_atomic_concurrent_decisions_and_final_state(tmp_path):
    directory = write_survey(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(
                executor.map(
                    lambda i: client.post(
                        f"/api/surveys/demo/work_orders/wo-{i}/decision",
                        json={"decision": "approve", "note": f"review {i}"},
                    ),
                    range(3),
                )
            )
        assert all(response.status_code == 200 for response in responses)
        saved = json.loads((directory / "agent" / "decisions.json").read_text())
        assert len(saved) == 3
        assert all(
            value["decided_at"] and value["agent_run_id"] == "run-a"
            for value in saved.values()
        )
        retry = client.post(
            "/api/surveys/demo/work_orders/wo-0/decision", json={"decision": "approve"}
        )
        assert retry.status_code == 200
        assert retry.json()["decision_note"] == "review 0"
        assert (
            client.post(
                "/api/surveys/demo/work_orders/wo-0/decision",
                json={"decision": "reject"},
            ).status_code
            == 409
        )


def test_new_run_does_not_inherit_stale_human_decision(tmp_path):
    directory = write_survey(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/api/surveys/demo/work_orders/wo-0/decision",
                json={"decision": "approve"},
            ).status_code
            == 200
        )
        path = directory / "agent" / "agent_result.json"
        payload = json.loads(path.read_text())
        payload["run_id"] = "run-b"
        path.write_text(json.dumps(payload))
        order = client.get("/api/surveys/demo/agent").json()["work_orders"][0]
        assert order["status"] == "pending_approval"
        assert "decision" not in order


@pytest.mark.parametrize(
    "status,audit", [("failed", True), ("unavailable", True), ("completed", False)]
)
def test_incomplete_review_cannot_be_approved(tmp_path, status, audit):
    directory = write_survey(tmp_path)
    path = directory / "agent" / "agent_result.json"
    payload = json.loads(path.read_text())
    payload.update(status=status, audit={"passed": audit})
    path.write_text(json.dumps(payload))
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/api/surveys/demo/work_orders/wo-0/decision",
                json={"decision": "approve"},
            ).status_code
            == 409
        )


def test_media_and_result_symlinks_cannot_escape(tmp_path):
    directory = write_survey(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-secret"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"private")
    (directory / "evidence").symlink_to(outside, target_is_directory=True)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/surveys/demo/evidence/secret.jpg").status_code == 404
        (directory / "result.json").unlink()
        (directory / "result.json").symlink_to(outside / "secret.jpg")
        assert client.get("/api/surveys/demo").status_code == 404


@pytest.mark.parametrize(
    "filename,content,expected",
    [
        ("clip.exe", b"data", 415),
        ("clip.mp4", b"", 422),
        ("clip.mp4", b"not a video", 422),
    ],
)
def test_upload_rejects_invalid_video_and_cleans_staging(
    tmp_path, filename, content, expected
):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/surveys/upload",
            files={"video": (filename, content, "application/octet-stream")},
        )
        assert response.status_code == expected
        assert client.get("/api/surveys").json() == []
        assert not list((tmp_path / ".marg" / "uploads").glob("*"))


def test_upload_size_bound_and_gpx_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("MARG_MAX_VIDEO_BYTES", "10")
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/api/surveys/upload", files={"video": ("clip.mp4", b"x" * 11)}
            ).status_code
            == 413
        )
    monkeypatch.delenv("MARG_MAX_VIDEO_BYTES")
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/surveys/upload",
            files={"video": ("clip.mp4", clip), "gpx": ("track.gpx", b"<gpx/>")},
        )
        assert response.status_code == 422
        assert "timestamped" in response.json()["detail"]
        assert client.get("/api/surveys").json() == []


def test_local_upload_runs_without_aws_and_cleans_original(monkeypatch, tmp_path):
    # Real multipart/video decoding and job persistence, isolated model execution.
    fixture = SurveyResult(
        survey_id="old",
        video="old",
        geo_source="synthetic",
        frames=3,
        processed_frames=1,
        keyframes=0,
        detections={},
    )
    monkeypatch.setattr(
        api, "run_pipeline", lambda *args: (args[2].mkdir(parents=True), fixture)[1]
    )
    monkeypatch.setattr(api, "_run_agent", lambda *args: {"status": "completed"})
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/surveys/upload",
            data={"name": "South road"},
            files={"video": ("../../clip.mp4", clip, "video/mp4")},
        )
        assert response.status_code == 202
        job = response.json()
        assert job["survey_id"].startswith("south-road-")
        finished = wait_job(client, job["status_url"])
        assert finished["status"] == "succeeded"
        assert finished["agent_status"] == "completed"
        result = client.get(f"/api/surveys/{job['survey_id']}").json()
        assert result["survey_id"] == job["survey_id"]
        assert result["video"] == "video.mp4"
        assert not list((tmp_path / ".marg" / "uploads").glob("*"))


def test_failed_pipeline_is_durable_and_visible(monkeypatch, tmp_path):
    def fail(*args):
        raise JobFailure("Privacy models are unavailable")

    monkeypatch.setattr(api, "run_pipeline", fail)
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/surveys/upload", files={"video": ("clip.mp4", clip)}
        )
        job = wait_job(client, response.json()["status_url"])
        assert job["status"] == "failed"
        assert job["error"] == "Privacy models are unavailable"
        listed = client.get("/api/surveys").json()[0]
        assert listed["status"] == "failed"
        assert listed["error"] == job["error"]
        assert not list((tmp_path / ".marg" / "uploads").glob("*"))


def test_runtime_recovers_dead_process_without_replaying(tmp_path):
    runtime = JobRuntime(tmp_path)
    job = runtime.create("demo", "agent")
    with sqlite3.connect(runtime.path) as connection:
        connection.execute(
            "UPDATE jobs SET owner=? WHERE id=?", ("999999999-dead", job["job_id"])
        )
    restarted = JobRuntime(tmp_path)
    restarted.recover()
    recovered = restarted.get(job["job_id"])
    assert recovered["status"] == "failed"
    assert "server restart" in recovered["error"]
    assert recovered["finished_at"]
    restarted.shutdown()
    with pytest.raises(QueueFull):
        restarted.create("another", "agent")


def test_corrupt_survey_does_not_break_other_surveys(tmp_path):
    write_survey(tmp_path)
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "result.json").write_text("{bad")
    with TestClient(create_app(tmp_path)) as client:
        surveys = {value["id"]: value for value in client.get("/api/surveys").json()}
        assert surveys["demo"]["status"] == "done"
        assert surveys["broken"]["status"] == "failed"
        assert client.get("/api/surveys/broken").status_code == 500


def test_cloud_queue_preflight_prevents_orphan_uploads(monkeypatch, tmp_path):
    class FakeStore:
        def __init__(self, *args):
            pass

        def upload_file(self, *args):
            raise AssertionError("Upload must not start without an SQS queue")

    monkeypatch.setenv("MARG_S3_BUCKET", "fixture-bucket")
    monkeypatch.setenv("MARG_S3_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(api, "S3SurveyStore", FakeStore)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/capabilities").json()["uploads"]["enabled"] is False
        response = client.post(
            "/api/surveys/upload", files={"video": ("clip.mp4", b"video")}
        )
        assert response.status_code == 503
        assert "queue" in response.json()["detail"]
        assert not (tmp_path / ".marg").exists()


def test_stale_cloud_decision_returns_conflict_without_local_write(
    monkeypatch, tmp_path
):
    from botocore.exceptions import ClientError

    class FakeStore:
        def __init__(self, *args):
            pass

        def object_key(self, survey_id, name):
            return f"{survey_id}/{name}"

        def download_file(self, key, path):
            if not Path(path).is_file():
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return path

        def survey_status(self, survey_id):
            return {"status": "done"}

        def save_decision(self, *args):
            raise ClientError(
                {
                    "Error": {
                        "Code": "ConditionalCheckFailedException",
                        "Message": "stale",
                    }
                },
                "UpdateItem",
            )

    directory = write_survey(tmp_path)
    monkeypatch.setenv("MARG_S3_BUCKET", "fixture-bucket")
    monkeypatch.setenv("MARG_S3_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(api, "S3SurveyStore", FakeStore)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/surveys/demo/work_orders/wo-0/decision", json={"decision": "approve"}
        )
        assert response.status_code == 409
        assert "Refresh" in response.json()["detail"]
        assert not (directory / "agent" / "decisions.json").exists()


def test_runtime_recovers_reused_container_pid(tmp_path):
    import os

    runtime = JobRuntime(tmp_path)
    job = runtime.create("demo", "agent")
    with sqlite3.connect(runtime.path) as connection:
        connection.execute(
            "UPDATE jobs SET owner=? WHERE id=?",
            (f"{os.getpid()}-previous-container", job["job_id"]),
        )
    restarted = JobRuntime(tmp_path)
    restarted.recover()
    assert restarted.get(job["job_id"])["status"] == "failed"


class MemoryCloud:
    def __init__(self, cache):
        self.cache = cache
        self.objects = {}
        self.status = {
            "survey_id": "demo",
            "status": "processing",
            "agent_status": "running",
        }
        self.uploaded = []

    def object_key(self, survey_id, suffix=""):
        return f"{survey_id}/{suffix}".rstrip("/")

    def download_file(self, key, path):
        from botocore.exceptions import ClientError

        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])
        return destination

    def sync_survey(self, survey_id):
        self.download_file(
            f"{survey_id}/result.json", self.cache / survey_id / "result.json"
        )
        return self.cache / survey_id

    def list_survey_ids(self):
        return ["demo"]

    def list_statuses(self):
        return [self.status]

    def survey_status(self, survey_id):
        return self.status if survey_id == "demo" else {}

    def upload_file(self, path, key):
        self.uploaded.append(key)
        self.objects[key] = Path(path).read_bytes()

    def save_json(self, survey_id, name, payload):
        self.objects[self.object_key(survey_id, name)] = json.dumps(payload).encode()

    def save_work_order(self, *args):
        pass

    def update_survey_status(self, survey_id, status, **kwargs):
        self.status = {"survey_id": survey_id, "status": status, **kwargs}


def configure_cloud(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    remote = write_survey(tmp_path / "remote")
    cloud = MemoryCloud(cache)
    cloud.objects["demo/result.json"] = (remote / "result.json").read_bytes()
    monkeypatch.setenv("MARG_S3_BUCKET", "fixture-bucket")
    monkeypatch.setenv("MARG_S3_CACHE_DIR", str(cache))
    monkeypatch.setattr(api, "S3SurveyStore", lambda *args: cloud)
    return cache, remote, cloud


def test_cloud_metadata_published_after_first_list_is_visible(monkeypatch, tmp_path):
    cache, remote, cloud = configure_cloud(monkeypatch, tmp_path)
    with TestClient(create_app(cache)) as client:
        listed = client.get("/api/surveys").json()[0]
        assert listed["status"] == "processing"
        assert listed["agent_status"] == "running"
        assert client.get("/api/surveys/demo/agent").status_code == 404
        cloud.objects["demo/agent/agent_result.json"] = (
            remote / "agent" / "agent_result.json"
        ).read_bytes()
        cloud.status.update(status="done", agent_status="completed")
        assert client.get("/api/surveys/demo/agent").json()["run_id"] == "run-a"
        cloud.save_json(
            "demo",
            "agent/decisions.json",
            {
                "wo-0": {
                    "decision": "approve",
                    "status": "approved",
                    "agent_run_id": "run-a",
                }
            },
        )
        assert (
            client.get("/api/surveys/demo/agent").json()["work_orders"][0]["status"]
            == "approved"
        )
        replacement = json.loads(cloud.objects["demo/agent/agent_result.json"])
        replacement["run_id"] = "run-b"
        cloud.save_json("demo", "agent/agent_result.json", replacement)
        fresh = client.get("/api/surveys/demo/agent").json()
        assert fresh["run_id"] == "run-b"
        assert fresh["work_orders"][0]["status"] == "pending_approval"
        updated_result = json.loads(cloud.objects["demo/result.json"])
        updated_result["frames"] = 40
        cloud.save_json("demo", "result.json", updated_result)
        assert client.get("/api/surveys/demo").json()["frames"] == 40
        assert client.get("/api/surveys").json()[0]["status"] == "done"


def test_cloud_processing_blocks_stale_decision(monkeypatch, tmp_path):
    cache, remote, cloud = configure_cloud(monkeypatch, tmp_path)
    cloud.objects["demo/agent/agent_result.json"] = (
        remote / "agent" / "agent_result.json"
    ).read_bytes()
    with TestClient(create_app(cache)) as client:
        response = client.post(
            "/api/surveys/demo/work_orders/wo-0/decision", json={"decision": "approve"}
        )
        assert response.status_code == 409
        assert "cloud processing" in response.json()["detail"]


def test_cloud_rerun_hydrates_images_and_publishes_inspection_crop(
    monkeypatch, tmp_path
):
    from marg.vision.models import Segment, SurveyInstance

    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    cloud.status.update(status="done", agent_status="completed")
    result = SurveyResult.model_validate_json(cloud.objects["demo/result.json"])
    result.instances = [
        SurveyInstance(
            id=0,
            bbox=[572, 186, 352, 118],
            class_name="D40",
            confidence=0.4,
            fused_conf=0.4,
            severity=5,
            area_m2=0.5,
            lat=28.6,
            lon=77.2,
            keyframe_ids=[0],
            frame_confs=[0.4],
            frame_classes=["D40"],
            evidence_path="evidence/inst_0000.jpg",
        )
    ]
    result.segments = [Segment(id=0, instance_ids=[0], lat=28.6, lon=77.2, radius_m=25)]
    result.keyframe_paths = ["keyframes/kf_00000.jpg"]
    cloud.objects["demo/result.json"] = result.model_dump_json().encode()
    image = (Path(__file__).parent / "fixtures" / "pothole_bengaluru.jpg").read_bytes()
    cloud.objects["demo/evidence/inst_0000.jpg"] = image
    cloud.objects["demo/evidence_raw/inst_0000.jpg"] = image
    cloud.objects["demo/keyframes/kf_00000.jpg"] = image

    class Detector:
        def detect(self, frame):
            return []

    monkeypatch.setattr(api, "DNNDetector", lambda *args: Detector())
    monkeypatch.setattr("marg.agent.tools.detect_sensitive_regions", lambda frame: [])
    with TestClient(create_app(cache)) as client:
        assert client.get("/api/surveys/demo").status_code == 200
        assert not (cache / "demo" / "evidence").exists()
        job = client.post("/api/surveys/demo/run_agent").json()
        completed = wait_job(client, job["status_url"])
        assert completed["status"] == "succeeded"
        payload = client.get("/api/surveys/demo/agent").json()
        inspection = next(
            item for item in payload["trace"] if item["tool"] == "inspect_roi"
        )
        assert "error" not in inspection["output"]
        assert inspection["output"]["crop_path"] == "inspect_inst_0000.jpg"
        assert "demo/agent_crops/inspect_inst_0000.jpg" in cloud.uploaded
        assert (cache / "demo" / "evidence_raw" / "inst_0000.jpg").is_file()


def test_cloud_uploads_do_not_consume_local_executor_capacity(monkeypatch, tmp_path):
    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "fixture-queue")
    monkeypatch.setenv("MARG_JOB_CAPACITY", "1")

    class Queue:
        def send_message(self, **kwargs):
            return {"MessageId": "fixture"}

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: Queue())
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    with TestClient(create_app(cache)) as client:
        first = client.post("/api/surveys/upload", files={"video": ("first.mp4", clip)})
        assert first.status_code == 202
        cloud.status["status"] = "done"
        # No job/status/list polling occurs before the next admission.
        second = client.post(
            "/api/surveys/upload", files={"video": ("second.mp4", clip)}
        )
        assert second.status_code == 202
