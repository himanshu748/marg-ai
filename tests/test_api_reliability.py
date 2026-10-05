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
        self.deleted = []

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
        return self.status if survey_id == self.status["survey_id"] else {}

    def upload_file(self, path, key):
        self.uploaded.append(key)
        self.objects[key] = Path(path).read_bytes()

    def delete_file(self, key):
        self.deleted.append(key)
        self.objects.pop(key, None)

    def save_json(self, survey_id, name, payload):
        self.objects[self.object_key(survey_id, name)] = json.dumps(payload).encode()

    def save_work_order(self, *args):
        pass

    def update_survey_status(self, survey_id, status, **kwargs):
        if self.status["survey_id"] != survey_id:
            self.status = {"survey_id": survey_id}
        if status is not None:
            self.status["status"] = status
        self.status.update(kwargs)


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


@pytest.mark.parametrize("with_gpx", [False, True])
def test_cloud_queue_failure_cleans_originals_and_allows_retry(monkeypatch, tmp_path, with_gpx):
    from botocore.exceptions import ClientError

    from marg.worker import process_job

    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "fixture-queue")

    class Queue:
        def __init__(self):
            self.calls = []

        def send_message(self, **kwargs):
            self.calls.append(json.loads(kwargs["MessageBody"]))
            if len(self.calls) == 1:
                raise ClientError({"Error": {"Code": "ServiceUnavailable"}}, "SendMessage")
            return {"MessageId": "fixture"}

    queue = Queue()
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: queue)
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    files = {"video": ("clip.mp4", clip)}
    if with_gpx:
        files["gpx"] = ("track.gpx", b'<gpx version="1.1"><trk><trkseg><trkpt lat="28.6" lon="77.2"><time>2026-10-04T00:00:00Z</time></trkpt></trkseg></trk></gpx>')
    application = create_app(cache)
    with TestClient(application) as client:
        failure = client.post("/api/surveys/upload", files=files)
        assert failure.status_code == 503
        failed_status = cloud.status.copy()
        failed_job = application.state.jobs.get(str(failed_status["job_id"]))
        assert failed_status["status"] == failed_job["status"] == "failed"
        assert failed_status["error"] == failed_job["error"]
        assert failed_status["finished_at"] == failed_job["finished_at"]
        assert failed_status["video_key"] == failed_status["gpx_key"] == ""
        assert failed_status["admission_status"] == "failed"
        assert cloud.deleted == cloud.uploaded
        assert len(cloud.deleted) == (2 if with_gpx else 1)
        assert not any(key.startswith("uploads/") for key in cloud.objects)
        assert not list((cache / ".marg" / "uploads").iterdir())
        listed = next(row for row in client.get("/api/surveys").json() if row["survey_id"] == failed_status["survey_id"])
        assert listed["status"] == "failed"

        # A late delivery after an ambiguous send error must acknowledge the
        # failed admission without reading originals or restarting processing.
        monkeypatch.setattr(cloud, "download_file", lambda *args: pytest.fail("Failed admission must not download"))
        process_job(queue.calls[0], cloud)
        assert cloud.status == failed_status

        retry = client.post("/api/surveys/upload", files=files)
        assert retry.status_code == 202
        retried = retry.json()
        assert retried["survey_id"] != failed_status["survey_id"]
        assert cloud.status["status"] == "queued"
        assert cloud.status["survey_id"] == retried["survey_id"]
        assert cloud.status["video_key"] in cloud.objects
        if with_gpx:
            assert cloud.status["gpx_key"] in cloud.objects
        assert len(queue.calls) == 2
        assert client.get(retried["status_url"]).json()["status"] == "queued"
        assert client.get(failed_job["status_url"]).json()["status"] == "failed"


@pytest.mark.parametrize("compensation_failure", ["status", "delete"])
def test_cloud_upload_compensation_attempts_each_step(monkeypatch, tmp_path, caplog, compensation_failure):
    from botocore.exceptions import ClientError

    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "fixture-queue")
    queued_jobs = []
    original_update = cloud.update_survey_status
    original_delete = cloud.delete_file

    def update(survey_id, status, **values):
        if status == "queued":
            queued_jobs.append(values["job_id"])
        if status == "failed" and compensation_failure == "status":
            raise RuntimeError("status unavailable")
        original_update(survey_id, status, **values)

    attempted_deletes = []

    def delete(key):
        attempted_deletes.append(key)
        if len(attempted_deletes) == 1 and compensation_failure == "delete":
            raise RuntimeError("delete unavailable")
        original_delete(key)

    class Queue:
        def send_message(self, **kwargs):
            raise ClientError({"Error": {"Code": "QueueUnavailable"}}, "SendMessage")

    monkeypatch.setattr(cloud, "update_survey_status", update)
    monkeypatch.setattr(cloud, "delete_file", delete)
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: Queue())
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    files = {
        "video": ("clip.mp4", clip),
        "gpx": ("track.gpx", b'<gpx version="1.1"><trk><trkseg><trkpt lat="28.6" lon="77.2"><time>2026-10-04T00:00:00Z</time></trkpt></trkseg></trk></gpx>'),
    }
    application = create_app(cache)
    with TestClient(application) as client:
        assert client.post("/api/surveys/upload", files=files).status_code == 503
        assert application.state.jobs.get(queued_jobs[0])["status"] == "failed"
        assert attempted_deletes == cloud.uploaded
        assert not list((cache / ".marg" / "uploads").iterdir())
        assert "QueueUnavailable" in caplog.text  # Original queue error survives.
        if compensation_failure == "status":
            assert "Could not publish failed upload status" in caplog.text
            assert not any(key.startswith("uploads/") for key in cloud.objects)
        else:
            assert "Could not remove failed upload original" in caplog.text
            assert cloud.status["status"] == "failed"
            assert cloud.uploaded[1] not in cloud.objects


def test_cloud_delivery_during_failed_send_waits_for_admission(monkeypatch, tmp_path):
    from botocore.exceptions import ClientError

    from marg.worker import process_job

    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "fixture-queue")
    monkeypatch.setattr("marg.worker.run_pipeline", lambda *args: pytest.fail("Pending admission must not process"))
    monkeypatch.setattr(cloud, "download_file", lambda *args: pytest.fail("Pending admission must not download"))
    delivered = []

    class Queue:
        def send_message(self, **kwargs):
            body = json.loads(kwargs["MessageBody"])
            delivered.append(body)
            assert cloud.status["admission_status"] == "pending"
            with pytest.raises(RuntimeError, match="admission is pending"):
                process_job(body, cloud)
            assert cloud.status["status"] == "queued"
            raise ClientError({"Error": {"Code": "QueueUnavailable"}}, "SendMessage")

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: Queue())
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    application = create_app(cache)
    with TestClient(application) as client:
        response = client.post("/api/surveys/upload", files={"video": ("clip.mp4", clip)})
        assert response.status_code == 503
        assert cloud.status["status"] == "failed"
        assert cloud.status["admission_status"] == "failed"
        assert application.state.jobs.get(cloud.status["job_id"])["status"] == "failed"
        assert cloud.deleted == cloud.uploaded
        final_status = cloud.status.copy()
        process_job(delivered[0], cloud)
        assert cloud.status == final_status


def test_cloud_send_accepted_but_admission_write_fails_requires_reconciliation(monkeypatch, tmp_path, caplog):
    from marg.worker import process_job

    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "fixture-queue")
    original_update = cloud.update_survey_status
    messages = []

    def update(survey_id, status, **values):
        if values.get("admission_status") == "accepted":
            raise RuntimeError("status write unavailable")
        original_update(survey_id, status, **values)

    class Queue:
        def send_message(self, **kwargs):
            messages.append(json.loads(kwargs["MessageBody"]))
            return {"MessageId": "fixture"}

    monkeypatch.setattr(cloud, "update_survey_status", update)
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: Queue())
    monkeypatch.setattr("marg.worker.run_pipeline", lambda *args: pytest.fail("Pending admission must not process"))
    monkeypatch.setattr(cloud, "download_file", lambda *args: pytest.fail("Pending admission must not download"))
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    application = create_app(cache)
    with TestClient(application) as client:
        response = client.post("/api/surveys/upload", files={"video": ("clip.mp4", clip)})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "do not re-upload" in detail["message"]
        job = client.get(detail["status_url"]).json()
        assert job["job_id"] == detail["job_id"]
        assert job["status"] == "queued"
        assert job["admission_status"] == "pending"
        assert job["error"] == detail["message"]
        assert application.state.jobs.get(job["job_id"])["error"] == detail["message"]
        assert cloud.status["status"] == "queued"
        assert cloud.status["admission_status"] == "pending"
        assert cloud.status["video_key"] in cloud.objects
        assert not cloud.deleted
        assert not list((cache / ".marg" / "uploads").iterdir())
        assert "Could not confirm queued upload admission" in caplog.text
        before_delivery = cloud.status.copy()
        with pytest.raises(RuntimeError, match="admission is pending"):
            process_job(messages[0], cloud)
        assert cloud.status == before_delivery

        # Once an operator confirms admission (or a timed-out write becomes
        # visible), polling must clear the local reconciliation diagnostic.
        original_update(cloud.status["survey_id"], None, admission_status="accepted")
        reconciled = client.get(detail["status_url"]).json()
        assert reconciled["status"] == "queued"
        assert reconciled["admission_status"] == "accepted"
        assert reconciled["error"] is None


def test_cloud_accepted_upload_survives_staging_cleanup_failure(monkeypatch, tmp_path):
    cache, _, cloud = configure_cloud(monkeypatch, tmp_path)
    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "fixture-queue")

    class Queue:
        def send_message(self, **kwargs):
            return {"MessageId": "fixture"}

    def cleanup(path, *, ignore_errors=False):
        if not ignore_errors:
            raise OSError("staging filesystem unavailable")

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: Queue())
    monkeypatch.setattr(api.shutil, "rmtree", cleanup)
    clip = (Path(__file__).parent / "fixtures" / "clip3s.mp4").read_bytes()
    with TestClient(create_app(cache)) as client:
        response = client.post("/api/surveys/upload", files={"video": ("clip.mp4", clip)})
        assert response.status_code == 202
        assert cloud.status["status"] == "queued"
        assert cloud.status["video_key"] in cloud.objects
        assert not cloud.deleted
        assert client.get(response.json()["status_url"]).json()["status"] == "queued"


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
