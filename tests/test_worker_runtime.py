import json
import threading
from pathlib import Path

import pytest

from eval.eval_agent import build_scenario
from marg.agent.llm import BedrockLLM
from marg.worker import process_job, worker_loop


class MemoryStore:
    def __init__(self):
        self.status = {"survey_id": "job-1", "status": "queued", "video_key": "input/video.mp4", "created_at": "original"}
        self.uploads = []
        self.artifacts = {}

    def survey_status(self, survey_id):
        return self.status.copy()

    def update_survey_status(self, survey_id, status, **values):
        self.status.update({"status": status, **values})

    def download_file(self, key, path):
        path.write_bytes(b"test video")
        return path

    def upload_directory(self, survey_id, directory):
        paths = {path.relative_to(directory).as_posix(): path.read_text() for path in Path(directory).rglob("*") if path.is_file()}
        self.uploads.append(paths)
        self.artifacts.update(paths)


def test_provider_unavailable_does_not_discard_successful_vision(tmp_path, monkeypatch):
    store = MemoryStore()
    monkeypatch.delenv("MARG_BEDROCK_ENABLED", raising=False)

    def pipeline(video, gpx, output, config):
        output.mkdir()
        (output / "evidence.txt").write_text("vision evidence")
        return build_scenario("S2").result

    monkeypatch.setattr("marg.worker.run_pipeline", pipeline)
    monkeypatch.setattr("marg.worker.DNNDetector", lambda *args: object())
    monkeypatch.setattr("marg.worker.default_llm", lambda: BedrockLLM())
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: pytest.fail("Must not call AWS"))
    process_job({"survey_id": "job-1"}, store)
    assert store.status["status"] == "done"
    assert store.status["vision_status"] == "completed"
    assert store.status["agent_status"] == "unavailable"
    assert store.status["agent_error"]["code"] == "BedrockDisabled"
    assert store.status["video_key"] == "input/video.mp4"
    assert "result.json" in store.uploads[0]
    assert "agent/agent_result.json" not in store.uploads[0]
    result = json.loads(store.artifacts["agent/agent_result.json"])
    assert result["provider"]["kind"] == "bedrock"
    assert result["provider"]["live_inference"] is False
    assert store.artifacts["evidence.txt"] == "vision evidence"
    process_job({"survey_id": "job-1"}, store)
    assert len(store.uploads) == 2  # Completed duplicate performs no vision/inference.


def test_download_failure_is_persisted_for_retry(monkeypatch):
    store = MemoryStore()
    store.download_file = lambda *args: (_ for _ in ()).throw(RuntimeError("download failure"))
    with pytest.raises(RuntimeError, match="download failure"):
        process_job({"survey_id": "job-1"}, store)
    assert store.status["status"] == "failed"
    assert store.status["video_key"] == "input/video.mp4"


def test_worker_survives_job_failure_and_only_acknowledges_success(monkeypatch):
    stop = threading.Event()
    messages = ["[]", '{"survey_id":"broken"}', '{"survey_id":"good"}']
    deleted = []

    class Queue:
        def receive_message(self, **kwargs):
            body = messages.pop(0)
            if not messages:
                stop.set()
            return {"Messages": [{"Body": body, "ReceiptHandle": body}]}

        def delete_message(self, **kwargs):
            deleted.append(kwargs["ReceiptHandle"])

    def process(body, store):
        if body["survey_id"] == "broken":
            raise RuntimeError("vision execution failed")

    monkeypatch.setenv("MARG_SQS_QUEUE_URL", "test-queue")
    monkeypatch.setenv("MARG_S3_BUCKET", "test-bucket")
    monkeypatch.setattr("marg.worker.S3SurveyStore", lambda *args: object())
    monkeypatch.setattr("marg.worker.boto3.client", lambda *args, **kwargs: Queue())
    monkeypatch.setattr("marg.worker.process_job", process)
    worker_loop(stop)
    assert deleted == ['{"survey_id":"good"}']
