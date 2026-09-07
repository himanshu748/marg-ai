import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from marg.agent.llm import default_llm, provider_status
from marg.agent.loop import AgentRun, run_agent
from marg.agent.tools import ToolContext
from marg.agent.trace import Trace
from marg.store.aws import S3SurveyStore
from marg.store.local import (
    LocalStore,
    atomic_write_json,
    atomic_write_text,
    validate_survey_id,
)
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)
VISIBILITY_TIMEOUT = 900


def process_job(message: dict[str, object], store: S3SurveyStore) -> None:
    raw_id = message.get("survey_id")
    if not isinstance(raw_id, str):
        raise TypeError("A survey_id string is required")
    survey_id = validate_survey_id(raw_id)
    status = store.survey_status(survey_id)
    if status.get("status") == "done":
        logger.info("Skipping completed survey %s", survey_id)
        return
    if not isinstance(status.get("video_key"), str) or not status["video_key"]:
        raise ValueError("Survey upload metadata is missing")
    vision_complete = False
    try:
        store.update_survey_status(
            survey_id, "processing", vision_status="processing", agent_status="pending", error=None
        )
        with tempfile.TemporaryDirectory(prefix=f"marg-{survey_id}-") as temporary:
            workdir = Path(temporary)
            video_key = str(status["video_key"])
            video_path = store.download_file(video_key, workdir / Path(video_key).name)
            gpx_key = str(status.get("gpx_key") or "")
            gpx_path = store.download_file(gpx_key, workdir / Path(gpx_key).name) if gpx_key else None
            output = workdir / "survey"
            config = VisionConfig()
            result = run_pipeline(video_path, gpx_path, output, config)
            result.survey_id = survey_id
            atomic_write_text(output / "result.json", result.model_dump_json(indent=2))
            # Commit vision artifacts before a remote provider can fail or time out.
            store.upload_directory(survey_id, output)
            vision_complete = True
            store.update_survey_status(
                survey_id, "processing", vision_status="completed", agent_status="running",
                frames=result.frames, keyframes=result.keyframes,
                detections=sum(result.detections.values()), instances=len(result.instances),
                runtime=result.metrics.get("runtime_s", 0.0),
            )
            try:
                detector = DNNDetector(config.model_file(Path(__file__).resolve().parents[1]), config)
                context = ToolContext(
                    result=result, store=LocalStore(output / "agent" / "store"),
                    detector=detector, keyframe_dir=output / "keyframes",
                )
                agent_run = run_agent(result, default_llm(), context)
            except Exception:
                logger.exception("Agent review failed for survey %s; vision evidence is retained", survey_id)
                agent_run = AgentRun(
                    trace=Trace(), summary="Agent review failed. Vision evidence is available for review and retry.",
                    status="failed", provider=_configured_provider(),
                    error={"code": "AgentExecutionFailed", "message": "Agent review could not complete.", "retryable": True},
                )
            agent_dir = output / "agent"
            for order in agent_run.work_orders:
                order["agent_run_id"] = agent_run.run_id
                store.save_work_order(survey_id, str(order["work_order_id"]), order)
            agent_run.trace.write_jsonl(agent_dir / "trace.jsonl")
            atomic_write_json(agent_dir / "agent_result.json", agent_run.to_dict())
            # The second upload includes trace, decisions and redacted agent crops.
            store.upload_directory(survey_id, output)
            store.update_survey_status(
                survey_id, "done", vision_status="completed", agent_status=agent_run.status,
                agent_provider=agent_run.provider, agent_error=agent_run.error,
                run_id=agent_run.run_id, completed_at=agent_run.completed_at,
            )
    except Exception:
        logger.exception("Survey %s failed during processing or persistence", survey_id)
        try:
            store.update_survey_status(
                survey_id, "failed", vision_status="completed" if vision_complete else "failed",
                error="Survey processing or persistence failed. The queue will retry this job.",
            )
        except Exception:
            logger.exception("Could not persist failure status for survey %s", survey_id)
        raise


def _configured_provider() -> dict[str, object]:
    try:
        return provider_status()
    except ValueError:
        return {"kind": "invalid", "live_inference": False, "readiness": "misconfigured"}


def _extend_visibility(sqs: object, queue_url: str, receipt: str, finished: threading.Event) -> None:
    # SQS may redeliver even within a visibility window; completed-job checks are
    # still necessary. Failed messages remain in the queue for retry / its DLQ.
    while not finished.wait(VISIBILITY_TIMEOUT / 3):
        try:
            sqs.change_message_visibility(
                QueueUrl=queue_url, ReceiptHandle=receipt, VisibilityTimeout=VISIBILITY_TIMEOUT
            )
        except (BotoCoreError, ClientError):
            logger.exception("Could not extend queue message visibility")


def worker_loop(stop_event: threading.Event | None = None) -> None:
    queue_url = os.environ.get("MARG_SQS_QUEUE_URL")
    bucket = os.environ.get("MARG_S3_BUCKET")
    if not queue_url or not bucket:
        return
    stopped = stop_event or threading.Event()
    store = S3SurveyStore(
        bucket, os.environ.get("MARG_S3_PREFIX", ""),
        os.environ.get("MARG_S3_CACHE_DIR", "/tmp/marg-cache"),
    )
    sqs = boto3.client(
        "sqs", config=Config(connect_timeout=5, read_timeout=25, retries={"mode": "standard", "total_max_attempts": 3})
    )
    while not stopped.is_set():
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
            )
        except (BotoCoreError, ClientError):
            logger.exception("Queue receive failed; worker will retry")
            stopped.wait(5)
            continue
        for message in response.get("Messages", []):
            finished = threading.Event()
            heartbeat = None
            try:
                receipt = message.get("ReceiptHandle")
                if not isinstance(receipt, str) or not receipt:
                    raise ValueError("Queue message is missing a receipt handle")
                body = json.loads(message["Body"])
                if not isinstance(body, dict):
                    raise TypeError("Queue message body must be an object")
                heartbeat = threading.Thread(
                    target=_extend_visibility, args=(sqs, queue_url, receipt, finished), daemon=True
                )
                heartbeat.start()
                process_job(body, store)
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            except Exception:
                # A malformed message or vision failure must not kill the only worker
                # or be acknowledged as a successfully completed survey.
                logger.exception("Queue job failed and remains available for retry")
            finally:
                finished.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    worker_loop()


if __name__ == "__main__":
    main()
