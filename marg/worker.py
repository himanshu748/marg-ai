import json
import os
import tempfile
import threading
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from marg.agent.llm import default_llm
from marg.agent.loop import run_agent
from marg.agent.tools import ToolContext
from marg.store.aws import S3SurveyStore
from marg.store.local import LocalStore
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.pipeline import run as run_pipeline


def process_job(message: dict[str, object], store: S3SurveyStore) -> None:
    survey_id = str(message["survey_id"])
    status = store.survey_status(survey_id)
    created_at = status.get("created_at")
    store.update_survey_status(
        survey_id,
        "processing",
        **({"created_at": created_at} if created_at else {}),
    )
    with tempfile.TemporaryDirectory(prefix=f"marg-{survey_id}-") as temporary:
        workdir = Path(temporary)
        video_key = str(status["video_key"])
        video_path = store.download_file(video_key, workdir / Path(video_key).name)
        gpx_key = str(status.get("gpx_key", ""))
        gpx_path = store.download_file(gpx_key, workdir / Path(gpx_key).name) if gpx_key else None
        output = workdir / "survey"
        try:
            result = run_pipeline(video_path, gpx_path, output, VisionConfig())
            result.survey_id = survey_id
            (output / "result.json").write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
            detector = DNNDetector(
                VisionConfig().model_file(Path(__file__).resolve().parents[1]),
                VisionConfig(),
            )
            context = ToolContext(
                result=result,
                store=LocalStore(output / "agent" / "store"),
                detector=detector,
                keyframe_dir=output / "keyframes",
            )
            llm_kind = os.environ.get("MARG_AGENT_LLM", "mock")
            if "MARG_BEDROCK_MODEL_ID" in os.environ and "MARG_AGENT_LLM" not in os.environ:
                llm_kind = "bedrock"
            agent_run = run_agent(result, default_llm(llm_kind), context)
            agent_dir = output / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            agent_run.trace.write_jsonl(agent_dir / "trace.jsonl")
            (agent_dir / "agent_result.json").write_text(
                json.dumps(agent_run.to_dict(), indent=2),
                encoding="utf-8",
            )
            store.upload_directory(survey_id, output)
            store.update_survey_status(
                survey_id,
                "done",
                **({"created_at": created_at} if created_at else {}),
                frames=result.frames,
                keyframes=result.keyframes,
                detections=sum(result.detections.values()),
                instances=len(result.instances),
                runtime=result.metrics.get("runtime_s", 0.0),
            )
        except Exception as error:
            store.update_survey_status(
                survey_id,
                "failed",
                **({"created_at": created_at} if created_at else {}),
                error=str(error)[:500],
            )
            raise


def worker_loop(stop_event: threading.Event | None = None) -> None:
    queue_url = os.environ.get("MARG_SQS_QUEUE_URL")
    bucket = os.environ.get("MARG_S3_BUCKET")
    if not queue_url or not bucket:
        return
    store = S3SurveyStore(
        bucket,
        os.environ.get("MARG_S3_PREFIX", ""),
        os.environ.get("MARG_S3_CACHE_DIR", "/tmp/marg-cache"),
    )
    sqs = boto3.client("sqs")
    while stop_event is None or not stop_event.is_set():
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=900,
        )
        for message in response.get("Messages", []):
            receipt = str(message.get("ReceiptHandle", ""))
            try:
                body = json.loads(str(message["Body"]))
                if isinstance(body, dict):
                    process_job(body, store)
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            except (BotoCoreError, ClientError, KeyError, TypeError, ValueError):
                continue


def main() -> None:
    worker_loop()


if __name__ == "__main__":
    main()
