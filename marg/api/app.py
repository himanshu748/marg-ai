import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from botocore.exceptions import ClientError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from marg.agent.llm import default_llm
from marg.agent.loop import run_agent
from marg.agent.tools import ToolContext
from marg.store.aws import S3SurveyStore
from marg.store.local import LocalStore
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.models import SurveyResult


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = ""


_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="marg-agent")


def create_app(data_root: Path | str) -> FastAPI:
    bucket = os.environ.get("MARG_S3_BUCKET")
    root = Path(os.environ.get("MARG_S3_CACHE_DIR", str(data_root))).resolve()
    s3_store = (
        S3SurveyStore(
            bucket,
            os.environ.get("MARG_S3_PREFIX", ""),
            root,
        )
        if bucket
        else None
    )
    app = FastAPI(title="MargAI Survey Dashboard")

    def survey_dir(survey_id: str) -> Path:
        candidate = (root / survey_id).resolve()
        if s3_store is not None and not candidate.is_dir():
            try:
                s3_store.sync_survey(survey_id)
            except ClientError as error:
                if "NoSuchKey" in str(error) or "404" in str(error):
                    pass
        if candidate.parent != root or not candidate.is_dir():
            raise HTTPException(status_code=404, detail="Survey not found")
        return candidate

    def load_result(survey_id: str) -> tuple[Path, SurveyResult]:
        directory = survey_dir(survey_id)
        path = directory / "result.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Survey not found")
        return directory, SurveyResult.model_validate_json(path.read_text(encoding="utf-8"))

    def decisions_path(directory: Path) -> Path:
        return directory / "agent" / "decisions.json"

    def load_decisions(directory: Path) -> dict[str, dict[str, object]]:
        path = decisions_path(directory)
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def agent_payload(directory: Path) -> dict[str, object]:
        path = directory / "agent" / "agent_result.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Agent result not found")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise HTTPException(status_code=500, detail="Invalid agent result")
        decisions = load_decisions(directory)
        work_orders = value.get("work_orders", [])
        if isinstance(work_orders, list):
            for work_order in work_orders:
                if isinstance(work_order, dict):
                    decision = decisions.get(str(work_order.get("work_order_id")))
                    if decision:
                        work_order.update(decision)
        return value

    def safe_media(
        survey_id: str,
        category: Literal["keyframes", "crops", "evidence", "agent_crops"],
        name: str,
    ) -> Path:
        directory = survey_dir(survey_id)
        if Path(name).name != name:
            raise HTTPException(status_code=404, detail="Media not found")
        if s3_store is not None:
            try:
                s3_store.ensure_media(survey_id, category, name)
            except ClientError as error:
                if "NoSuchKey" in str(error) or "404" in str(error):
                    raise HTTPException(status_code=404, detail="Media not found") from error
        candidate = (directory / category / name).resolve()
        base = (directory / category).resolve()
        if candidate.parent != base or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Media not found")
        return candidate

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/surveys")
    def list_surveys() -> list[dict[str, object]]:
        surveys: list[dict[str, object]] = []
        if s3_store is not None:
            loaded_ids: set[str] = set()
            for survey_id in s3_store.list_survey_ids():
                try:
                    directory, result = load_result(survey_id)
                except HTTPException:
                    continue
                loaded_ids.add(survey_id)
                status = s3_store.survey_status(survey_id)
                surveys.append(
                    {
                        "id": directory.name,
                        "survey_id": result.survey_id,
                        "status": status.get("status", "done"),
                        "frames": result.frames,
                        "keyframes": result.keyframes,
                        "detections": result.detections,
                        "instances": len(result.instances),
                        "runtime": result.metrics.get("runtime_s", 0.0),
                        "geo_source": result.geo_source,
                    }
                )
            for status in s3_store.list_statuses():
                survey_id = str(status.get("survey_id", ""))
                if survey_id and survey_id not in loaded_ids:
                    surveys.append({"id": survey_id, "survey_id": survey_id, **status})
            return surveys
        if not root.is_dir():
            return surveys
        for directory in sorted(root.iterdir()):
            result_path = directory / "result.json"
            if not directory.is_dir() or not result_path.is_file():
                continue
            result = SurveyResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            surveys.append(
                {
                    "id": directory.name,
                    "survey_id": result.survey_id,
                    "status": "done",
                    "frames": result.frames,
                    "keyframes": result.keyframes,
                    "detections": result.detections,
                    "instances": len(result.instances),
                    "runtime": result.metrics.get("runtime_s", 0.0),
                    "geo_source": result.geo_source,
                }
            )
        return surveys

    @app.get("/api/surveys/{survey_id}")
    def get_survey(survey_id: str) -> dict[str, object]:
        _, result = load_result(survey_id)
        return result.model_dump(mode="json")

    @app.get("/api/surveys/{survey_id}/agent")
    def get_agent(survey_id: str) -> dict[str, object]:
        directory, _ = load_result(survey_id)
        return agent_payload(directory)

    @app.get("/api/surveys/{survey_id}/status")
    def get_status(survey_id: str) -> dict[str, object]:
        if s3_store is None:
            directory, result = load_result(survey_id)
            return {"survey_id": result.survey_id, "status": "done", "directory": str(directory)}
        status = s3_store.survey_status(survey_id)
        if not status:
            raise HTTPException(status_code=404, detail="Survey not found")
        return status

    @app.get("/api/surveys/{survey_id}/trace")
    def get_trace(survey_id: str) -> list[dict[str, object]]:
        directory, _ = load_result(survey_id)
        path = directory / "agent" / "trace.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Agent trace not found")
        entries: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    entries.append(value)
        return entries

    @app.get("/api/surveys/{survey_id}/keyframes/{name}")
    def get_keyframe(survey_id: str, name: str) -> FileResponse:
        return FileResponse(safe_media(survey_id, "keyframes", name))

    @app.get("/api/surveys/{survey_id}/crops/{name}")
    def get_crop(survey_id: str, name: str) -> FileResponse:
        return FileResponse(safe_media(survey_id, "crops", name))

    @app.get("/api/surveys/{survey_id}/evidence/{name}")
    def get_evidence(survey_id: str, name: str) -> FileResponse:
        return FileResponse(safe_media(survey_id, "evidence", name))

    @app.get("/api/surveys/{survey_id}/agent_crops/{name}")
    def get_agent_crop(survey_id: str, name: str) -> FileResponse:
        return FileResponse(safe_media(survey_id, "agent_crops", name))

    @app.post("/api/surveys/{survey_id}/work_orders/{work_order_id}/decision")
    def decide_work_order(
        survey_id: str, work_order_id: str, request: DecisionRequest
    ) -> dict[str, object]:
        directory, _ = load_result(survey_id)
        payload = agent_payload(directory)
        work_orders = payload.get("work_orders", [])
        if not isinstance(work_orders, list):
            raise HTTPException(status_code=404, detail="Work order not found")
        order = next(
            (
                item
                for item in work_orders
                if isinstance(item, dict) and str(item.get("work_order_id")) == work_order_id
            ),
            None,
        )
        if order is None:
            raise HTTPException(status_code=404, detail="Work order not found")
        decision = {
            "decision": request.decision,
            "decision_note": request.note,
            "status": "approved" if request.decision == "approve" else "rejected",
        }
        agent_dir = directory / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        decisions = load_decisions(directory)
        decisions[work_order_id] = decision
        decisions_path(directory).write_text(json.dumps(decisions, indent=2), encoding="utf-8")
        if s3_store is not None:
            s3_store.save_decision(survey_id, work_order_id, decision)
            s3_store.upload_file(
                decisions_path(directory),
                s3_store.object_key(survey_id, "agent/decisions.json"),
            )
        order.update(decision)
        return order

    @app.post("/api/surveys/upload", status_code=202)
    def upload_survey(
        video: UploadFile = File(...),  # noqa: B008
        gpx: UploadFile | None = File(default=None),  # noqa: B008
        name: str = Form(default=""),
    ) -> dict[str, str]:
        if s3_store is None:
            raise HTTPException(status_code=503, detail="S3 upload is not configured")
        survey_id = f"{_slug(name or video.filename or 'survey')}-{uuid4().hex[:10]}"
        video_key = f"uploads/{survey_id}/{Path(video.filename or 'video.bin').name}"
        s3_store.upload_fileobj(video.file, video_key, video.content_type)
        gpx_key = ""
        if gpx is not None and gpx.filename:
            gpx_key = f"uploads/{survey_id}/{Path(gpx.filename).name}"
            s3_store.upload_fileobj(gpx.file, gpx_key, gpx.content_type)
        s3_store.update_survey_status(
            survey_id,
            "queued",
            video_key=video_key,
            gpx_key=gpx_key,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        queue_url = os.environ.get("MARG_SQS_QUEUE_URL")
        if not queue_url:
            raise HTTPException(status_code=503, detail="SQS queue is not configured")
        import boto3

        boto3.client("sqs").send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({"survey_id": survey_id}),
        )
        return {"survey_id": survey_id, "status": "queued"}

    @app.post("/api/surveys/{survey_id}/run_agent")
    def run_agent_endpoint(
        survey_id: str, llm: Literal["mock", "bedrock"] = "mock"
    ) -> dict[str, str]:
        directory, _ = load_result(survey_id)
        _EXECUTOR.submit(_run_agent, directory, llm, s3_store)
        return {"status": "started", "survey_id": survey_id, "llm": llm}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app


def _run_agent(
    directory: Path,
    llm_kind: str,
    s3_store: S3SurveyStore | None = None,
) -> None:
    result = SurveyResult.model_validate_json(
        (directory / "result.json").read_text(encoding="utf-8")
    )
    agent_dir = directory / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    config = VisionConfig()
    detector = DNNDetector(config.model_file(Path(__file__).resolve().parents[2]), config)
    context = ToolContext(
        result=result,
        store=s3_store or LocalStore(agent_dir / "store"),
        detector=detector,
        keyframe_dir=directory / "keyframes",
    )
    run = run_agent(result, default_llm(llm_kind), context)
    run.trace.write_jsonl(agent_dir / "trace.jsonl")
    (agent_dir / "agent_result.json").write_text(
        json.dumps(run.to_dict(), indent=2), encoding="utf-8"
    )
    if s3_store is not None:
        s3_store.upload_directory(result.survey_id, directory / "agent")


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return normalized.strip("-")[:40] or "survey"


app = create_app(Path("outputs"))
