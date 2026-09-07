import fcntl
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from marg.agent.llm import default_llm, provider_status
from marg.agent.loop import run_agent
from marg.agent.tools import ToolContext
from marg.api.runtime import JobConflict, JobFailure, JobRuntime, QueueFull, now
from marg.api.security import install_auth
from marg.store.aws import S3SurveyStore
from marg.store.local import LocalStore, atomic_write_json, validate_survey_id
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.geo import GeoTrack
from marg.vision.models import SurveyResult
from marg.vision.pipeline import run as run_pipeline
from marg.vision.privacy import privacy_status

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=2000)


def create_app(data_root: Path | str) -> FastAPI:
    bucket = os.environ.get("MARG_S3_BUCKET")
    root = Path(
        os.environ.get("MARG_S3_CACHE_DIR", str(data_root)) if bucket else data_root
    ).resolve()
    read_only = os.environ.get("MARG_READ_ONLY") == "1"
    token = os.environ.get("MARG_API_TOKEN") or None
    bedrock_enabled = os.environ.get("MARG_BEDROCK_ENABLED") == "1"
    queue_url = os.environ.get("MARG_SQS_QUEUE_URL")
    max_video_bytes = _positive_env("MARG_MAX_VIDEO_BYTES", 256 * 1024 * 1024)
    max_gpx_bytes = _positive_env("MARG_MAX_GPX_BYTES", 5 * 1024 * 1024)
    runtime = JobRuntime(
        root,
        workers=_positive_env("MARG_JOB_WORKERS", 2),
        capacity=_positive_env("MARG_JOB_CAPACITY", 8),
    )
    s3_store = (
        S3SurveyStore(bucket, os.environ.get("MARG_S3_PREFIX", ""), root)
        if bucket
        else None
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime.recover()
        yield
        runtime.shutdown()

    app = FastAPI(title="MargAI Survey Workbench", lifespan=lifespan)
    app.state.jobs = runtime
    install_auth(app, token, os.environ.get("MARG_COOKIE_SECURE") == "1")

    @app.middleware("http")
    async def upload_length_guard(request: Request, call_next):
        if request.url.path == "/api/surveys/upload":
            length = request.headers.get("content-length", "")
            if (
                length.isdigit()
                and int(length) > max_video_bytes + max_gpx_bytes + 1024 * 1024
            ):
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Upload exceeds the configured size limit"},
                )
        return await call_next(request)

    @app.exception_handler(BotoCoreError)
    @app.exception_handler(ClientError)
    async def aws_error(request: Request, error: Exception):
        if (
            isinstance(error, ClientError)
            and error.response.get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        ):
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "This work order has changed. Refresh the review before deciding."
                },
            )
        LOGGER.error("AWS operation failed for %s", request.url.path, exc_info=error)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "AWS storage is temporarily unavailable. Retry shortly or check the server configuration."
            },
        )

    def writable(action: str = "Changes") -> None:
        if read_only:
            raise HTTPException(
                status_code=403,
                detail=f"{action} are disabled on the public demo; run locally or on AWS",
            )

    def validate_id(survey_id: str) -> None:
        try:
            validate_survey_id(survey_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Survey not found") from error

    def survey_dir(survey_id: str, refresh_result: bool = False) -> Path:
        validate_id(survey_id)
        candidate = (root / survey_id).resolve()
        if candidate.parent != root:
            raise HTTPException(status_code=404, detail="Survey not found")
        if s3_store is not None and not (candidate / "result.json").is_file():
            try:
                s3_store.sync_survey(survey_id)
            except ClientError as error:
                if str(error.response.get("Error", {}).get("Code")) in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise HTTPException(
                        status_code=404, detail="Survey not found"
                    ) from error
                raise
        elif s3_store is not None and refresh_result:
            active = runtime.latest(survey_id)
            if (
                not active
                or active["kind"] != "agent"
                or active["status"] not in {"queued", "running"}
            ):
                s3_store.download_file(
                    s3_store.object_key(survey_id, "result.json"),
                    candidate / "result.json",
                )
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail="Survey not found")
        return candidate

    def json_file(path: Path, description: str) -> dict[str, object]:
        if not path.is_file() or path.resolve().parent != path.parent.resolve():
            raise HTTPException(status_code=404, detail=f"{description} not found")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("Expected object")
            return value
        except (TypeError, ValueError, OSError) as error:
            LOGGER.warning("Unreadable %s: %s", description, path)
            raise HTTPException(
                status_code=500,
                detail=f"Saved {description.lower()} is unreadable; restore or reprocess the survey",
            ) from error

    def load_result(survey_id: str) -> tuple[Path, SurveyResult]:
        directory = survey_dir(survey_id, refresh_result=True)
        try:
            return directory, SurveyResult.model_validate(
                json_file(directory / "result.json", "Survey")
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=500,
                detail="Saved survey data is invalid; restore or reprocess the survey",
            ) from error

    def decisions_path(directory: Path) -> Path:
        agent_dir = directory / "agent"
        if agent_dir.resolve().parent != directory:
            raise HTTPException(status_code=404, detail="Agent result not found")
        return agent_dir / "decisions.json"

    def load_decisions(directory: Path) -> dict[str, object]:
        path = decisions_path(directory)
        return json_file(path, "Review decisions") if path.is_file() else {}

    def refresh_agent_metadata(directory: Path) -> None:
        if s3_store is None:
            return
        active = runtime.latest(directory.name)
        if (
            active
            and active["kind"] == "agent"
            and active["status"] in {"queued", "running"}
        ):
            return
        agent_dir = decisions_path(directory).parent
        for name in ("agent_result.json", "decisions.json"):
            target = agent_dir / name
            try:
                s3_store.download_file(
                    s3_store.object_key(directory.name, f"agent/{name}"), target
                )
            except ClientError as error:
                if str(error.response.get("Error", {}).get("Code")) not in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise
                target.unlink(missing_ok=True)

    def agent_payload(directory: Path) -> dict[str, object]:
        refresh_agent_metadata(directory)
        value = json_file(
            decisions_path(directory).parent / "agent_result.json", "Agent result"
        )
        decisions = load_decisions(directory)
        for order in (
            value.get("work_orders", [])
            if isinstance(value.get("work_orders"), list)
            else []
        ):
            if isinstance(order, dict):
                decision = decisions.get(str(order.get("work_order_id")))
                if isinstance(decision, dict) and decision.get(
                    "agent_run_id"
                ) == value.get("run_id"):
                    order.update(decision)
        return value

    def safe_media(survey_id: str, category: str, name: str) -> Path:
        directory = survey_dir(survey_id)
        base = (directory / category).resolve()
        candidate = (base / name).resolve()
        if (
            Path(name).name != name
            or base.parent != directory
            or candidate.parent != base
        ):
            raise HTTPException(status_code=404, detail="Media not found")
        if s3_store:
            try:
                s3_store.ensure_media(survey_id, category, name)
            except ClientError as error:
                if str(error.response.get("Error", {}).get("Code")) in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise HTTPException(
                        status_code=404, detail="Media not found"
                    ) from error
                raise
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Media not found")
        return candidate

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, object]:
        return {
            "read_only": read_only,
            "storage": "aws" if s3_store else "local",
            "authentication_required": bool(token),
            "uploads": {
                "enabled": not read_only and (s3_store is None or bool(queue_url)),
                "max_video_bytes": max_video_bytes,
                "max_gpx_bytes": max_gpx_bytes,
                "accepted_video_extensions": sorted(VIDEO_EXTENSIONS),
                "gpx_optional": True,
            },
            "agent": {
                "default_mode": "mock",
                "modes": ["mock", "bedrock"] if bedrock_enabled else ["mock"],
                "enabled": not read_only,
                "bedrock_enabled": bedrock_enabled,
                "label": "Deterministic review",
                "provider": provider_status("mock"),
                "bedrock": provider_status("bedrock"),
            },
            "processing": {
                "workers": runtime.workers,
                "capacity": runtime.capacity,
                "durable_jobs": True,
            },
            "privacy": privacy_status(),
        }

    @app.get("/healthz")
    @app.get("/api/health")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "read_only": read_only,
            "storage": "aws" if s3_store else "local",
            "bedrock_enabled": bedrock_enabled,
        }

    def refresh_job(job: dict[str, object] | None) -> dict[str, object] | None:
        if (
            job
            and job.get("external")
            and s3_store
            and job["status"] in {"queued", "running"}
        ):
            status = s3_store.survey_status(str(job["survey_id"]))
            normalized = {
                "queued": "queued",
                "processing": "running",
                "done": "succeeded",
                "failed": "failed",
            }.get(str(status.get("status")))
            if normalized:
                fields = {
                    key: status[key]
                    for key in ("agent_status", "vision_status", "error")
                    if key in status
                }
                if normalized == "running" and not job.get("started_at"):
                    fields["started_at"] = status.get("started_at", now())
                if normalized in {"succeeded", "failed"}:
                    fields["finished_at"] = status.get("finished_at", now())
                job = runtime.update(str(job["job_id"]), status=normalized, **fields)
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        job = refresh_job(runtime.get(job_id))
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/surveys")
    def list_surveys() -> list[dict[str, object]]:
        surveys: dict[str, dict[str, object]] = {}
        if s3_store:
            ids, statuses = s3_store.list_survey_ids(), s3_store.list_statuses()
        else:
            ids = (
                [
                    path.name
                    for path in sorted(root.iterdir())
                    if path.is_dir()
                    and not path.name.startswith(".")
                    and (path / "result.json").is_file()
                ]
                if root.is_dir()
                else []
            )
            statuses = []
        for survey_id in ids:
            try:
                directory, result = load_result(survey_id)
            except HTTPException as error:
                if error.status_code == 500:
                    surveys[survey_id] = {
                        "id": survey_id,
                        "survey_id": survey_id,
                        "status": "failed",
                        "error": error.detail,
                    }
                continue
            job = refresh_job(runtime.latest(survey_id, "survey"))
            modified = (directory / "result.json").stat().st_mtime
            surveys[survey_id] = {
                "id": directory.name,
                "survey_id": result.survey_id,
                "status": "done",
                "frames": result.frames,
                "keyframes": result.keyframes,
                "detections": result.detections,
                "instances": len(result.instances),
                "runtime": result.metrics.get("runtime_s", 0.0),
                "geo_source": result.geo_source,
                "created_at": job["created_at"]
                if job
                else datetime.fromtimestamp(modified, timezone.utc).isoformat(),
                "agent_job": runtime.latest(survey_id, "agent"),
            }
            if job:
                surveys[survey_id]["job"] = job
                if job["status"] != "succeeded":
                    surveys[survey_id]["status"] = job["status"]
            if s3_store or (directory / "agent" / "agent_result.json").is_file():
                try:
                    agent = agent_payload(directory)
                    surveys[survey_id]["agent_status"] = agent.get(
                        "status", "completed"
                    )
                    surveys[survey_id]["agent_provider"] = agent.get("provider")
                except HTTPException as error:
                    surveys[survey_id]["agent_status"] = (
                        "pending" if error.status_code == 404 else "failed"
                    )
        for status in statuses:
            survey_id = str(status.get("survey_id", ""))
            if survey_id:
                if survey_id in surveys:
                    for key in (
                        "status",
                        "created_at",
                        "agent_status",
                        "agent_provider",
                        "error",
                        "vision_status",
                    ):
                        if key in status:
                            surveys[survey_id][key] = status[key]
                else:
                    surveys[survey_id] = {"id": survey_id, **status}
        for saved_job in runtime.surveys():
            job = refresh_job(saved_job) or saved_job
            survey_id = str(job["survey_id"])
            surveys.setdefault(
                survey_id,
                {
                    "id": survey_id,
                    "survey_id": survey_id,
                    "status": job["status"],
                    "created_at": job["created_at"],
                    "error": job.get("error"),
                    "job": job,
                },
            )
        return sorted(
            surveys.values(),
            key=lambda value: str(value.get("created_at", "")),
            reverse=True,
        )

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
        validate_id(survey_id)
        job = refresh_job(runtime.latest(survey_id))
        if job:
            return {
                **job,
                "survey_job": refresh_job(runtime.latest(survey_id, "survey")),
                "agent_job": runtime.latest(survey_id, "agent"),
            }
        if s3_store:
            status = s3_store.survey_status(survey_id)
            if status:
                return status
        _, result = load_result(survey_id)
        return {
            "survey_id": result.survey_id,
            "status": "done",
            "agent_job": None,
            "survey_job": None,
        }

    @app.get("/api/surveys/{survey_id}/trace")
    def get_trace(survey_id: str) -> list[dict[str, object]]:
        directory, _ = load_result(survey_id)
        payload = agent_payload(directory)
        trace = payload.get("trace")
        if isinstance(trace, list):
            return [item for item in trace if isinstance(item, dict)]
        path = decisions_path(directory).parent / "trace.jsonl"
        if not path.is_file() or path.resolve().parent != path.parent.resolve():
            raise HTTPException(status_code=404, detail="Agent trace not found")
        try:
            return [
                value
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and isinstance(value := json.loads(line), dict)
            ]
        except ValueError as error:
            raise HTTPException(
                status_code=500, detail="Saved agent trace is unreadable"
            ) from error

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
        writable("Decisions")
        directory, _ = load_result(survey_id)
        with _survey_lock(directory):
            active = runtime.latest(survey_id)
            if active and active["status"] in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail="Wait for the active review to finish before deciding",
                )
            if s3_store and s3_store.survey_status(survey_id).get("status") in {
                "queued",
                "processing",
            }:
                raise HTTPException(
                    status_code=409,
                    detail="Wait for cloud processing to finish before deciding",
                )
            payload = agent_payload(directory)
            audit = payload.get("audit")
            if payload.get("status", "completed") != "completed" or (
                isinstance(audit, dict) and not audit.get("passed")
            ):
                raise HTTPException(
                    status_code=409,
                    detail="This review did not complete successfully. Run a successful review before deciding.",
                )
            orders = payload.get("work_orders", [])
            order = (
                next(
                    (
                        item
                        for item in orders
                        if isinstance(item, dict)
                        and str(item.get("work_order_id")) == work_order_id
                    ),
                    None,
                )
                if isinstance(orders, list)
                else None
            )
            if order is None:
                raise HTTPException(status_code=404, detail="Work order not found")
            if order.get("decision") == request.decision:
                return order
            if order.get("status") != "pending_approval":
                raise HTTPException(
                    status_code=409,
                    detail="Only work orders awaiting approval can receive a decision",
                )
            decision = {
                "decision": request.decision,
                "decision_note": request.note.strip(),
                "status": "approved" if request.decision == "approve" else "rejected",
                "decided_at": now(),
                "agent_run_id": payload.get("run_id"),
            }
            decisions = load_decisions(directory)
            if s3_store:
                if not payload.get("run_id"):
                    raise HTTPException(
                        status_code=409,
                        detail="Run a fresh review before deciding on this legacy cloud work order.",
                    )
                decision = s3_store.save_decision(survey_id, work_order_id, decision)
            decisions[work_order_id] = decision
            if s3_store:
                s3_store.save_json(survey_id, "agent/decisions.json", decisions)
            atomic_write_json(decisions_path(directory), decisions)
            order.update(decision)
            return order

    def create_job(survey_id: str, kind: str, **values: object) -> dict[str, object]:
        refresh_job(runtime.latest(survey_id))
        try:
            return runtime.create(survey_id, kind, **values)
        except JobConflict as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(error),
                    "job_id": error.job["job_id"],
                    "status_url": error.job["status_url"],
                },
            ) from error
        except QueueFull as error:
            raise HTTPException(
                status_code=429, detail=str(error), headers={"Retry-After": "5"}
            ) from error

    @app.post("/api/surveys/upload", status_code=202)
    def upload_survey(
        video: Annotated[UploadFile, File()],
        gpx: Annotated[UploadFile | None, File()] = None,
        name: Annotated[str, Form(max_length=100)] = "",
    ) -> dict[str, object]:
        writable("Uploads")
        if s3_store and not queue_url:
            raise HTTPException(
                status_code=503, detail="The AWS processing queue is not configured"
            )
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=415, detail="Choose an MP4, MOV, AVI, MKV or WebM video"
            )
        if gpx and gpx.filename and Path(gpx.filename).suffix.lower() != ".gpx":
            raise HTTPException(
                status_code=415, detail="The optional location file must be GPX"
            )
        survey_id = (
            f"{_slug(name or Path(video.filename or 'survey').stem)}-{uuid4().hex[:10]}"
        )
        staging = root / ".marg" / "uploads" / survey_id
        staging.mkdir(parents=True, exist_ok=False)
        video_path = staging / f"video{suffix}"
        gpx_path = staging / "track.gpx" if gpx and gpx.filename else None
        accepted = False
        job: dict[str, object] | None = None
        try:
            _copy_upload(video, video_path, max_video_bytes, "Video")
            _validate_video(video_path)
            if gpx_path and gpx:
                _copy_upload(gpx, gpx_path, max_gpx_bytes, "GPX")
                _validate_gpx(gpx_path)
            job = create_job(
                survey_id,
                "survey",
                name=name.strip() or Path(video.filename or "Survey").stem,
                llm="mock",
                external=bool(s3_store),
                geo_source="gpx" if gpx_path else "synthetic",
            )
            if s3_store:
                video_key = s3_store.object_key(
                    "uploads", f"{survey_id}/{video_path.name}"
                )
                gpx_key = (
                    s3_store.object_key("uploads", f"{survey_id}/track.gpx")
                    if gpx_path
                    else ""
                )
                s3_store.upload_file(video_path, video_key)
                if gpx_path:
                    s3_store.upload_file(gpx_path, gpx_key)
                s3_store.update_survey_status(
                    survey_id,
                    "queued",
                    video_key=video_key,
                    gpx_key=gpx_key,
                    created_at=job["created_at"],
                    job_id=job["job_id"],
                )
                import boto3

                boto3.client("sqs").send_message(
                    QueueUrl=queue_url, MessageBody=json.dumps({"survey_id": survey_id})
                )
                shutil.rmtree(staging)
            else:
                runtime.submit(
                    job,
                    lambda: _process_local_survey(
                        root / survey_id, video_path, gpx_path
                    ),
                )
            accepted = True
            return job
        except HTTPException:
            raise
        except Exception:
            if job:
                runtime.update(
                    str(job["job_id"]),
                    status="failed",
                    error="The survey could not be queued. Retry the upload.",
                    finished_at=now(),
                )
            raise
        finally:
            video.file.close()
            if gpx:
                gpx.file.close()
            if not accepted:
                shutil.rmtree(staging, ignore_errors=True)

    @app.post("/api/surveys/{survey_id}/run_agent")
    def run_agent_endpoint(
        survey_id: str, llm: Literal["mock", "bedrock"] = "mock"
    ) -> dict[str, object]:
        writable("Agent runs")
        if llm == "bedrock" and not bedrock_enabled:
            raise HTTPException(
                status_code=403,
                detail="Bedrock is disabled until access is verified. Deterministic review is available now.",
            )
        directory, _ = load_result(survey_id)
        with _survey_lock(directory):
            job = create_job(survey_id, "agent", llm=llm)
            runtime.submit(job, lambda: _agent_job(directory, llm, s3_store))
        return job

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (static / "index.html").read_text(encoding="utf-8")

    return app


@contextmanager
def _survey_lock(directory: Path):
    with (directory / ".review.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run_agent(
    directory: Path, llm_kind: str, s3_store: S3SurveyStore | None = None
) -> dict[str, object]:
    result = SurveyResult.model_validate_json(
        (directory / "result.json").read_text(encoding="utf-8")
    )
    if s3_store:
        _hydrate_review_media(result, directory, s3_store)
    agent_dir = directory / "agent"
    if agent_dir.resolve().parent != directory.resolve():
        raise JobFailure("The survey agent directory is invalid")
    agent_dir.mkdir(parents=True, exist_ok=True)
    config = VisionConfig()
    detector = DNNDetector(
        config.model_file(Path(__file__).resolve().parents[2]), config
    )
    context = ToolContext(
        result=result,
        store=LocalStore(agent_dir / "store"),
        detector=detector,
        keyframe_dir=directory / "keyframes",
    )
    run = run_agent(result, default_llm(llm_kind), context)
    payload = run.to_dict()
    payload["llm"] = llm_kind
    run.trace.write_jsonl(agent_dir / "trace.jsonl")
    if s3_store:
        for order in run.work_orders:
            s3_store.save_work_order(
                result.survey_id,
                str(order["work_order_id"]),
                {**order, "agent_run_id": run.run_id},
            )
        for path in agent_dir.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.name not in {"agent_result.json", "decisions.json"}
            ):
                s3_store.upload_file(
                    path,
                    s3_store.object_key(
                        result.survey_id,
                        f"agent/{path.relative_to(agent_dir).as_posix()}",
                    ),
                )
        crop_dir = directory / "agent_crops"
        if crop_dir.is_dir() and crop_dir.resolve().parent == directory.resolve():
            for path in crop_dir.iterdir():
                if path.is_file() and not path.is_symlink():
                    s3_store.upload_file(
                        path,
                        s3_store.object_key(
                            result.survey_id, f"agent_crops/{path.name}"
                        ),
                    )
        s3_store.save_json(result.survey_id, "agent/agent_result.json", payload)
    atomic_write_json(agent_dir / "agent_result.json", payload)
    return payload


def _hydrate_review_media(
    result: SurveyResult, directory: Path, store: S3SurveyStore
) -> None:
    """Fetch inputs before review so browser cache visits cannot affect policy."""
    import cv2

    required = set(result.keyframe_paths)
    required.update(
        instance.evidence_path
        for instance in result.instances
        if instance.evidence_path
    )
    for instance in result.instances:
        if not instance.evidence_path:
            required.update(
                f"keyframes/kf_{keyframe_id:05d}.jpg"
                for keyframe_id in instance.keyframe_ids
            )
            if not instance.keyframe_ids and not result.keyframe_paths:
                raise JobFailure(
                    "A damage instance has no recorded inspection image. Reprocess the survey before review."
                )
    for relative in sorted(required):
        path = Path(relative)
        if len(path.parts) != 2 or path.parts[0] not in {"keyframes", "evidence"}:
            raise JobFailure(
                "A survey image path is invalid. Restore or reprocess this survey."
            )
        destination = directory / path
        if (
            destination.resolve().parent != (directory / path.parts[0]).resolve()
            or (directory / path.parts[0]).resolve().parent != directory.resolve()
        ):
            raise JobFailure("A survey image directory is invalid.")
        try:
            store.download_file(
                store.object_key(result.survey_id, relative), destination
            )
        except ClientError as error:
            if str(error.response.get("Error", {}).get("Code")) in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                raise JobFailure(
                    "A required survey image is missing from storage. Restore or reprocess the survey before review."
                ) from error
            raise
        if cv2.imread(str(destination)) is None:
            raise JobFailure(
                "A required survey image cannot be decoded. Restore or reprocess the survey before review."
            )
        if path.parts[0] == "evidence":
            raw_dir = directory / "evidence_raw"
            raw_path = raw_dir / path.name
            if (
                raw_dir.resolve().parent != directory.resolve()
                or raw_path.resolve().parent != raw_dir.resolve()
            ):
                raise JobFailure("The private evidence directory is invalid.")
            try:
                store.download_file(
                    store.object_key(result.survey_id, f"evidence_raw/{path.name}"),
                    raw_path,
                )
            except ClientError as error:
                if str(error.response.get("Error", {}).get("Code")) not in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise
                raw_path.unlink(missing_ok=True)
            else:
                if cv2.imread(str(raw_path)) is None:
                    raise JobFailure(
                        "A private inspection image cannot be decoded. Reprocess the survey before review."
                    )


def _agent_job(
    directory: Path, llm_kind: str, s3_store: S3SurveyStore | None = None
) -> dict[str, object]:
    payload = _run_agent(directory, llm_kind, s3_store)
    if payload.get("status", "completed") != "completed":
        error = payload.get("error")
        message = (
            str(error.get("message", "The review could not complete"))
            if isinstance(error, dict)
            else "The review could not complete"
        )
        raise JobFailure(message)
    return {
        "agent_status": payload.get("status", "completed"),
        "run_id": payload.get("run_id"),
        "provider": payload.get("provider"),
    }


def _process_local_survey(
    directory: Path, video: Path, gpx: Path | None
) -> dict[str, object]:
    try:
        config = VisionConfig()
        if not config.model_file(Path(__file__).resolve().parents[2]).is_file():
            raise JobFailure(
                "The road-damage detector model is missing. Install the model before retrying."
            )
        result = run_pipeline(video, gpx, directory, config)
        if not result.frames:
            (directory / "result.json").unlink(missing_ok=True)
            raise JobFailure(
                "No video frames could be decoded. Try a different video encoding."
            )
        result.survey_id, result.video, result.gpx = (
            directory.name,
            video.name,
            gpx.name if gpx else None,
        )
        atomic_write_json(directory / "result.json", result.model_dump(mode="json"))
        try:
            agent = _run_agent(directory, "mock")
            return {
                "vision_status": "completed",
                "agent_status": agent.get("status", "completed"),
                "agent_error": agent.get("error"),
                "result_url": f"/api/surveys/{directory.name}",
            }
        except Exception:
            LOGGER.exception(
                "Review failed after vision completed for %s", directory.name
            )
            return {
                "vision_status": "completed",
                "agent_status": "failed",
                "agent_error": "Vision completed, but review failed. Open the survey and retry deterministic review.",
                "result_url": f"/api/surveys/{directory.name}",
            }
    finally:
        shutil.rmtree(video.parent, ignore_errors=True)


def _copy_upload(upload: UploadFile, destination: Path, limit: int, label: str) -> None:
    size = 0
    with destination.open("xb") as handle:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"{label} exceeds the {limit // (1024 * 1024)} MB limit",
                )
            handle.write(chunk)
    if not size:
        raise HTTPException(status_code=422, detail=f"{label} is empty")


def _validate_video(path: Path) -> None:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not capture.isOpened() or not ok or frame is None:
            raise HTTPException(
                status_code=422,
                detail="The file is not a readable video. Export an MP4 with H.264 encoding and retry.",
            )
    finally:
        capture.release()


def _validate_gpx(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ValueError("Document declarations are not supported")
        track = GeoTrack.from_path(path, VisionConfig())
        if not track.points or any(
            not (-90 <= point.lat <= 90 and -180 <= point.lon <= 180)
            for point in track.points
        ):
            raise ValueError("Timestamped valid coordinates are required")
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail="GPX must contain timestamped track points with valid latitude and longitude",
        ) from error


def _positive_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _slug(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isascii() and character.isalnum() else "-"
        for character in value
    )
    return normalized.strip("-")[:40] or "survey"


app = create_app(Path(os.environ.get("MARG_DATA_ROOT", "outputs")))
