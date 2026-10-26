"""Cookie-isolated, bounded live reviews of the public judge fixtures."""

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from marg.agent.modal_llm import MODEL, Allowance

COOKIE = "marg_live_session"
SURVEYS = {"pothole_kumasi", "pothole_cars"}


class StartReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    survey_id: Literal["pothole_kumasi", "pothole_cars"]


class LiveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=2000)


async def small_json(request: Request, model):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 16_384:
            raise HTTPException(413, "Live review request is too large")
    try:
        return model.model_validate_json(body)
    except ValidationError:
        raise HTTPException(422, "Invalid live review request") from None


class LiveReviews:
    def __init__(self, seeds: Path, root: Path, runner):
        self.seeds, self.root, self.runner = seeds.resolve(), root.resolve(), runner
        if self.root.is_relative_to(self.seeds) or self.seeds.is_relative_to(self.root):
            raise ValueError("Live review data must be separate from recorded fixtures")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "reviews.sqlite3"
        self.allowance = Allowance(Path(os.environ["MARG_MODAL_ALLOWANCE_DB"]))
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="marg-live"
        )
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, owner TEXT, payload TEXT)"
            )
            for job_id, raw in db.execute("SELECT id,payload FROM reviews").fetchall():
                job = json.loads(raw)
                if job["status"] in {"queued", "running"}:
                    job.update(
                        status="failed",
                        error="Review interrupted by a restart. It was not retried.",
                    )
                    db.execute(
                        "UPDATE reviews SET payload=? WHERE id=?",
                        (json.dumps(job), job_id),
                    )

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def status(self):
        allowance = self.allowance.status()
        return {
            "enabled": allowance["available"]
            and os.environ.get("MARG_MODAL_ENABLED") == "1"
            and bool(os.environ.get("MARG_MODAL_PROXY_TOKEN")),
            "model_id": MODEL,
            "provider": "Modal",
            "allowed_surveys": sorted(SURVEYS),
            "allowance": allowance,
            "isolated": True,
            "uploads_enabled": False,
        }

    @staticmethod
    def owner(request: Request) -> str:
        secret = request.cookies.get(COOKIE, "")
        if not secret or len(secret) > 128:
            raise HTTPException(404, "Live review not found in this browser session")
        return hashlib.sha256(secret.encode()).hexdigest()

    def get(self, job_id: str, owner: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM reviews WHERE id=? AND owner=?", (job_id, owner)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Live review not found in this browser session")
        return json.loads(row[0])

    def update(self, job_id: str, **changes):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(
                db.execute(
                    "SELECT payload FROM reviews WHERE id=?", (job_id,)
                ).fetchone()[0]
            )
            job.update(changes)
            db.execute(
                "UPDATE reviews SET payload=? WHERE id=?", (json.dumps(job), job_id)
            )

    def execute(self, job_id: str, survey: str):
        self.update(job_id, status="running")
        directory = self.root / "jobs" / job_id / survey
        try:
            shutil.copytree(
                self.seeds / survey,
                directory,
                ignore=shutil.ignore_patterns("agent", "agent_crops", ".review.lock"),
            )
            result = self.runner(directory, "modal")
            completed = result.get("status") == "completed"
            self.update(
                job_id,
                status="succeeded" if completed else "failed",
                result=result,
                error=None
                if completed
                else result.get("error", {}).get(
                    "message", "The live review could not complete."
                ),
            )
        except Exception:  # noqa: BLE001 - sanitize unexpected errors at the background job boundary
            # Provider errors are sanitized by the runner; infrastructure exceptions never expose paths or credentials.
            self.update(
                job_id,
                status="failed",
                error="The live review could not complete. Recorded evidence is unchanged.",
            )

    def install(self, app: FastAPI):
        @app.get("/api/live-reviews")
        def capabilities():
            return self.status()

        @app.post("/api/live-reviews", status_code=202)
        async def start(request: Request, response: Response):
            data = await small_json(request, StartReview)
            if not self.status()["enabled"]:
                raise HTTPException(
                    429, "The live review allowance is unavailable or expired"
                )
            if not (self.seeds / data.survey_id / "result.json").is_file():
                raise HTTPException(404, "Judge fixture is unavailable")
            secret = request.cookies.get(COOKIE) or secrets.token_urlsafe(32)
            if len(secret) > 128:
                raise HTTPException(400, "Invalid browser session")
            owner = hashlib.sha256(secret.encode()).hexdigest()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                rows = db.execute("SELECT owner,payload FROM reviews").fetchall()
                if len(rows) >= 20 or sum(1 for who, _ in rows if who == owner) >= 2:
                    raise HTTPException(
                        429, "This demo has reached its live review run limit"
                    )
                if any(
                    json.loads(raw)["status"] in {"queued", "running"}
                    for _, raw in rows
                ):
                    raise HTTPException(
                        429,
                        "Another live review is running. Please retry when it finishes.",
                    )
                job_id = uuid4().hex
                job = {
                    "job_id": job_id,
                    "kind": "live",
                    "survey_id": data.survey_id,
                    "status": "queued",
                    "status_url": f"/api/live-reviews/{job_id}",
                    "decisions": {},
                }
                db.execute(
                    "INSERT INTO reviews VALUES (?,?,?)",
                    (job_id, owner, json.dumps(job)),
                )
            response.set_cookie(
                COOKIE,
                secret,
                httponly=True,
                secure=os.environ.get("MARG_COOKIE_SECURE") == "1",
                samesite="strict",
                max_age=86400,
                path="/api/live-reviews",
            )
            self.executor.submit(self.execute, job_id, data.survey_id)
            return job

        @app.get("/api/live-reviews/{job_id}")
        def get(job_id: str, request: Request):
            return self.get(job_id, self.owner(request))

        @app.get("/api/live-reviews/{job_id}/media/{category}/{name}")
        def media(job_id: str, category: str, name: str, request: Request):
            job = self.get(job_id, self.owner(request))
            if category != "agent_crops" or Path(name).name != name:
                raise HTTPException(404, "Live review image not found")
            base = (self.root / "jobs" / job_id / job["survey_id"] / category).resolve()
            candidate = (base / name).resolve()
            if (
                candidate.parent != base
                or not candidate.is_file()
                or candidate.suffix.lower() not in {".jpg", ".png"}
            ):
                raise HTTPException(404, "Live review image not found")
            return FileResponse(candidate)

        @app.post("/api/live-reviews/{job_id}/decisions/{order_id}")
        async def decide(job_id: str, order_id: str, request: Request):
            data = await small_json(request, LiveDecision)
            owner = self.owner(request)
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT payload FROM reviews WHERE id=? AND owner=?",
                    (job_id, owner),
                ).fetchone()
                if row is None:
                    raise HTTPException(
                        404, "Live review not found in this browser session"
                    )
                job = json.loads(row[0])
                result = job.get("result", {})
                order = next(
                    (
                        o
                        for o in result.get("work_orders", [])
                        if o.get("work_order_id") == order_id
                    ),
                    None,
                )
                if (
                    job["status"] != "succeeded"
                    or result.get("audit", {}).get("passed") is not True
                ):
                    raise HTTPException(
                        409, "A completed review with passing policy checks is required"
                    )
                if order is None:
                    raise HTTPException(404, "Live work order not found")
                if order_id in job["decisions"]:
                    raise HTTPException(409, "A decision has already been recorded")
                job["decisions"][order_id] = {
                    "decision": data.decision,
                    "note": data.note,
                    "dispatch": False,
                    "agent_run_id": result.get("run_id"),
                }
                db.execute(
                    "UPDATE reviews SET payload=? WHERE id=?", (json.dumps(job), job_id)
                )
            return job
