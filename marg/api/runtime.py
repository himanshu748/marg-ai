"""Bounded background execution with a durable, queryable job ledger.

The dashboard owns one runtime per process. Jobs interrupted by a process exit are
reported as failed on the next start; they are never silently replayed (a replay
could repeat model charges or human-facing side effects).
"""

import json
import logging
import os
import shutil
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

LOGGER = logging.getLogger(__name__)
ACTIVE = ("queued", "running")
_LIVE_OWNERS: WeakValueDictionary = WeakValueDictionary()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobConflict(RuntimeError):
    def __init__(self, job: dict[str, object]) -> None:
        self.job = job
        super().__init__("A job is already running for this survey")


class QueueFull(RuntimeError):
    pass


class JobFailure(RuntimeError):
    """A safe, user-visible background failure."""


class JobRuntime:
    def __init__(self, root: Path, workers: int = 2, capacity: int = 8) -> None:
        self.root = root
        self.path = root / ".marg" / "jobs.sqlite3"
        self.workers = workers
        self.capacity = capacity
        self._guard = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        self.owner = f"{os.getpid()}-{uuid4().hex}"
        _LIVE_OWNERS[self.owner] = self

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "id TEXT PRIMARY KEY, survey_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "status TEXT NOT NULL, owner TEXT NOT NULL, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _discard_upload(self, job: dict[str, object]) -> None:
        if job.get("kind") != "survey" or job.get("external"):
            return
        base = (self.root / ".marg" / "uploads").resolve()
        path = (base / str(job["survey_id"])).resolve()
        if path.parent == base:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, object] | None:
        return json.loads(row["payload"]) if row else None

    def recover(self) -> None:
        if not self.path.is_file():
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchall()
            for row in rows:
                job = self._decode(row)
                assert job is not None
                if job.get("external") or row["owner"] == self.owner:
                    continue
                # Another API worker may still own this task. Only recover dead
                # processes; a second app in the same process cannot steal it.
                try:
                    owner_pid = int(row["owner"].split("-", 1)[0])
                    # Containers commonly restart with PID 1. A prior runtime
                    # from this PID is alive only if this process still owns it.
                    if owner_pid == os.getpid():
                        if row["owner"] in _LIVE_OWNERS:
                            continue
                    else:
                        os.kill(owner_pid, 0)
                        continue
                except (ProcessLookupError, ValueError):
                    pass
                except PermissionError:
                    continue
                job.update(
                    status="failed",
                    error="Processing was interrupted by a server restart. Submit a new run.",
                    finished_at=now(),
                )
                self._write(connection, job)
                self._discard_upload(job)

    def _write(self, connection: sqlite3.Connection, job: dict[str, object]) -> None:
        job["updated_at"] = now()
        connection.execute(
            "UPDATE jobs SET status=?, updated_at=?, payload=? WHERE id=?",
            (job["status"], job["updated_at"], json.dumps(job), job["job_id"]),
        )

    def create(
        self, survey_id: str, kind: str, **metadata: object
    ) -> dict[str, object]:
        with self._guard:
            if self._closed:
                raise QueueFull("The server is shutting down; retry shortly")
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM jobs WHERE survey_id=? AND status IN ('queued','running') LIMIT 1",
                    (survey_id,),
                ).fetchone()
                if row:
                    raise JobConflict(self._decode(row) or {})
                active_rows = connection.execute(
                    "SELECT payload FROM jobs WHERE status IN ('queued','running')"
                ).fetchall()
                count = sum(
                    not json.loads(row["payload"]).get("external")
                    for row in active_rows
                )
                if not metadata.get("external") and count >= self.capacity:
                    raise QueueFull(
                        "The processing queue is full; retry after a current job finishes"
                    )
                created = now()
                job_id = uuid4().hex
                job: dict[str, object] = {
                    **metadata,
                    "job_id": job_id,
                    "survey_id": survey_id,
                    "kind": kind,
                    "status": "queued",
                    "created_at": created,
                    "updated_at": created,
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "status_url": f"/api/jobs/{job_id}",
                }
                connection.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        survey_id,
                        kind,
                        "queued",
                        self.owner,
                        created,
                        created,
                        json.dumps(job),
                    ),
                )
                return job

    def get(self, job_id: str) -> dict[str, object] | None:
        if not self.path.is_file():
            return None
        with self._connect() as connection:
            return self._decode(
                connection.execute(
                    "SELECT * FROM jobs WHERE id=?", (job_id,)
                ).fetchone()
            )

    def latest(
        self, survey_id: str, kind: str | None = None
    ) -> dict[str, object] | None:
        if not self.path.is_file():
            return None
        query = "SELECT * FROM jobs WHERE survey_id=?"
        args: tuple[str, ...] = (survey_id,)
        if kind:
            query += " AND kind=?"
            args += (kind,)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as connection:
            return self._decode(connection.execute(query, args).fetchone())

    def surveys(self) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE kind='survey' ORDER BY created_at DESC"
            ).fetchall()
            latest: dict[str, dict[str, object]] = {}
            for row in rows:
                latest.setdefault(row["survey_id"], self._decode(row) or {})
            return list(latest.values())

    def update(self, job_id: str, **values: object) -> dict[str, object]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._decode(
                connection.execute(
                    "SELECT * FROM jobs WHERE id=?", (job_id,)
                ).fetchone()
            )
            if job is None:
                raise KeyError(job_id)
            job.update(values)
            self._write(connection, job)
            return job

    def submit(
        self, job: dict[str, object], task: Callable[[], dict[str, object] | None]
    ) -> None:
        with self._guard:
            if self._closed:
                self.update(
                    str(job["job_id"]),
                    status="failed",
                    error="Server is shutting down",
                    finished_at=now(),
                )
                raise QueueFull("The server is shutting down; retry shortly")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.workers, thread_name_prefix="marg-job"
                )
            self._executor.submit(self._execute, str(job["job_id"]), task)

    def _execute(
        self, job_id: str, task: Callable[[], dict[str, object] | None]
    ) -> None:
        self.update(job_id, status="running", started_at=now())
        try:
            result = task() or {}
        except Exception as error:
            LOGGER.exception("Background job %s failed", job_id)
            message = (
                str(error)
                if isinstance(error, JobFailure)
                else "Processing failed. Check the server logs, then retry."
            )
            self.update(job_id, status="failed", error=message, finished_at=now())
        else:
            self.update(job_id, **result, status="succeeded", finished_at=now())

    def shutdown(self) -> None:
        with self._guard:
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if self.path.is_file():
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE owner=? AND status IN ('queued','running')",
                    (self.owner,),
                ).fetchall()
                for row in rows:
                    job = self._decode(row)
                    assert job is not None
                    if not job.get("external"):
                        job.update(
                            status="failed",
                            error="Processing was interrupted by server shutdown. Submit a new run.",
                            finished_at=now(),
                        )
                        self._write(connection, job)
                        self._discard_upload(job)
        _LIVE_OWNERS.pop(self.owner, None)
