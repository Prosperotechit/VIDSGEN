"""SQLite job repository.

Small and dependency-free. All status changes go through `transition()`, a
conditional UPDATE (``... WHERE status IN (...)``). That makes races safe: for example,
a worker cannot overwrite a job the user has just cancelled.

The interface is deliberately narrow so it can be re-implemented on Postgres later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import GenerationParams, Job, JobError, JobStatus, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    mode              TEXT NOT NULL,
    status            TEXT NOT NULL,
    progress          REAL NOT NULL DEFAULT 0,
    params            TEXT NOT NULL,
    input_image_key   TEXT,
    input_image_info  TEXT,
    output_key        TEXT,
    output_size_bytes INTEGER,
    provider          TEXT NOT NULL,
    provider_job_id   TEXT,
    error_code        TEXT,
    error_message     TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    started_at        TEXT,
    completed_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

_SIMPLE_FIELDS = {
    "progress", "input_image_key", "output_key", "output_size_bytes",
    "provider_job_id", "started_at", "completed_at",
}


def _ts(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


class JobRepository:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ mapping
    @staticmethod
    def _row_to_job(r: sqlite3.Row) -> Job:
        err = JobError(code=r["error_code"], message=r["error_message"] or "") if r["error_code"] else None
        return Job(
            id=r["id"], mode=r["mode"], status=r["status"], progress=r["progress"],
            params=GenerationParams.model_validate_json(r["params"]),
            input_image_key=r["input_image_key"],
            input_image_info=json.loads(r["input_image_info"]) if r["input_image_info"] else None,
            output_key=r["output_key"], output_size_bytes=r["output_size_bytes"],
            provider=r["provider"], provider_job_id=r["provider_job_id"], error=err,
            created_at=r["created_at"], updated_at=r["updated_at"],
            started_at=r["started_at"], completed_at=r["completed_at"],
        )

    @staticmethod
    def _encode(fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in fields.items():
            if k == "status":
                out["status"] = JobStatus(v).value
            elif k == "error":
                out["error_code"] = v.code if v else None
                out["error_message"] = v.message if v else None
            elif k in _SIMPLE_FIELDS:
                out[k] = _ts(v) if isinstance(v, datetime) else v
            else:
                raise KeyError(f"Field {k!r} cannot be updated")
        out["updated_at"] = _ts(utcnow())
        return out

    # ------------------------------------------------------------ queries
    def insert(self, job: Job) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO jobs (id, mode, status, progress, params, input_image_key, input_image_info,
                       output_key, output_size_bytes, provider, provider_job_id, error_code, error_message,
                       created_at, updated_at, started_at, completed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job.id, job.mode.value, job.status.value, job.progress, job.params.model_dump_json(),
                 job.input_image_key, json.dumps(job.input_image_info) if job.input_image_info else None,
                 job.output_key, job.output_size_bytes, job.provider, job.provider_job_id,
                 job.error.code if job.error else None, job.error.message if job.error else None,
                 _ts(job.created_at), _ts(job.updated_at), _ts(job.started_at), _ts(job.completed_at)),
            )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(r) if r else None

    def list(self, *, limit: int, offset: int, status: JobStatus | None = None) -> tuple[list[Job], int]:
        where, args = ("WHERE status = ?", [status.value]) if status else ("", [])
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM jobs {where}", args).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [self._row_to_job(r) for r in rows], total

    def ids_with_status(self, status: JobStatus) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at, rowid", (status.value,)
            ).fetchall()
        return [r["id"] for r in rows]

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {s.value: 0 for s in JobStatus} | {r["status"]: r["n"] for r in rows}

    def transition(self, job_id: str, from_statuses: Iterable[JobStatus], **fields: Any) -> bool:
        """Atomically update a job only if its current status is in `from_statuses`.

        Returns True if the row was updated. Use this for every change, so that
        concurrent actors (worker vs. cancel request) cannot clobber each other.
        """
        enc = self._encode(fields)
        froms = [JobStatus(s).value for s in from_statuses]
        sets = ", ".join(f"{k} = ?" for k in enc)
        marks = ",".join("?" * len(froms))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET {sets} WHERE id = ? AND status IN ({marks})",
                [*enc.values(), job_id, *froms],
            )
        return cur.rowcount == 1
