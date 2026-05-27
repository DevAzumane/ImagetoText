"""
api/middleware/job_store.py  –  Thread-safe in-memory job registry.

Original helper names _update_job / _get_job are kept as aliases.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def update_job(run_id: str, **fields: Any) -> None:
    with _lock:
        if run_id not in _jobs:
            _jobs[run_id] = {"run_id": run_id}
        _jobs[run_id].update(fields)


def get_job(run_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(run_id)
        return dict(job) if job is not None else None


def delete_job(run_id: str) -> None:
    with _lock:
        _jobs.pop(run_id, None)


# ── Legacy aliases ────────────────────────────────────────────────────────────
_update_job = update_job
_get_job = get_job
