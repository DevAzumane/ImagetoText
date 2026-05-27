"""
api/schemas/job_schemas.py  –  Pydantic models for job endpoints
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ── Job state ─────────────────────────────────────────────────────────────────

JobStatus = Literal["queued", "running", "done", "error"]


class JobStartResponse(BaseModel):
    run_id: str


class JobProgress(BaseModel):
    run_id: str
    status: JobStatus
    percent: int = Field(ge=0, le=100)
    message: str
    download_url: str | None = None
    error: str | None = None


# ── Font catalogue ────────────────────────────────────────────────────────────

class FontEntry(BaseModel):
    key: str
    label: str


class FontsResponse(BaseModel):
    fonts: list[FontEntry]
    preview_text: str
