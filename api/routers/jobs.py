"""
api/routers/jobs.py  –  POST /api/start-job   GET /api/progress/{run_id}
"""

from __future__ import annotations

from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Annotated

from api.middleware.file_validator import (
    validate_images,
    buffer_upload,
    FileValidationError,
)
from api.middleware.job_store import get_job
from api.schemas import JobStartResponse, JobProgress
from api.services.job_service import launch_job, new_run_id

router = APIRouter()


@router.post("/start-job", response_model=JobStartResponse, status_code=202)
async def start_job(
    input_images: Annotated[list[UploadFile], File()],
    alphabet_image: Annotated[UploadFile | None, File()] = None,
    title: Annotated[str, Form()] = "Notes",
    page_template: Annotated[str, Form()] = "classic",
    output_format: Annotated[str, Form()] = "pdf",
    handwriting_mode: Annotated[str, Form()] = "custom",
    font_key: Annotated[str, Form()] = "caveat",
    summary_lines: Annotated[int, Form()] = 3,
):
    """Accept uploaded images and start a background OCR pipeline job."""
    validate_images(input_images)

    run_id = new_run_id()

    input_bufs = [await buffer_upload(f) for f in input_images if f and f.filename]
    alphabet_buf = await buffer_upload(alphabet_image)

    form_data = {
        "title": title,
        "page_template": page_template,
        "output_format": output_format,
        "handwriting_mode": handwriting_mode,
        "font_key": font_key,
        "summary_lines": summary_lines,
    }

    await launch_job(run_id, form_data, input_bufs, alphabet_buf)
    return JobStartResponse(run_id=run_id)


@router.get("/progress/{run_id}", response_model=JobProgress)
async def progress(run_id: str):
    """Poll the current state of a running job."""
    job = get_job(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{run_id}' not found.")
    return JobProgress(**job)
