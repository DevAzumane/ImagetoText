"""
api/services/job_service.py  –  Async-safe bridge between the FastAPI router
and the synchronous pipeline code.

Runs the blocking pipeline in a ThreadPoolExecutor so the event loop
stays free.
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api.services.pipeline_service import run_automatic_job
from api.middleware.job_store import update_job
from api.middleware.file_validator import BufferedUpload

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")
FONT_KEY_ALIASES = {
    "patrick_hand": "patrick",
}


def safe_name(name: str) -> str:
    """Sanitise a filename (original _safe_name logic)."""
    safe = _SAFE_RE.sub("_", Path(name).name).strip()
    return safe or "upload.png"


def new_run_id() -> str:
    return uuid.uuid4().hex[:10]


# ── Background worker ─────────────────────────────────────────────────────────

def _run_pipeline(
    run_id: str,
    form_data: dict,
    input_files: list[BufferedUpload],
    alphabet_file: BufferedUpload | None,
) -> None:
    """Executed in a thread. Calls the existing pipeline code unchanged."""
    try:

        def report(pct: int, msg: str) -> None:
            update_job(
                run_id,
                status="running",
                percent=max(0, min(99, int(pct))),
                message=msg,
            )

        result = run_automatic_job(
            run_id=run_id,
            input_files=input_files,
            alphabet_file=alphabet_file,
            title=form_data.get("title", "Notes"),
            page_template=form_data.get("page_template", "classic"),
            output_format=form_data.get("output_format", "pdf"),
            handwriting_mode=form_data.get("handwriting_mode", "custom"),
            font_key=FONT_KEY_ALIASES.get(
                form_data.get("font_key", "caveat"),
                form_data.get("font_key", "caveat"),
            ),
            summary_lines=int(form_data.get("summary_lines", 3)),
            progress_callback=report,
        )

        dl_name = Path(result["download_path"]).name
        update_job(
            run_id,
            status="done",
            percent=100,
            message="Complete",
            download_url=f"/api/download/{run_id}/{safe_name(dl_name)}",
        )
    except Exception as exc:  # noqa: BLE001
        update_job(
            run_id,
            status="error",
            percent=100,
            message="Failed",
            error=str(exc),
        )


async def launch_job(
    run_id: str,
    form_data: dict,
    input_files: list[BufferedUpload],
    alphabet_file: BufferedUpload | None,
) -> None:
    """Non-blocking: submits the pipeline to the thread pool."""
    update_job(run_id, status="queued", percent=1, message="Queued")
    _executor.submit(
        _run_pipeline,
        run_id,
        form_data,
        input_files,
        alphabet_file,
    )
