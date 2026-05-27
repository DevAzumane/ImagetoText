"""
api/routers/downloads.py  –  GET /api/download/{run_id}/{filename}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.services.job_service import safe_name
from configs.settings import Settings, get_settings

router = APIRouter()


@router.get("/download/{run_id}/{filename}")
async def download(
    run_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
):
    """Stream the generated file back to the caller as an attachment."""
    path = settings.RUNS_DIR / safe_name(run_id) / "rendered" / safe_name(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return FileResponse(
        str(path),
        filename=filename,
        media_type="application/octet-stream",
    )
