"""
api/routers/fonts.py  –  GET /api/fonts   GET /api/font-file/{font_key}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.schemas import FontEntry, FontsResponse
from configs.settings import Settings, get_settings

router = APIRouter()

FONT_KEY_ALIASES = {
    "patrick_hand": "patrick",
}


@router.get("/fonts", response_model=FontsResponse)
async def list_fonts(settings: Settings = Depends(get_settings)):
    """Return the full font catalogue for the React font picker."""
    entries = [
        FontEntry(key=key, label=meta["label"])
        for key, meta in settings.FONT_OPTIONS.items()
    ]
    return FontsResponse(fonts=entries, preview_text=settings.PREVIEW_TEXT)


@router.get("/font-file/{font_key}")
async def font_file(font_key: str, settings: Settings = Depends(get_settings)):
    """Serve the raw TTF so React can render live @font-face previews."""
    options = settings.FONT_OPTIONS
    font_key = FONT_KEY_ALIASES.get(font_key, font_key)
    if font_key not in options:
        raise HTTPException(status_code=404, detail=f"Font '{font_key}' not found.")
    path = options[font_key]["path"]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Font file for '{font_key}' is missing on disk.",
        )
    return FileResponse(str(path), media_type="font/ttf")
