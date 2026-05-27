"""
api/middleware/file_validator.py  –  Upload guards used by job router
"""

from __future__ import annotations

from pathlib import Path
from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB


class FileValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)


class BufferedUpload:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.data = data

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.data)


def validate_image(file: UploadFile, *, required: bool = True) -> None:
    if not file or not file.filename:
        if required:
            raise FileValidationError("No file was uploaded.")
        return
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported type '{ext}'. Allowed: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )


def validate_images(files: list[UploadFile]) -> None:
    if not files:
        raise FileValidationError("At least one input image is required.")
    for f in files:
        validate_image(f)


async def buffer_upload(file: UploadFile | None) -> BufferedUpload | None:
    """Read an UploadFile into a BytesIO buffer (preserves original _buffer_upload logic)."""
    if not file or not file.filename:
        return None
    data = await file.read()
    return BufferedUpload(file.filename, data)


# Legacy alias (original app.py name)
_buffer_upload = buffer_upload
