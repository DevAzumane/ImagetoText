"""
api/middleware/error_middleware.py  –  Global exception → JSON response mapping
"""

from __future__ import annotations

import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("scribeocr")


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(FileNotFoundError)
    async def not_found(_: Request, exc: FileNotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(PermissionError)
    async def permission_error(_: Request, exc: PermissionError):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(Exception)
    async def generic(_: Request, exc: Exception):
        logger.error("Unhandled exception:\n%s", traceback.format_exc())
        return JSONResponse(
            {"detail": "An unexpected server error occurred."},
            status_code=500,
        )
