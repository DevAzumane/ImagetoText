"""
api/main.py  –  FastAPI application factory
Run: uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.error_middleware import register_exception_handlers
from api.routers import jobs, fonts, downloads
from configs.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    settings.FONTS_DIR.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ScribeOCR API",
        description="Handwriting OCR automation backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom middleware ─────────────────────────────────────────────────────
    app.add_middleware(LoggingMiddleware)
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(jobs.router,      prefix="/api", tags=["jobs"])
    app.include_router(fonts.router,     prefix="/api", tags=["fonts"])
    app.include_router(downloads.router, prefix="/api", tags=["downloads"])

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/api/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    return app


app = create_app()
