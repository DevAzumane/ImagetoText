"""
configs/settings.py  –  Single source of truth for all constants.

All variable names from the original app.py are preserved exactly so the
existing engine/, pipeline/, segmentation/, and writer/ modules keep working
without any import changes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


# ── Absolute root of the project (llama/) ────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(default="change-me-in-production")

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # ── File limits ───────────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH: int = Field(default=50 * 1024 * 1024)  # 50 MB

    # ── Paths (same names used across the project) ────────────────────────────
    BASE_DIR: Path = BASE_DIR
    RUNS_DIR: Path = BASE_DIR / "data" / "runs"
    FONTS_DIR: Path = BASE_DIR / "assets" / "fonts" / "preexisting"
    LOGS_DIR: Path = BASE_DIR / "logs"
    INPUT_DIR: Path = BASE_DIR / "data" / "input"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "output"
    GLYPHS_DIR: Path = BASE_DIR / "glyphs"

    # ── Page templates (original variable name: PAGE_TEMPLATES) ───────────────
    PAGE_TEMPLATES: list[str] = Field(
        default=["random", "classic", "spiral", "exam", "aged"]
    )

    # ── Font options (original variable name: FONT_OPTIONS) ───────────────────
    # Defined as a property below so Path resolution is dynamic.

    # ── Preview text (original variable name: preview_text) ───────────────────
    PREVIEW_TEXT: str = "The quick notes"

    MODEL_NAME: str = "qwen2.5vl:3b"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"debug", "development", "dev"}:
                return True
        return value

    @property
    def FONT_OPTIONS(self) -> dict[str, dict]:
        """Same structure as the original FONT_OPTIONS dict."""
        return {
            "caveat": {
                "label": "Caveat",
                "path": self.BASE_DIR / "assets" / "fonts" / "Caveat-VariableFont_wght.ttf",
            },
            "patrick": {
                "label": "Patrick Hand",
                "path": self.FONTS_DIR / "PatrickHand-Regular.ttf",
            },
            "caveat_bold": {
                "label": "Caveat Bold",
                "path": self.FONTS_DIR / "Caveat-Bold.ttf",
            },
            "caveat_semibold": {
                "label": "Caveat SemiBold",
                "path": self.FONTS_DIR / "Caveat-SemiBold.ttf",
            },
            "apalu": {
                "label": "Apalu",
                "path": self.FONTS_DIR / "Apalu.ttf",
            },
        }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (FastAPI Depends-friendly)."""
    return Settings()


# Compatibility exports for the original engine/pipeline modules.
_settings = get_settings()
RUNS_DIR = _settings.RUNS_DIR
GLYPH_OUTPUT_DIR = _settings.BASE_DIR / "glyphs" / "segmented"
PAGE_TEMPLATES = tuple(_settings.PAGE_TEMPLATES)
FONT_OPTIONS = _settings.FONT_OPTIONS
