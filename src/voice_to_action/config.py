"""Runtime configuration loaded from environment variables (optionally via a ``.env`` file)."""

from __future__ import annotations

import os

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only hit when dotenv is not installed
    pass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DB_PATH: str = os.getenv("VTA_DB_PATH", "data/tasks.db")
SPACY_MODEL: str = os.getenv("VTA_SPACY_MODEL", "en_core_web_sm")
VOICE_BACKEND: str = os.getenv("VTA_VOICE_BACKEND", "google")
WHISPER_MODEL: str = os.getenv("VTA_WHISPER_MODEL", "base")
LANGUAGE: str = os.getenv("VTA_LANGUAGE", "en-US")
DUE_SOON_MINUTES: int = _int_env("VTA_DUE_SOON_MINUTES", 15)
END_OF_DAY_HOUR: int = _int_env("VTA_EOD_HOUR", 18)
