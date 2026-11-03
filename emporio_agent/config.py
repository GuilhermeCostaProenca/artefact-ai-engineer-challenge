"""Runtime configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Campo Grande/MS does not observe daylight saving time (UTC-4).
STORE_TIMEZONE = ZoneInfo("America/Campo_Grande")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_EFFORTS = ("low", "medium", "high")


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    data_dir: Path
    runtime_dir: Path
    fixed_now: datetime | None

    def now(self) -> datetime:
        """Current time in the store's time zone (or the frozen demo clock)."""
        return self.fixed_now or datetime.now(STORE_TIMEZONE)


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    effort = os.getenv("EMPORIO_EFFORT", "medium").strip().lower()
    if effort not in ALLOWED_EFFORTS:
        raise ValueError(f"EMPORIO_EFFORT must be one of {ALLOWED_EFFORTS}, got {effort!r}")

    fixed_now = None
    raw_now = os.getenv("EMPORIO_NOW", "").strip()
    if raw_now:
        fixed_now = datetime.fromisoformat(raw_now)
        if fixed_now.tzinfo is None:
            fixed_now = fixed_now.replace(tzinfo=STORE_TIMEZONE)

    return Settings(
        model=os.getenv("EMPORIO_MODEL", "claude-opus-5-5").strip(),
        effort=effort,
        data_dir=_resolve(os.getenv("EMPORIO_DATA_DIR", "data")),
        runtime_dir=_resolve(os.getenv("EMPORIO_RUNTIME_DIR", "runtime")),
        fixed_now=fixed_now,
    )


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
