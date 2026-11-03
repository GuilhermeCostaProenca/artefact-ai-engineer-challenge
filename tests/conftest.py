from datetime import datetime

import pytest

from emporio_agent.config import PROJECT_ROOT, STORE_TIMEZONE, Settings
from emporio_agent.data_store import DataStore
from emporio_agent.tools import ToolContext

FIXED_NOW = datetime(2026, 11, 3, 14, 0, tzinfo=STORE_TIMEZONE)  # a Tuesday, store open


@pytest.fixture(scope="session")
def store() -> DataStore:
    return DataStore(PROJECT_ROOT / "data")


@pytest.fixture
def ctx(store, tmp_path) -> ToolContext:
    return ToolContext(store=store, now=lambda: FIXED_NOW, runtime_dir=tmp_path)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        model="claude-opus-5-5",
        effort="medium",
        data_dir=PROJECT_ROOT / "data",
        runtime_dir=tmp_path,
        fixed_now=FIXED_NOW,
    )
