import io
import sys

import pytest

from emporio_agent import cli
from emporio_agent.config import load_settings


def test_invalid_effort_is_rejected_with_clear_message(monkeypatch):
    monkeypatch.setenv("EMPORIO_EFFORT", "turbo")
    with pytest.raises(ValueError, match="EMPORIO_EFFORT"):
        load_settings()


def test_invalid_clock_is_rejected_with_clear_message(monkeypatch):
    monkeypatch.setenv("EMPORIO_NOW", "ontem")
    with pytest.raises(ValueError, match="EMPORIO_NOW must be an ISO 8601"):
        load_settings()


def test_cli_reports_bad_config_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("EMPORIO_DATA_DIR", "does-not-exist")
    assert cli.main([]) == 1


def test_legacy_code_page_output_is_switched_to_utf8(monkeypatch):
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", legacy)
    cli.ensure_utf8_output()
    print("🎸 Empório")  # raised UnicodeEncodeError before the fix
    assert sys.stdout.encoding == "utf-8"
