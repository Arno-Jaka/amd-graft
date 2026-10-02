"""Configuration tests.

Config is read at import time, so a mistake here is silent: a blank model
name simply disables the agent and nothing fails. These tests reload the
module with a controlled environment.
"""

from __future__ import annotations

import importlib

import pytest

from graft import config


def reload(monkeypatch, **env):
    for key in (
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY_ENV",
        "AGENT_ENABLED",
        "AGENT_MAX_TOOL_CALLS",
        "AGENT_TIMEOUT_S",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config)


def teardown_function():
    importlib.reload(config)


def test_llm_settings_come_from_the_environment(monkeypatch):
    reloaded = reload(monkeypatch, LLM_MODEL="qwen", LLM_BASE_URL="http://h:8000/v1")
    assert reloaded.LLM_MODEL == "qwen"
    assert reloaded.LLM_BASE_URL == "http://h:8000/v1"


def test_blank_endpoint_disables_the_agent_by_default(monkeypatch):
    reloaded = reload(monkeypatch)
    assert reloaded.LLM_MODEL == ""
    assert reloaded.LLM_BASE_URL == ""
    assert reloaded.AGENT_ENABLED is True


def test_values_are_stripped(monkeypatch):
    reloaded = reload(monkeypatch, LLM_MODEL="  qwen  ", LLM_BASE_URL=" http://h ")
    assert reloaded.LLM_MODEL == "qwen"
    assert reloaded.LLM_BASE_URL == "http://h"


@pytest.mark.parametrize(
    "raw,expected",
    [("0", False), ("false", False), ("no", False), ("1", True), ("on", True)],
)
def test_agent_enabled_is_parsed(monkeypatch, raw, expected):
    assert reload(monkeypatch, AGENT_ENABLED=raw).AGENT_ENABLED is expected


def test_numeric_settings_are_parsed(monkeypatch):
    reloaded = reload(monkeypatch, AGENT_MAX_TOOL_CALLS="3", AGENT_TIMEOUT_S="12.5")
    assert reloaded.AGENT_MAX_TOOL_CALLS == 3
    assert reloaded.AGENT_TIMEOUT_S == 12.5


def test_dotenv_parsing_handles_comments_quotes_and_junk(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# a comment\n"
        "\n"
        "GRAFT_TEST_PLAIN=plain\n"
        'GRAFT_TEST_QUOTED="http://quoted:8000/v1"\n'
        "GRAFT_TEST_SINGLE='single'\n"
        "GRAFT_TEST_SPACED =  spaced  \n"
        "MANGLED_LINE_WITHOUT_EQUALS\n",
        encoding="utf-8",
    )
    import os

    for key in list(os.environ):
        if key.startswith("GRAFT_TEST_"):
            monkeypatch.delenv(key, raising=False)

    importlib.reload(config)._load_dotenv(dotenv)

    assert os.environ["GRAFT_TEST_PLAIN"] == "plain"
    assert os.environ["GRAFT_TEST_QUOTED"] == "http://quoted:8000/v1"
    assert os.environ["GRAFT_TEST_SINGLE"] == "single"
    assert os.environ["GRAFT_TEST_SPACED"] == "spaced"
    assert "MANGLED_LINE_WITHOUT_EQUALS" not in os.environ


def test_shell_environment_beats_the_dotenv_file(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-shell")
    reloaded = importlib.reload(config)
    reloaded._load_dotenv(dotenv)
    assert reloaded.LLM_MODEL == "from-shell"


def test_missing_dotenv_is_not_an_error(monkeypatch, tmp_path):
    reloaded = importlib.reload(config)
    reloaded._load_dotenv(tmp_path / "does-not-exist")


def test_device_resolution_falls_back_to_cpu():
    import torch

    device = config.resolve_device()
    assert device.type in {"cpu", "cuda"}
    assert device.type == "cuda" or not torch.cuda.is_available()
