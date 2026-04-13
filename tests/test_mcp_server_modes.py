from __future__ import annotations

import os
import sys
import types

from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import build_server
from dymad.agent.mcp._bootstrap import configure_headless_matplotlib_backend


class FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, object] = {}

    def tool(self, fn):
        self.tools[fn.__name__] = fn
        return fn


def test_build_server_user_mode_registers_only_high_level_tools(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))

    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        mode="user",
    )

    assert "compile_training_request" in server.tools
    assert "compile_analysis_request" in server.tools
    assert "register_checkpoint" not in server.tools
    assert "train_model" not in server.tools


def test_build_server_developer_mode_registers_only_raw_tools(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))

    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        mode="developer",
    )

    assert "register_checkpoint" in server.tools
    assert "train_model" in server.tools
    assert "compile_training_request" not in server.tools
    assert "compile_analysis_request" not in server.tools


def test_configure_headless_matplotlib_backend_defaults_to_agg(monkeypatch) -> None:
    monkeypatch.delenv("MPLBACKEND", raising=False)

    configure_headless_matplotlib_backend()

    assert os.environ["MPLBACKEND"] == "Agg"


def test_configure_headless_matplotlib_backend_respects_existing_backend(monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "TkAgg")

    configure_headless_matplotlib_backend()

    assert os.environ["MPLBACKEND"] == "TkAgg"
