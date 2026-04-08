from __future__ import annotations

import sys
import types

from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import DemoTools, build_server


def test_demo_tools_wrap_success_and_errors(tmp_path) -> None:
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))

    registered = tools.register_checkpoint(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
    )
    checkpoint_handle = registered["data"]["summary"]["handle"]
    request = tools.prepare_prediction_request(
        checkpoint_handle=checkpoint_handle,
        horizon=4,
        has_control=True,
    )
    listed = tools.list_objects()
    missing = tools.describe_object(handle="chk_missing")
    invalid = tools.describe_object(handle="bad")

    assert registered["ok"] is True
    assert request["ok"] is True
    assert request["data"]["summary"]["derived_from"] == checkpoint_handle
    assert [item["kind"] for item in listed["data"]["objects"]] == [
        "checkpoint",
        "prediction_request",
    ]
    assert missing["ok"] is False
    assert missing["error"]["type"] == "ObjectNotFoundError"
    assert invalid["ok"] is False
    assert invalid["error"]["type"] == "ObjectNotFoundError"


def test_plan_checkpoint_prediction_returns_json_safe_plan(tmp_path) -> None:
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))

    response = tools.plan_checkpoint_prediction(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
        horizon=3,
        has_graph=True,
    )

    assert response["ok"] is True
    assert response["data"]["plan"]["entrypoint"] == "dymad.io.checkpoint.load_model"
    assert response["data"]["plan"]["notes"]


def test_build_server_registers_demo_tools(monkeypatch, tmp_path) -> None:
    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self, fn):
            self.tools[fn.__name__] = fn
            return fn

    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))

    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        name="DyMAD Test",
    )

    assert server.name == "DyMAD Test"
    assert set(server.tools) == {
        "describe_object",
        "evaluate_model",
        "inspect_dataset",
        "list_objects",
        "plan_checkpoint_prediction",
        "prepare_prediction_request",
        "register_dataset_file",
        "register_checkpoint",
        "train_model",
    }
    response = server.tools["register_checkpoint"](
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
    )
    assert response["ok"] is True
