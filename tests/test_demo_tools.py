from __future__ import annotations

import json
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


def test_demo_tools_expose_json_safe_registry_discovery(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    dataset_path.write_bytes(b"placeholder")
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))

    models = tools.list_model_capabilities()
    resolved = tools.resolve_model_capability(key_or_alias="GKBF")
    profiles = tools.list_profile_capabilities()
    training = tools.list_training_capabilities()
    training_detail = tools.describe_training_capability(model_key="kbf", dataset_kind="regular")

    assert models["ok"] is True
    assert resolved["ok"] is True
    assert profiles["ok"] is True
    assert training["ok"] is True
    assert training_detail["ok"] is True
    assert resolved["data"]["capability"]["key"] == "kbf"
    assert training_detail["data"]["detail"]["capability"]["model_key"] == "kbf"
    assert any(
        entry["key"] == "repeat"
        for entry in training_detail["data"]["detail"]["phase_entry_schemas"]
    )
    json.dumps(models)
    json.dumps(profiles)
    json.dumps(training)
    json.dumps(training_detail)


def test_demo_tools_filter_training_capabilities_by_dataset_handle(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    dataset_path.write_bytes(b"placeholder")
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    dataset_handle = tools.register_dataset_file(path=str(dataset_path), kind="graph")["data"][
        "summary"
    ]["handle"]

    response = tools.list_training_capabilities(dataset_handle=dataset_handle)

    assert response["ok"] is True
    assert response["data"]["dataset_kind"] == "graph"
    assert {capability["dataset_kind"] for capability in response["data"]["capabilities"]} == {
        "graph"
    }


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
        "compile_analysis_request",
        "compile_training_request",
        "describe_training_capability",
        "describe_object",
        "evaluate_checkpoint",
        "evaluate_model",
        "inspect_dataset",
        "list_analysis_capabilities",
        "list_model_capabilities",
        "list_objects",
        "list_profile_capabilities",
        "list_training_capabilities",
        "plan_checkpoint_prediction",
        "prepare_prediction_request",
        "register_dataset_file",
        "register_checkpoint",
        "resolve_model_capability",
        "run_analysis_request",
        "train_compiled_request",
        "train_model",
    }
    response = server.tools["register_checkpoint"](
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
    )
    assert response["ok"] is True
