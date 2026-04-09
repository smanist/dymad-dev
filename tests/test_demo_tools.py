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


def test_demo_tools_discovery_and_validation_surfaces_are_json_safe(tmp_path) -> None:
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    dataset_path = tmp_path / "train.npz"
    dataset_path.write_bytes(b"placeholder")
    dataset = tools.register_dataset_file(path=str(dataset_path))

    families = tools.list_model_families()
    family = tools.describe_model_family(model_ref="dymad.models.collections:KBF")
    profiles = tools.list_reference_profiles(dataset_kind="regular")
    profile = tools.describe_reference_profile(profile_name="kbf-regular-default")
    compatibility = tools.validate_dataset_compatibility(
        dataset_handle=dataset["data"]["summary"]["handle"],
        model_ref="dymad.models.collections:LDM",
    )

    assert families["ok"] is True
    assert family["ok"] is True
    assert profiles["ok"] is True
    assert profile["ok"] is True
    assert compatibility["ok"] is True


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
        "describe_model_family",
        "describe_reference_profile",
        "evaluate_model",
        "inspect_dataset",
        "inspect_training_run",
        "list_objects",
        "list_model_families",
        "list_reference_profiles",
        "list_training_artifacts",
        "materialize_training_config",
        "register_dataset_file",
        "register_checkpoint",
        "train_model",
        "validate_dataset_compatibility",
        "validate_training_config",
    }
    response = server.tools["register_checkpoint"](
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
    )
    assert response["ok"] is True
