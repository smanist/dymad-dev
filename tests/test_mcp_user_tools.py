from __future__ import annotations

from pathlib import Path

import yaml

import dymad.io
from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import UserTools
from tests.test_mcp_train_eval_tools import _patch_fake_trainers, _write_regular_dataset


def test_user_tools_compile_train_and_evaluate_flow(tmp_path, monkeypatch) -> None:
    _patch_fake_trainers(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    def fake_load_model(model, checkpoint_path, *, context=None, **kwargs):
        del model, checkpoint_path, context, kwargs

        def predict_fn(x0, t, u=None, p=None, **inner_kwargs):
            del t, u, p, inner_kwargs
            return 0.5 * x0

        return object(), predict_fn

    def fake_plot_trajectory(traj, ts, model_name=None, prefix=".", **kwargs):
        del traj, ts, kwargs
        path = Path(prefix) / f"{model_name}_prediction.png"
        path.write_bytes(b"plot")

    monkeypatch.setattr(dymad.io, "load_model", fake_load_model)
    monkeypatch.setattr("dymad.agent.exec.workflow.plot_trajectory", fake_plot_trajectory)

    tools = UserTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_dataset_handle = tools._context.facade.register_dataset_file(
        path=str(dataset_path)
    ).handle
    detail = tools.describe_training_capability(
        model_key="kbf",
        dataset_handle=train_dataset_handle,
    )
    evaluation = tools.list_evaluation_capabilities(dataset_handle=train_dataset_handle)

    compiled = tools.compile_training_request(
        train_dataset_handle=train_dataset_handle,
        model_key="kbf",
        run_name="user_mode_run",
        overrides={"model": {"koopman_dimension": 6}},
    )
    compiled_handle = compiled["data"]["summary"]["handle"]
    trained = tools.train_compiled_request(
        compiled_request_handle=compiled_handle,
        artifact_root=str(tmp_path / "outputs"),
    )
    checkpoint_handle = trained["data"]["result"]["checkpoint_summary"]["handle"]
    evaluated = tools.evaluate_checkpoint(
        checkpoint_handle=checkpoint_handle,
        test_dataset_handle=train_dataset_handle,
        metric="rollout_rmse",
        artifact_root=str(tmp_path / "eval"),
    )

    config_path = tmp_path / "outputs" / "user_mode_run.yaml"
    materialized = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert compiled["ok"] is True
    assert detail["ok"] is True
    assert evaluation["ok"] is True
    assert detail["data"]["dataset_kind"] == "regular"
    assert detail["data"]["detail"]["capability"]["model_key"] == "kbf"
    assert detail["data"]["detail"]["translation_guidance"][0] == (
        "For any ordered trainer names mentioned by the user, emit one "
        "overrides.phases entry per trainer in the same order."
    )
    assert (
        detail["data"]["detail"]["translation_guidance"][1]
        == "Supported optimizer trainer names are Linear, Weak, and NODE."
    )
    assert (
        detail["data"]["detail"]["constraint_notes"][0]
        == "Setting encoder_layers=0 or decoder_layers=0 only yields a true identity map "
        "when the latent dimension matches the dataset state dimension."
    )
    assert detail["data"]["detail"]["examples"][0] == {
        "name": "linear_then_node_from_plain_english",
        "user_request": "Use staged training: first a Linear phase for initialization, then a "
        "NODE phase for refinement.",
        "overrides": {
            "phases": [
                {"trainer": "Linear", "name": "initialization"},
                {"trainer": "NODE", "name": "refinement"},
            ]
        },
        "notes": [
            "This uses the minimal legacy optimizer shorthand because the user did not "
            "specify per-phase hyperparameters."
        ],
    }
    assert detail["data"]["detail"]["examples"][1] == {
        "name": "weak_then_node_from_plain_english",
        "user_request": "Use weak form training first, then refine with NODE.",
        "overrides": {
            "phases": [
                {"trainer": "Weak"},
                {"trainer": "NODE"},
            ]
        },
        "notes": [
            "The same ordered-trainer translation rule applies to any supported mix of "
            "Linear, Weak, and NODE phases."
        ],
    }
    assert evaluation["data"]["capabilities"][0]["supported_metrics"] == ["rollout_rmse"]
    assert compiled["data"]["compiled_request"]["model_key"] == "kbf"
    assert compiled["data"]["compiled_request"]["reference_profile"] == "kbf-regular-default"
    assert trained["ok"] is True
    assert trained["data"]["result"]["reference_profile"] == "kbf-regular-default"
    assert evaluated["ok"] is True
    assert evaluated["data"]["result"]["metrics"]["rmse_mean"] >= 0.0
    assert materialized["model"]["koopman_dimension"] == 6


def test_user_tools_compile_training_request_accepts_json_string_overrides(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    tools = UserTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_dataset_handle = tools._context.facade.register_dataset_file(
        path=str(dataset_path)
    ).handle

    compiled = tools.compile_training_request(
        train_dataset_handle=train_dataset_handle,
        model_key="kbf",
        overrides='{"model": {"koopman_dimension": 5}}',
    )

    assert compiled["ok"] is True
    assert (
        compiled["data"]["compiled_request"]["effective_config"]["model"]["koopman_dimension"] == 5
    )


def test_user_tools_compile_training_request_surfaces_identity_dimension_mismatch(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    tools = UserTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_dataset_handle = tools._context.facade.register_dataset_file(
        path=str(dataset_path)
    ).handle

    compiled = tools.compile_training_request(
        train_dataset_handle=train_dataset_handle,
        model_key="lti",
        overrides={"model": {"encoder_layers": 0, "decoder_layers": 0}},
    )

    assert compiled["ok"] is False
    assert compiled["error"]["type"] == "TrainingCompileValidationError"
    assert "identity map" in compiled["error"]["message"]


def test_build_server_registers_user_tools(monkeypatch, tmp_path) -> None:
    import sys
    import types

    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self, fn):
            self.tools[fn.__name__] = fn
            return fn

    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))

    from dymad.agent.mcp import build_server

    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        name="DyMAD User Tools Test",
    )

    assert "compile_training_request" in server.tools
    assert "train_compiled_request" in server.tools
    assert "evaluate_checkpoint" in server.tools
    assert "list_evaluation_capabilities" in server.tools


def test_build_server_compile_training_request_accepts_json_string_overrides(
    monkeypatch, tmp_path
) -> None:
    import sys
    import types

    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self, fn):
            self.tools[fn.__name__] = fn
            return fn

    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))

    from dymad.agent.mcp import build_server

    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    train_dataset_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle
    server = build_server(context=context, name="DyMAD User Tools Test")

    compiled = server.tools["compile_training_request"](
        train_dataset_handle=train_dataset_handle,
        model_key="kbf",
        overrides='{"model": {"koopman_dimension": 5}}',
    )

    assert compiled["ok"] is True
    assert (
        compiled["data"]["compiled_request"]["effective_config"]["model"]["koopman_dimension"] == 5
    )
