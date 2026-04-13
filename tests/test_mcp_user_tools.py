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
    assert detail["data"]["dataset_kind"] == "regular"
    assert detail["data"]["detail"]["capability"]["model_key"] == "kbf"
    assert compiled["data"]["compiled_request"]["model_key"] == "kbf"
    assert compiled["data"]["compiled_request"]["reference_profile"] == "kbf-regular-default"
    assert trained["ok"] is True
    assert trained["data"]["result"]["reference_profile"] == "kbf-regular-default"
    assert evaluated["ok"] is True
    assert evaluated["data"]["result"]["metrics"]["rmse_mean"] >= 0.0
    assert materialized["model"]["koopman_dimension"] == 6


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
