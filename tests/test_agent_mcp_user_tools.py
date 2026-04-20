from __future__ import annotations

import time
from pathlib import Path

import yaml

import dymad.io
from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import UserTools
from tests.test_agent_mcp_train_eval_tools import (
    _configure_worker_bootstrap,
    _write_regular_dataset,
)


def test_user_tools_compile_start_poll_and_evaluate_flow(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch)
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
    started = tools.start_training_run(
        compiled_request_handle=compiled_handle,
        artifact_root=str(tmp_path / "outputs"),
    )
    training_run_handle = started["data"]["summary"]["handle"]
    for _ in range(100):
        polled = tools.describe_training_run(training_run_handle=training_run_handle)
        assert polled["ok"] is True
        if polled["data"]["training_run"]["status"] == "SUCCEEDED":
            break
        if polled["data"]["training_run"]["status"] == "FAILED":
            raise AssertionError(polled["data"]["training_run"])
        time.sleep(0.05)
    else:
        raise AssertionError("training run did not finish")
    checkpoint_handle = polled["data"]["training_run"]["checkpoint_handle"]
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
        == "Encode hyperparameter sweep requests as overrides.cv.param_grid, with optional "
        "overrides.cv.metric to choose the optimization metric."
    )
    assert (
        detail["data"]["detail"]["translation_guidance"][2]
        == "For Nelder-Mead-like requests, set overrides.cv.search.mode='nelder_mead_like' "
        "and provide optional simplex coefficients or max_iterations in overrides.cv.search."
    )
    assert (
        detail["data"]["detail"]["translation_guidance"][3]
        == "Use overrides.cv.selection to control model choice policy (goal and tie_breakers)."
    )
    assert (
        detail["data"]["detail"]["translation_guidance"][4]
        == "Supported optimizer trainer names are Linear, Weak, and NODE."
    )
    assert detail["data"]["detail"]["cv_schema"] == {
        "supported": True,
        "workflow_kind": "single_split_param_sweep",
        "allowed_keys": ["param_grid", "metric", "search", "selection"],
        "default_metric": "total",
        "param_grid_value_forms": ["list", "linspace_tuple", "logspace_tuple"],
        "search_schema": {
            "allowed_keys": [
                "mode",
                "max_iterations",
                "reflection",
                "expansion",
                "contraction",
                "shrink",
            ],
            "mode_options": ["grid", "nelder_mead_like"],
            "default_mode": "grid",
        },
        "selection_schema": {
            "allowed_keys": ["goal", "tie_breakers"],
            "goal_options": ["minimize", "maximize"],
            "default_goal": "minimize",
            "tie_breaker_options": ["std_metric", "param_l1", "combo_index"],
            "default_tie_breakers": ["std_metric", "combo_index"],
        },
        "notes": [
            "This v1 user-mode CV surface runs the existing single-split parameter sweep; it is "
            "not true k-fold cross-validation.",
            "The best parameter combination is selected by cv.selection (default: minimize mean "
            "metric, then std_metric, then combo_index).",
            "Param-grid dotted keys may target either explicit phases.* paths or legacy "
            "training.* shorthand, which is normalized onto the first optimizer phase.",
            "cv.search.mode='nelder_mead_like' runs a Nelder-Mead-like adaptive candidate "
            "path over numeric param_grid values in single-split mode; non-numeric values "
            "fall back to grid order.",
        ],
    }
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
    assert detail["data"]["detail"]["examples"][2] == {
        "name": "hyperparameter_sweep_from_plain_english",
        "user_request": "Sweep Koopman dimensions 4 and 6, and choose the model with the lowest "
        "total validation metric.",
        "overrides": {
            "cv": {
                "param_grid": {"model.koopman_dimension": [4, 6]},
                "metric": "total",
            }
        },
        "notes": [
            "This uses the existing single-split CV sweep runtime rather than true k-fold "
            "cross-validation."
        ],
    }
    assert detail["data"]["detail"]["examples"][3] == {
        "name": "nelder_mead_like_single_split_cv",
        "user_request": "Use a Nelder-Mead-like single-split CV policy, stopping after 12 "
        "iterations and preferring lower variance when metrics tie.",
        "overrides": {
            "cv": {
                "param_grid": {"model.koopman_dimension": [4, 6, 8]},
                "search": {
                    "mode": "nelder_mead_like",
                    "max_iterations": 12,
                    "reflection": 1.0,
                    "expansion": 2.0,
                    "contraction": 0.5,
                    "shrink": 0.5,
                },
                "selection": {
                    "goal": "minimize",
                    "tie_breakers": ["std_metric", "combo_index"],
                },
            }
        },
        "notes": [
            "This executes a Nelder-Mead-like adaptive search path over single-split "
            "param_grid candidates."
        ],
    }
    assert evaluation["data"]["capabilities"][0]["supported_metrics"] == ["rollout_rmse"]
    assert compiled["data"]["compiled_request"]["model_key"] == "kbf"
    assert compiled["data"]["compiled_request"]["reference_profile"] == "kbf-regular-default"
    assert started["ok"] is True
    assert started["data"]["training_run"]["reference_profile"] == "kbf-regular-default"
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


def test_user_tools_compile_training_request_accepts_cv_overrides_and_surfaces_cv_artifacts(
    tmp_path, monkeypatch
) -> None:
    _configure_worker_bootstrap(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    tools = UserTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_dataset_handle = tools._context.facade.register_dataset_file(
        path=str(dataset_path)
    ).handle

    compiled = tools.compile_training_request(
        train_dataset_handle=train_dataset_handle,
        model_key="kbf",
        run_name="user_mode_cv",
        overrides={"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}}},
    )

    assert compiled["ok"] is True
    assert compiled["data"]["compiled_request"]["effective_config"]["cv"] == {
        "param_grid": {"model.koopman_dimension": [4, 6]}
    }

    started = tools.start_training_run(
        compiled_request_handle=compiled["data"]["summary"]["handle"],
        artifact_root=str(tmp_path / "outputs"),
    )
    for _ in range(100):
        trained = tools.describe_training_run(
            training_run_handle=started["data"]["summary"]["handle"]
        )
        assert trained["ok"] is True
        if trained["data"]["training_run"]["status"] == "SUCCEEDED":
            break
        if trained["data"]["training_run"]["status"] == "FAILED":
            raise AssertionError(trained["data"]["training_run"])
        time.sleep(0.05)
    else:
        raise AssertionError("training run did not finish")

    assert trained["ok"] is True
    assert Path(trained["data"]["training_run"]["artifacts"]["cv_results_path"]).is_file()
    assert Path(trained["data"]["training_run"]["artifacts"]["cv_plot_path"]).is_file()


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

        def tool(self, fn=None, **kwargs):
            del kwargs
            if fn is None:
                return self.tool
            self.tools[fn.__name__] = fn
            return fn

    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))

    from dymad.agent.mcp import build_server

    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        name="DyMAD User Tools Test",
    )

    assert "compile_training_request" in server.tools
    assert "start_training_run" in server.tools
    assert "describe_training_run" in server.tools
    assert "read_training_run_log" in server.tools
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

        def tool(self, fn=None, **kwargs):
            del kwargs
            if fn is None:
                return self.tool
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
