from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

import dymad.io
import dymad.training
from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import DemoTools


def _write_regular_dataset(path: Path, *, with_control: bool = True) -> None:
    t = np.linspace(0.0, 1.0, 6)
    x = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [1.2, 0.0], [1.6, 0.0], [2.0, 0.0]],
        ]
    )
    payload = {"t": t, "x": x}
    if with_control:
        payload["u"] = np.ones((3, 6, 1)) * 0.1
    np.savez_compressed(path, **payload)


def _write_graph_dataset(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 5)
    x = np.array(
        [
            [[0.0, 0.0, 0.1, 0.1], [0.1, 0.0, 0.2, 0.1], [0.2, 0.0, 0.3, 0.1], [0.3, 0.0, 0.4, 0.1], [0.4, 0.0, 0.5, 0.1]],
            [[0.0, 0.0, 0.2, 0.2], [0.2, 0.0, 0.3, 0.2], [0.4, 0.0, 0.4, 0.2], [0.6, 0.0, 0.5, 0.2], [0.8, 0.0, 0.6, 0.2]],
        ]
    )
    adj = np.array([[0, 1], [1, 0]])
    np.savez_compressed(path, t=t, x=x, adj=adj)


class _FakeTrainer:
    def __init__(self, config_path, model_class, config_mod=None, device=None, max_workers=1):
        del model_class, config_mod, max_workers
        self.config_path = Path(config_path)
        self.device = torch.device("cpu") if device is None else device

    def train(self):
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        run_name = config["model"]["name"]
        run_root = self.config_path.parent / run_name
        run_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_root / f"{run_name}.pt"
        torch.save(
            {
                "config": config,
                "train_md": {
                    "n_state_features": 2,
                    "n_aux_features": 0,
                    "n_control_features": 1,
                    "n_parameters": 0,
                    "transform_x_state": None,
                    "transform_u_state": None,
                },
                "valid_md": {
                    "n_state_features": 2,
                    "n_aux_features": 0,
                    "n_control_features": 1,
                    "n_parameters": 0,
                    "transform_x_state": None,
                    "transform_u_state": None,
                },
                "model_state_dict": {},
                "best_loss": {"valid_total": 0.1},
                "hist": [{"train_total": [0.2], "valid_total": [0.1]}],
                "crit": [],
                "epoch_times": [0.5],
                "converged": False,
                "device": "cpu",
                "epoch": 1,
            },
            checkpoint_path,
        )
        np.savez_compressed(
            run_root / f"{run_name}_summary.npz",
            model_name=run_name,
            total_training_time=1.0,
            avg_epoch_time=0.5,
            final_train_loss=0.2,
            final_valid_loss=0.1,
            best_valid_loss=np.array({"valid_total": 0.1}, dtype=object),
            convergence_epoch=1,
            hist=np.array([{"train_total": [0.2], "valid_total": [0.1]}], dtype=object),
            crit_name=None,
            crit_epoch=np.array([]),
            crits=np.array([]),
        )
        (run_root / f"{run_name}_history.png").write_bytes(b"history")
        if config.get("plotting", {}).get("prediction", True):
            (run_root / f"{run_name}_prediction.png").write_bytes(b"prediction")


def _patch_fake_trainers(monkeypatch) -> None:
    monkeypatch.setattr(dymad.training, "WeakFormTrainer", _FakeTrainer)
    monkeypatch.setattr(dymad.training, "NODETrainer", _FakeTrainer)
    monkeypatch.setattr(dymad.training, "LinearTrainer", _FakeTrainer)
    monkeypatch.setattr(dymad.training, "StackedTrainer", _FakeTrainer)


def test_register_dataset_file_and_inspect_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))

    registered = tools.register_dataset_file(path=str(dataset_path))
    dataset_handle = registered["data"]["summary"]["handle"]
    inspected = tools.inspect_dataset(dataset_handle=dataset_handle)

    assert registered["ok"] is True
    assert registered["data"]["dataset"]["kind"] == "regular"
    assert inspected["ok"] is True
    assert inspected["data"]["inspection"]["n_trajectories"] == 3
    assert inspected["data"]["inspection"]["state_dim"] == 2
    assert inspected["data"]["inspection"]["control_dim"] == 1
    assert inspected["data"]["inspection"]["has_graph"] is False


def test_inspect_dataset_without_control_inputs(tmp_path) -> None:
    dataset_path = tmp_path / "train_no_u.npz"
    _write_regular_dataset(dataset_path, with_control=False)
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))

    registered = tools.register_dataset_file(path=str(dataset_path))
    inspected = tools.inspect_dataset(dataset_handle=registered["data"]["summary"]["handle"])

    assert inspected["data"]["inspection"]["control_dim"] == 0


def test_train_model_infers_profile_and_persists_run(tmp_path, monkeypatch) -> None:
    _patch_fake_trainers(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_handle = tools.register_dataset_file(path=str(dataset_path))["data"]["summary"]["handle"]

    response = tools.train_model(
        train_dataset_handle=train_handle,
        model_ref="dymad.models.collections:KBF",
        artifact_root=str(tmp_path / "outputs"),
        run_name="kbf_case",
        config={"model": {"koopman_dimension": 8}},
    )

    result = response["data"]["result"]
    assert response["ok"] is True
    assert result["reference_profile"] == "kbf-regular-default"
    assert result["trainer_kind"] == "weak_form"
    assert result["run_summary"]["kind"] == "training_run"
    assert result["checkpoint_summary"]["kind"] == "checkpoint"
    assert Path(result["artifacts"]["checkpoint_path"]).is_file()
    assert Path(result["artifacts"]["training_summary_path"]).is_file()


def test_train_model_replaces_phases_and_selects_stacked(tmp_path, monkeypatch) -> None:
    _patch_fake_trainers(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_handle = tools.register_dataset_file(path=str(dataset_path))["data"]["summary"]["handle"]

    response = tools.train_model(
        train_dataset_handle=train_handle,
        model_ref="dymad.models.collections:KBF",
        artifact_root=str(tmp_path / "outputs"),
        run_name="stacked_case",
        config={
            "phases": [
                {"type": "optimizer", "name": "Warmup", "trainer": "Weak", "n_epochs": 5},
                {"type": "optimizer", "name": "Refine", "trainer": "NODE", "n_epochs": 7},
            ]
        },
    )

    config_path = tmp_path / "outputs" / "stacked_case.yaml"
    materialized = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert response["ok"] is True
    assert response["data"]["result"]["trainer_kind"] == "stacked"
    assert len(materialized["phases"]) == 2
    assert [phase["name"] for phase in materialized["phases"]] == ["Warmup", "Refine"]


def test_train_model_rejects_reserved_runtime_paths(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_handle = tools.register_dataset_file(path=str(dataset_path))["data"]["summary"]["handle"]

    response = tools.train_model(
        train_dataset_handle=train_handle,
        model_ref="dymad.models.collections:KBF",
        artifact_root=str(tmp_path / "outputs"),
        config={"data": {"path": "/tmp/override.npz"}},
    )

    assert response["ok"] is False
    assert "reserved" in response["error"]["message"]


def test_train_model_rejects_run_name_with_path_separators(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    train_handle = tools.register_dataset_file(path=str(dataset_path))["data"]["summary"]["handle"]

    response = tools.train_model(
        train_dataset_handle=train_handle,
        model_ref="dymad.models.collections:KBF",
        artifact_root=str(tmp_path / "outputs"),
        run_name="../escape",
    )

    assert response["ok"] is False
    assert "run_name" in response["error"]["message"]


def test_evaluate_model_regular_writes_metrics_and_plot(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "test.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    def fake_load_model(model, checkpoint_path, *, context=None, **kwargs):
        del model, checkpoint_path, context, kwargs

        def predict_fn(x0, t, u=None, p=None, **kwargs):
            del t, u, p, kwargs
            return 0.5 * x0

        return object(), predict_fn

    def fake_plot_trajectory(traj, ts, model_name=None, prefix=".", **kwargs):
        del traj, ts, kwargs
        path = Path(prefix) / f"{model_name}_prediction.png"
        path.write_bytes(b"plot")

    monkeypatch.setattr(dymad.io, "load_model", fake_load_model)
    monkeypatch.setattr("dymad.agent.exec.workflow.plot_trajectory", fake_plot_trajectory)

    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    dataset_handle = tools.register_dataset_file(path=str(dataset_path))["data"]["summary"]["handle"]
    checkpoint_summary = tools.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path=str(tmp_path / "fake.pt"),
    )

    response = tools.evaluate_model(
        checkpoint_handle=checkpoint_summary["data"]["summary"]["handle"],
        test_dataset_handle=dataset_handle,
        metric="rollout_rmse",
        artifact_root=str(tmp_path / "evals"),
        plot_selection="median",
        max_plots=1,
    )

    result = response["data"]["result"]
    assert response["ok"] is True
    assert Path(result["artifacts"]["metrics_path"]).is_file()
    assert len(result["artifacts"]["plot_paths"]) == 1
    metrics = json.loads(Path(result["artifacts"]["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["metric"] == "rollout_rmse"
    assert result["evaluation_summary"]["kind"] == "evaluation"


def test_evaluate_model_passes_active_context_to_loader(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "test.npz"
    _write_regular_dataset(dataset_path, with_control=False)
    captured: dict[str, object] = {}

    def fake_load_model(model, checkpoint_path, *, context=None, **kwargs):
        del model, checkpoint_path, kwargs
        captured["context"] = context

        def predict_fn(x0, t, u=None, p=None, **predict_kwargs):
            del t, u, p, predict_kwargs
            return np.asarray(x0)

        return object(), predict_fn

    monkeypatch.setattr(dymad.io, "load_model", fake_load_model)

    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    dataset_handle = tools.register_dataset_file(path=str(dataset_path))["data"]["summary"]["handle"]
    checkpoint_summary = tools.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path=str(tmp_path / "fake.pt"),
    )

    response = tools.evaluate_model(
        checkpoint_handle=checkpoint_summary["data"]["summary"]["handle"],
        test_dataset_handle=dataset_handle,
        metric="rollout_rmse",
        artifact_root=str(tmp_path / "evals"),
    )

    assert response["ok"] is True
    assert captured["context"] is tools.context


def test_evaluate_model_graph_skips_plot(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "graph_test.npz"
    _write_graph_dataset(dataset_path)

    def fake_load_model(model, checkpoint_path, *, context=None, **kwargs):
        del model, checkpoint_path, context, kwargs

        def predict_fn(x0, t, u=None, p=None, ei=None, ew=None, ea=None, **kwargs):
            del t, u, p, ei, ew, ea, kwargs
            return np.asarray(x0)

        return object(), predict_fn

    monkeypatch.setattr(dymad.io, "load_model", fake_load_model)

    tools = DemoTools(context=build_default_context(artifact_root=tmp_path / "artifacts"))
    dataset_handle = tools.register_dataset_file(path=str(dataset_path), kind="graph")["data"]["summary"]["handle"]
    checkpoint_summary = tools.register_checkpoint(
        model_ref="dymad.models.collections:GKBF",
        checkpoint_path=str(tmp_path / "graph_fake.pt"),
    )

    response = tools.evaluate_model(
        checkpoint_handle=checkpoint_summary["data"]["summary"]["handle"],
        test_dataset_handle=dataset_handle,
        metric="rollout_rmse",
        artifact_root=str(tmp_path / "evals"),
    )

    result = response["data"]["result"]
    assert response["ok"] is True
    assert result["artifacts"]["plot_paths"] == []
    assert result["plot_skipped_reason"] == "graph plotting unsupported in v1"
