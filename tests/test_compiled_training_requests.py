from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

import dymad.training
from dymad.agent.compiler import TrainingRequest, compile_training_request
from dymad.agent.exec.context import build_default_context


def _write_regular_dataset(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 6)
    x = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [1.2, 0.0], [1.6, 0.0], [2.0, 0.0]],
        ]
    )
    payload = {"t": t, "x": x, "u": np.ones((3, 6, 1)) * 0.1}
    np.savez_compressed(path, **payload)


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


def test_compiled_training_request_persists_and_rehydrates(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)

    first = build_default_context(artifact_root=artifact_root)
    dataset = first.facade.register_dataset_file(path=str(dataset_path))
    compiled = compile_training_request(
        facade=first.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="persisted_compile",
            overrides={"model": {"koopman_dimension": 8}},
            seed=123,
            device="cpu",
            max_workers=3,
        ),
    )
    summary = first.facade.register_compiled_training_request(compiled_request=compiled)

    second = build_default_context(artifact_root=artifact_root)
    record = second.facade.get_compiled_training_request(summary.handle)
    described = second.facade.describe_object(summary.handle)
    listed = second.facade.list_objects(kind="compiled_training_request")

    assert record.model_key == "kbf"
    assert record.model_ref == "dymad.models.collections:KBF"
    assert record.reference_profile == "kbf-regular-default"
    assert record.effective_run_name == "persisted_compile"
    assert record.effective_config["model"]["koopman_dimension"] == 8
    assert record.seed == 123
    assert record.device == "cpu"
    assert record.max_workers == 3
    assert described.kind == "compiled_training_request"
    assert described.derived_from == dataset.handle
    assert "kbf/regular" in described.preview
    assert [item.handle for item in listed] == [summary.handle]


def test_executor_trains_from_compiled_request_handle(tmp_path, monkeypatch) -> None:
    _patch_fake_trainers(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    outputs_root = tmp_path / "outputs"
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)

    first = build_default_context(artifact_root=artifact_root)
    dataset = first.facade.register_dataset_file(path=str(dataset_path))
    compiled = compile_training_request(
        facade=first.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="compiled_run",
            overrides={"model": {"koopman_dimension": 7}},
            seed=99,
        ),
    )
    compiled_summary = first.facade.register_compiled_training_request(compiled_request=compiled)

    second = build_default_context(artifact_root=artifact_root)
    result = second.executor.train_compiled_request(
        compiled_request_handle=compiled_summary.handle,
        artifact_root=str(outputs_root),
    )

    config_path = outputs_root / "compiled_run.yaml"
    materialized = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert result.reference_profile == "kbf-regular-default"
    assert result.trainer_kind == "weak_form"
    assert result.run_summary.kind == "training_run"
    assert result.checkpoint_summary.kind == "checkpoint"
    assert Path(result.artifacts["checkpoint_path"]).is_file()
    assert materialized["model"]["name"] == "compiled_run"
    assert materialized["model"]["koopman_dimension"] == 7
