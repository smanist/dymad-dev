from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

import dymad.training
from dymad.agent.exec.context import build_default_context


def _write_regular_dataset(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 6)
    x = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
        ]
    )
    np.savez_compressed(path, t=t, x=x)


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
        torch.save({"config": config}, run_root / f"{run_name}.pt")
        np.savez_compressed(
            run_root / f"{run_name}_summary.npz",
            final_valid_loss=0.1,
            avg_epoch_time=0.2,
            best_valid_loss=np.array({"valid_total": 0.1}, dtype=object),
        )
        (run_root / f"{run_name}_history.png").write_bytes(b"history")


def _patch_fake_trainers(monkeypatch) -> None:
    monkeypatch.setattr(dymad.training, "WeakFormTrainer", _FakeTrainer)
    monkeypatch.setattr(dymad.training, "NODETrainer", _FakeTrainer)
    monkeypatch.setattr(dymad.training, "LinearTrainer", _FakeTrainer)
    monkeypatch.setattr(dymad.training, "StackedTrainer", _FakeTrainer)


def test_executor_lists_model_families_and_describes_kbf(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")

    families = context.executor.list_model_families()
    family = context.executor.describe_model_family(model_ref="dymad.models.collections:KBF")

    assert any(item.model_ref == "dymad.models.collections:DGKMSK" for item in families)
    assert family.name == "KBF"
    assert family.expects_graph_data is False
    assert family.default_predictor == "continuous"


def test_executor_dataset_compatibility_reports_graph_mismatch(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset = context.facade.register_dataset_file(path=str(dataset_path))

    compatibility = context.executor.validate_dataset_compatibility(
        dataset_handle=dataset.handle,
        model_ref="dymad.models.collections:GKBF",
    )

    assert compatibility.is_compatible is False
    assert compatibility.expected_dataset_kind == "graph"


def test_executor_validation_rejects_reserved_paths_without_raising(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset = context.facade.register_dataset_file(path=str(dataset_path))

    validation = context.executor.validate_training_config(
        train_dataset_handle=dataset.handle,
        model_ref="dymad.models.collections:KBF",
        config={"data": {"path": "/tmp/override.npz"}},
    )

    assert validation.is_valid is False
    assert validation.rejection_reason is not None
    assert "reserved" in validation.rejection_reason


def test_executor_materialize_and_artifact_listing_use_standard_paths(
    tmp_path, monkeypatch
) -> None:
    _patch_fake_trainers(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset = context.facade.register_dataset_file(path=str(dataset_path))

    materialized = context.executor.materialize_training_config(
        train_dataset_handle=dataset.handle,
        model_ref="dymad.models.collections:KBF",
        artifact_root=str(tmp_path / "outputs"),
        run_name="phase1_case",
    )
    trained = context.executor.train_model(
        train_dataset_handle=dataset.handle,
        model_ref="dymad.models.collections:KBF",
        artifact_root=str(tmp_path / "outputs"),
        run_name="phase1_case",
    )

    artifacts = context.executor.list_training_artifacts(run_handle=trained.run_summary.handle)

    assert Path(materialized.config_path).is_file()
    assert materialized.trainer_kind == "weak_form"
    assert artifacts.paths["config_path"] == materialized.config_path
    assert artifacts.exists["checkpoint_path"] is True
    assert artifacts.exists["training_summary_path"] is True
