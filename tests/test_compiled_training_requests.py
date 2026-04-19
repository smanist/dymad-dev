from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from dymad.agent.compiler import TrainingRequest, compile_training_request
from dymad.agent.exec.context import build_default_context
from tests.test_mcp_train_eval_tools import _configure_worker_bootstrap


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


def test_executor_starts_training_run_from_compiled_request_handle(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch)
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
    result = second.executor.start_training_run(
        compiled_request_handle=compiled_summary.handle,
        artifact_root=str(outputs_root),
    )
    assert result.summary.kind == "training_run"
    assert result.training_run.reference_profile == "kbf-regular-default"

    for _ in range(100):
        described = second.executor.describe_training_run(training_run_handle=result.summary.handle)
        if described.training_run.status.value == "SUCCEEDED":
            break
        if described.training_run.status.value == "FAILED":
            raise AssertionError(described.training_run)
        time.sleep(0.05)
    else:
        raise AssertionError("training run did not finish")

    config_path = outputs_root / "compiled_run.yaml"
    materialized = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert described.training_run.checkpoint_handle is not None
    assert Path(described.training_run.artifacts["checkpoint_path"]).is_file()
    assert materialized["model"]["name"] == "compiled_run"
    assert materialized["model"]["koopman_dimension"] == 7
