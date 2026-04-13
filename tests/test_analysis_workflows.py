from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dymad.agent.compiler import AnalysisRequest, compile_analysis_request
from dymad.agent.exec.context import build_default_context
from dymad.agent.registry import list_analysis_capabilities, list_evaluation_capabilities


def _write_regular_dataset(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 6)
    x = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
        ]
    )
    np.savez_compressed(path, t=t, x=x)


def test_analysis_registry_lists_supported_capabilities() -> None:
    capabilities = {capability.key: capability for capability in list_analysis_capabilities()}

    assert set(capabilities) == {"spectral_koopman", "vortex_transform_modes"}
    assert capabilities["spectral_koopman"].requires_checkpoint is True
    assert capabilities["vortex_transform_modes"].dataset_input_keys == (
        "train_dataset_handle",
        "test_dataset_handle",
    )


def test_evaluation_registry_lists_supported_capabilities() -> None:
    capabilities = {capability.key: capability for capability in list_evaluation_capabilities()}

    assert set(capabilities) == {"checkpoint_rollout"}
    assert capabilities["checkpoint_rollout"].supported_metrics == ("rollout_rmse",)
    assert capabilities["checkpoint_rollout"].parameter_schema["metric"]["enum"] == ["rollout_rmse"]


def test_compile_and_persist_analysis_request_round_trip(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=artifact_root)
    dataset_summary = context.facade.register_dataset_file(path=str(dataset_path))

    compiled = compile_analysis_request(
        request=AnalysisRequest(
            workflow_key="vortex_transform_modes",
            dataset_handles={
                "train_dataset_handle": dataset_summary.handle,
                "test_dataset_handle": dataset_summary.handle,
            },
            parameters={"config_path": "scripts/vortex/vor_model.yaml", "index": 1},
        )
    )
    summary = context.facade.register_compiled_analysis_request(compiled_request=compiled)

    second = build_default_context(artifact_root=artifact_root)
    record = second.facade.get_compiled_analysis_request(summary.handle)

    assert summary.kind == "compiled_analysis_request"
    assert record.workflow_key == "vortex_transform_modes"
    assert record.dataset_handles["train_dataset_handle"] == dataset_summary.handle
    assert record.parameters["index"] == 1


def test_run_analysis_request_executes_spectral_path(tmp_path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    outputs_root = tmp_path / "outputs"
    context = build_default_context(artifact_root=artifact_root)
    checkpoint = context.facade.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path="checkpoints/fake.pt",
    )
    compiled = compile_analysis_request(
        request=AnalysisRequest(
            workflow_key="spectral_koopman",
            checkpoint_handle=checkpoint.handle,
            parameters={"dt": 0.2},
        )
    )
    compiled_summary = context.facade.register_compiled_analysis_request(compiled_request=compiled)

    class _FakeSnapshot:
        obs_dim = 4
        sample_count = 12

    class _FakeCtx:
        snapshot = _FakeSnapshot()

    class FakeSpectralAnalysis:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._wd = np.array([1.0 + 0.0j, 0.9 + 0.1j])
            self._wd_full = np.array([1.0 + 0.0j, 0.9 + 0.1j, 0.8 - 0.2j])
            self._ctx = _FakeCtx()

    monkeypatch.setattr("dymad.sako.base.SpectralAnalysis", FakeSpectralAnalysis)

    result = context.executor.run_analysis_request(
        compiled_request_handle=compiled_summary.handle,
        artifact_root=str(outputs_root),
    )

    summary = json.loads(Path(result.artifacts["summary_path"]).read_text(encoding="utf-8"))
    assert result.workflow_key == "spectral_koopman"
    assert summary["n_eigs"] == 2
    assert summary["obs_dim"] == 4


def test_run_analysis_request_executes_vortex_mode_path(tmp_path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    outputs_root = tmp_path / "outputs"
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=artifact_root)
    dataset_summary = context.facade.register_dataset_file(path=str(dataset_path))
    compiled = compile_analysis_request(
        request=AnalysisRequest(
            workflow_key="vortex_transform_modes",
            dataset_handles={
                "train_dataset_handle": dataset_summary.handle,
                "test_dataset_handle": dataset_summary.handle,
            },
            parameters={
                "config_path": "scripts/vortex/vor_model.yaml",
                "index": 1,
                "nx": 1,
                "ny": 2,
            },
        )
    )
    compiled_summary = context.facade.register_compiled_analysis_request(compiled_request=compiled)

    def fake_compute(**kwargs):
        del kwargs
        return {
            "index": 1,
            "ref": np.ones((1, 1, 2)),
            "rel_dx_error": 0.01,
            "rel_dz_error": 0.02,
        }

    monkeypatch.setattr(
        "dymad.agent.exec.vortex_analysis.compute_vortex_mode_analysis", fake_compute
    )

    result = context.executor.run_analysis_request(
        compiled_request_handle=compiled_summary.handle,
        artifact_root=str(outputs_root),
    )

    summary = json.loads(Path(result.artifacts["summary_path"]).read_text(encoding="utf-8"))
    assert result.workflow_key == "vortex_transform_modes"
    assert summary["rel_dx_error"] == 0.01
    assert summary["ny"] == 2
