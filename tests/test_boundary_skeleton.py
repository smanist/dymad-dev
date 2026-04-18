import numpy as np
import pytest

from dymad.agent.exec.context import build_default_context, resolve_artifact_root
from dymad.agent.facade.handles import (
    CheckpointHandle,
    DatasetHandle,
    EvaluationHandle,
    HandleValidationError,
    PredictionHandle,
    SpectralSnapshotHandle,
    TrainingRunHandle,
)
from dymad.sako.adapter import SpectralEigensystem
from dymad.sako.snapshot import build_spectral_snapshot


def test_checkpoint_prediction_handle_flow() -> None:
    context = build_default_context()

    plan = context.executor.plan_checkpoint_prediction(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
        horizon=12,
        has_control=True,
    )

    checkpoint_handle = CheckpointHandle.parse(plan.checkpoint_handle)
    prediction_handle = PredictionHandle.parse(plan.prediction_handle)

    checkpoint_summary = context.facade.describe_object(checkpoint_handle.value)
    prediction_summary = context.facade.describe_object(prediction_handle.value)
    request = context.facade.get_prediction_request(prediction_handle.value)

    assert plan.entrypoint == "dymad.io.checkpoint.load_model"
    assert checkpoint_summary.kind == "checkpoint"
    assert prediction_summary.kind == "prediction_request"
    assert prediction_summary.derived_from == checkpoint_handle.value
    assert request.horizon == 12
    assert request.has_control is True
    assert request.has_graph is False


def test_handles_reject_invalid_shapes() -> None:
    with pytest.raises(HandleValidationError):
        CheckpointHandle.parse("bad")

    with pytest.raises(HandleValidationError):
        DatasetHandle.parse("not_a_dataset")

    with pytest.raises(HandleValidationError):
        TrainingRunHandle.parse("not_a_run")

    with pytest.raises(HandleValidationError):
        EvaluationHandle.parse("not_an_eval")

    with pytest.raises(HandleValidationError):
        PredictionHandle.parse("not_a_prediction")

    with pytest.raises(HandleValidationError):
        SpectralSnapshotHandle.parse("not_a_spectral_snapshot")


def test_spectral_snapshot_handle_flow() -> None:
    context = build_default_context()
    checkpoint = context.facade.register_checkpoint(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
    )
    snapshot = build_spectral_snapshot(
        model_class="LDM",
        checkpoint_path="checkpoints/lti.pt",
        encoded_p0=np.ones((6, 3)),
        encoded_p1=np.zeros((6, 3)),
        weights=(np.eye(3),),
        input_dim=2,
        obs_dim=3,
        metadata={"source": "boundary-test"},
    )

    summary = context.facade.register_spectral_snapshot(
        checkpoint_handle=checkpoint.handle,
        snapshot=snapshot,
    )
    handle = SpectralSnapshotHandle.parse(summary.handle)
    record = context.facade.get_spectral_snapshot(handle.value)
    described = context.facade.describe_object(handle.value)

    assert summary.kind == "spectral_snapshot"
    assert summary.derived_from == checkpoint.handle
    assert record.checkpoint_handle == checkpoint.handle
    assert record.snapshot.sample_count == 6
    assert record.snapshot.koopman_weights.mode == "full"
    assert described.preview == "samples=6, obs_dim=3"


def test_spectral_exec_flow_resolves_snapshot_handle() -> None:
    context = build_default_context()
    events: list[str] = []

    original_register_checkpoint = context.facade.register_checkpoint
    original_register_snapshot = context.facade.register_spectral_snapshot
    original_get_snapshot = context.facade.get_spectral_snapshot

    def traced_register_checkpoint(*, model_ref: str, checkpoint_path: str, device: str = "cpu"):
        events.append("facade.register_checkpoint")
        return original_register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def traced_register_snapshot(*, checkpoint_handle: str, snapshot):
        events.append("facade.register_spectral_snapshot")
        return original_register_snapshot(
            checkpoint_handle=checkpoint_handle,
            snapshot=snapshot,
        )

    def traced_get_snapshot(handle: str):
        events.append("facade.get_spectral_snapshot")
        return original_get_snapshot(handle)

    context.facade.register_checkpoint = traced_register_checkpoint
    context.facade.register_spectral_snapshot = traced_register_snapshot
    context.facade.get_spectral_snapshot = traced_get_snapshot

    snapshot = build_spectral_snapshot(
        model_class="LDM",
        checkpoint_path="checkpoints/lti.pt",
        encoded_p0=np.ones((6, 3)),
        encoded_p1=np.zeros((6, 3)),
        weights=(np.eye(3),),
        input_dim=2,
        obs_dim=3,
    )
    plan = context.executor.plan_spectral_analysis(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
        snapshot=snapshot,
    )
    handle = SpectralSnapshotHandle.parse(plan.spectral_snapshot_handle)
    eigensystem = SpectralEigensystem(
        discrete_eigs=np.array([1.0 + 0j]),
        left_eigvecs=np.array([[1.0 + 0j]]),
        right_eigvecs=np.array([[1.0 + 0j]]),
        projector=np.array([[1.0 + 0j]]),
        dt=1.0,
    )
    adapter = context.executor.materialize_spectral_adapter(
        plan=plan,
        eigensystem=eigensystem,
    )

    assert plan.entrypoint == "dymad.sako.SpectralAnalysis"
    assert handle.value == plan.spectral_snapshot_handle
    assert adapter.snapshot.sample_count == 6
    assert events == [
        "facade.register_checkpoint",
        "facade.register_spectral_snapshot",
        "facade.get_spectral_snapshot",
    ]


def test_build_default_context_does_not_create_default_artifact_dir_until_needed(
    monkeypatch, tmp_path
) -> None:
    repo_root = tmp_path / "workspace"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    monkeypatch.setenv("PWD", str(repo_root))
    monkeypatch.chdir("/")

    expected_root = repo_root / ".dymad" / "artifacts"
    context = build_default_context()

    assert context.artifact_store is None
    assert not expected_root.exists()

    context.facade.register_checkpoint(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path=str(repo_root / "checkpoints" / "lti.pt"),
    )

    assert expected_root.is_dir()
    assert (expected_root / "checkpoints").is_dir()


def test_resolve_artifact_root_prefers_repo_root_for_anchor_path(tmp_path) -> None:
    repo_root = tmp_path / "mcp_run"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    nested = repo_root / "data"
    nested.mkdir()
    dataset_path = nested / "train.npz"

    resolved = resolve_artifact_root(anchor_path=dataset_path)

    assert resolved == repo_root / ".dymad" / "artifacts"
