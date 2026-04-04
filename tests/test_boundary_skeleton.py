import pytest
import numpy as np

from dymad.exec.context import build_default_context
from dymad.facade.handles import (
    CheckpointHandle,
    HandleValidationError,
    PredictionHandle,
    SpectralSnapshotHandle,
)
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
