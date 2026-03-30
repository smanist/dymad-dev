import pytest

from dymad.exec.context import build_default_context
from dymad.facade.handles import CheckpointHandle, HandleValidationError, PredictionHandle


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
