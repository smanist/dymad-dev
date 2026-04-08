from __future__ import annotations

import numpy as np
import numpy.testing as npt

from dymad.exec.context import build_default_context
from dymad.sako.snapshot import build_spectral_snapshot


def test_persisted_handles_rehydrate_across_contexts(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    first = build_default_context(artifact_root=artifact_root)
    plan = first.executor.plan_checkpoint_prediction(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
        horizon=9,
        has_control=True,
        has_graph=False,
    )

    second = build_default_context(artifact_root=artifact_root)
    checkpoint = second.facade.get_checkpoint(plan.checkpoint_handle)
    request = second.facade.get_prediction_request(plan.prediction_handle)
    summaries = second.facade.list_objects()

    assert checkpoint.model_ref == "dymad.models.collections:LDM"
    assert checkpoint.checkpoint_path == "checkpoints/lti.pt"
    assert request.horizon == 9
    assert request.has_control is True
    assert [summary.handle for summary in summaries] == sorted(
        [plan.checkpoint_handle, plan.prediction_handle]
    )


def test_persisted_spectral_snapshots_round_trip(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    first = build_default_context(artifact_root=artifact_root)
    checkpoint = first.facade.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path="checkpoints/kbf.pt",
    )
    snapshot = build_spectral_snapshot(
        model_class="KBF",
        checkpoint_path="checkpoints/kbf.pt",
        encoded_p0=np.arange(12, dtype=float).reshape(4, 3),
        encoded_p1=np.arange(12, 24, dtype=float).reshape(4, 3),
        weights=(np.eye(3),),
        input_dim=2,
        obs_dim=3,
        metadata={"source": "roundtrip"},
    )
    summary = first.facade.register_spectral_snapshot(
        checkpoint_handle=checkpoint.handle,
        snapshot=snapshot,
    )

    second = build_default_context(artifact_root=artifact_root)
    record = second.facade.get_spectral_snapshot(summary.handle)
    described = second.facade.describe_object(summary.handle)

    assert record.checkpoint_handle == checkpoint.handle
    assert record.snapshot.model_class == "KBF"
    assert record.snapshot.metadata == {"source": "roundtrip"}
    assert described.preview == "samples=4, obs_dim=3"
    npt.assert_array_equal(record.snapshot.encoded_p0, snapshot.encoded_p0)
    npt.assert_array_equal(record.snapshot.encoded_p1, snapshot.encoded_p1)
    npt.assert_array_equal(
        record.snapshot.koopman_weights.full_matrix,
        snapshot.koopman_weights.full_matrix,
    )
