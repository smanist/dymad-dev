from __future__ import annotations

import numpy as np
import numpy.testing as npt

from dymad.agent.exec.context import build_default_context
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


def test_persisted_dataset_run_and_evaluation_round_trip(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_path = tmp_path / "train.npz"
    np.savez_compressed(
        dataset_path,
        t=np.linspace(0.0, 1.0, 5),
        x=np.ones((2, 5, 2)),
        u=np.zeros((2, 5, 1)),
    )

    first = build_default_context(artifact_root=artifact_root)
    dataset = first.facade.register_dataset_file(path=str(dataset_path))
    checkpoint = first.facade.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path="checkpoints/kbf.pt",
    )
    run = first.facade.register_training_run(
        model_ref="dymad.models.collections:KBF",
        train_dataset_handle=dataset.handle,
        valid_dataset_handle=None,
        reference_profile="kbf-regular-default",
        checkpoint_handle=checkpoint.handle,
        artifact_root=str(tmp_path / "outputs"),
        run_name="kbf_run",
    )
    evaluation = first.facade.register_evaluation(
        checkpoint_handle=checkpoint.handle,
        test_dataset_handle=dataset.handle,
        metric="rollout_rmse",
        metrics_path=str(tmp_path / "metrics.json"),
        plot_paths=[str(tmp_path / "plot.png")],
    )

    second = build_default_context(artifact_root=artifact_root)
    dataset_record = second.facade.get_dataset(dataset.handle)
    run_record = second.facade.get_training_run(run.handle)
    evaluation_record = second.facade.get_evaluation(evaluation.handle)
    kinds = [summary.kind for summary in second.facade.list_objects()]

    assert dataset_record.path == str(dataset_path.resolve())
    assert run_record.reference_profile == "kbf-regular-default"
    assert run_record.checkpoint_handle == checkpoint.handle
    assert evaluation_record.metric == "rollout_rmse"
    assert evaluation_record.plot_paths == [str(tmp_path / "plot.png")]
    assert kinds == ["checkpoint", "dataset", "evaluation", "training_run"]


def test_persisted_prediction_result_round_trip(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_path = tmp_path / "test.npz"
    np.savez_compressed(
        dataset_path,
        t=np.linspace(0.0, 1.0, 5),
        x=np.ones((2, 5, 2)),
    )

    first = build_default_context(artifact_root=artifact_root)
    dataset = first.facade.register_dataset_file(path=str(dataset_path))
    checkpoint = first.facade.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path="checkpoints/kbf.pt",
    )
    prediction_dir = tmp_path / "prediction_result"
    prediction_dir.mkdir()
    predictions_path = prediction_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        truth=np.array([np.ones((5, 2))], dtype=object),
        predictions=np.array([np.zeros((5, 2))], dtype=object),
        times=np.array([np.linspace(0.0, 1.0, 5)], dtype=object),
        controls=np.array([None], dtype=object),
        parameters=np.array([None], dtype=object),
        edge_indices=np.array([None], dtype=object),
        edge_weights=np.array([None], dtype=object),
        edge_attrs=np.array([None], dtype=object),
        selected_indices=np.array([0]),
    )
    summary = first.facade.register_prediction_result(
        checkpoint_handle=checkpoint.handle,
        dataset_handle=dataset.handle,
        prediction_request_handle=None,
        artifact_dir=str(prediction_dir),
        predictions_path=str(predictions_path),
        dataset_kind="regular",
    )

    second = build_default_context(artifact_root=artifact_root)
    record = second.facade.get_prediction_result(summary.handle)
    described = second.facade.describe_object(summary.handle)

    assert record.checkpoint_handle == checkpoint.handle
    assert record.dataset_handle == dataset.handle
    assert record.dataset_kind == "regular"
    assert record.predictions_path == str(predictions_path)
    assert described.kind == "prediction_result"
    assert described.preview == f"regular @ {predictions_path}"


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
