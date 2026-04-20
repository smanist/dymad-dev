from __future__ import annotations

import numpy as np
import numpy.testing as npt

from dymad.agent.compiler import TrainingRequest, compile_training_request
from dymad.agent.exec.context import build_default_context
from dymad.agent.store.object_store import TrainingRunStatus
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
    compiled = compile_training_request(
        facade=first.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="kbf_run",
        ),
    )
    compiled_summary = first.facade.register_compiled_training_request(compiled_request=compiled)
    checkpoint = first.facade.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path="checkpoints/kbf.pt",
    )
    run = first.facade.register_training_run(
        compiled_request_handle=compiled_summary.handle,
        status=TrainingRunStatus.SUCCEEDED,
        created_at="2026-04-18T00:00:00+00:00",
        started_at="2026-04-18T00:00:01+00:00",
        finished_at="2026-04-18T00:00:02+00:00",
        pid=1234,
        log_path=str(tmp_path / "outputs" / "kbf_run" / "training.log"),
        config_path=str(tmp_path / "outputs" / "kbf_run.yaml"),
        run_root=str(tmp_path / "outputs" / "kbf_run"),
        model_ref="dymad.models.collections:KBF",
        train_dataset_handle=dataset.handle,
        valid_dataset_handle=None,
        reference_profile="kbf-regular-default",
        checkpoint_handle=checkpoint.handle,
        artifact_root=str(tmp_path / "outputs"),
        run_name="kbf_run",
        artifacts={
            "checkpoint_path": str(tmp_path / "outputs" / "kbf_run" / "kbf_run.pt"),
            "training_summary_path": str(tmp_path / "outputs" / "kbf_run" / "kbf_run_summary.npz"),
        },
        metrics={"final_valid_loss": 0.1},
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
    assert run_record.compiled_request_handle == compiled_summary.handle
    assert run_record.status is TrainingRunStatus.SUCCEEDED
    assert run_record.reference_profile == "kbf-regular-default"
    assert run_record.checkpoint_handle == checkpoint.handle
    assert run_record.metrics == {"final_valid_loss": 0.1}
    assert evaluation_record.metric == "rollout_rmse"
    assert evaluation_record.plot_paths == [str(tmp_path / "plot.png")]
    assert kinds == [
        "checkpoint",
        "dataset",
        "evaluation",
        "training_run",
        "compiled_training_request",
    ]
    assert second.facade.describe_object(run.handle).derived_from == compiled_summary.handle
    assert second.facade.describe_object(run.handle).preview.startswith("SUCCEEDED:")


def test_persistence_recreates_missing_kind_directories_on_write(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_path = tmp_path / "train.npz"
    np.savez_compressed(
        dataset_path,
        t=np.linspace(0.0, 1.0, 5),
        x=np.ones((2, 5, 2)),
        u=np.zeros((2, 5, 1)),
    )

    context = build_default_context(artifact_root=artifact_root)
    context.facade.register_checkpoint(
        model_ref="dymad.models.collections:KBF",
        checkpoint_path="checkpoints/bootstrap.pt",
    )
    datasets_dir = artifact_root / "datasets"
    compiled_dir = artifact_root / "compiled_training_requests"
    datasets_dir.rmdir()
    compiled_dir.rmdir()

    dataset = context.facade.register_dataset_file(path=str(dataset_path))
    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="recreate_dirs",
        ),
    )
    summary = context.facade.register_compiled_training_request(compiled_request=compiled)

    assert (datasets_dir / f"{dataset.handle}.json").is_file()
    assert (compiled_dir / f"{summary.handle}.json").is_file()


def test_training_run_updates_merge_latest_persisted_state(tmp_path) -> None:
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
    compiled = compile_training_request(
        facade=first.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="merge_run",
        ),
    )
    compiled_summary = first.facade.register_compiled_training_request(compiled_request=compiled)
    run = first.facade.register_training_run(
        compiled_request_handle=compiled_summary.handle,
        status=TrainingRunStatus.QUEUED,
        created_at="2026-04-18T00:00:00+00:00",
        model_ref=compiled.model_ref,
        train_dataset_handle=dataset.handle,
        valid_dataset_handle=None,
        reference_profile=compiled.profile.key,
        checkpoint_handle=None,
        artifact_root=str(tmp_path / "outputs"),
        run_name="merge_run",
        artifacts={},
        metrics={},
    )
    checkpoint = first.facade.register_checkpoint(
        model_ref=compiled.model_ref,
        checkpoint_path=str(tmp_path / "outputs" / "merge_run" / "merge_run.pt"),
    )

    second = build_default_context(artifact_root=artifact_root)
    second.facade.update_training_run(
        run.handle,
        status=TrainingRunStatus.SUCCEEDED,
        finished_at="2026-04-18T00:00:03+00:00",
        checkpoint_handle=checkpoint.handle,
        metrics={"final_valid_loss": 0.1},
    )

    first.facade.update_training_run(
        run.handle,
        pid=4321,
        log_path=str(tmp_path / "outputs" / "merge_run" / "training.log"),
    )

    reloaded = build_default_context(artifact_root=artifact_root).facade.get_training_run(
        run.handle
    )

    assert reloaded.status is TrainingRunStatus.SUCCEEDED
    assert reloaded.checkpoint_handle == checkpoint.handle
    assert reloaded.metrics == {"final_valid_loss": 0.1}
    assert reloaded.pid == 4321


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
