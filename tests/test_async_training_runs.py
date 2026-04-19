from __future__ import annotations

import time

from dymad.agent.compiler import TrainingRequest, compile_training_request
from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import DemoTools, UserTools
from dymad.agent.store.object_store import TrainingRunStatus
from tests.test_mcp_train_eval_tools import _configure_worker_bootstrap, _write_regular_dataset


def _poll_user_training_run(
    tools: UserTools, handle: str, *, timeout: float = 5.0
) -> dict[str, object]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = tools.describe_training_run(training_run_handle=handle)
        assert response["ok"] is True
        status = response["data"]["training_run"]["status"]
        if status in {"SUCCEEDED", "FAILED"}:
            return response
        time.sleep(0.05)
    raise AssertionError(f"training run {handle} did not finish")


def test_worker_failure_persists_structured_error_metadata(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch, mode="fail")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    context = build_default_context(artifact_root=tmp_path / "artifacts")
    tools = UserTools(context=context)
    train_dataset_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle
    compiled = tools.compile_training_request(
        train_dataset_handle=train_dataset_handle,
        model_key="kbf",
        run_name="failing_run",
    )

    started = tools.start_training_run(
        compiled_request_handle=compiled["data"]["summary"]["handle"],
        artifact_root=str(tmp_path / "outputs"),
    )
    polled = _poll_user_training_run(tools, started["data"]["summary"]["handle"])

    assert polled["data"]["training_run"]["status"] == "FAILED"
    assert polled["data"]["training_run"]["error_type"] == "RuntimeError"
    assert "simulated training failure" in polled["data"]["training_run"]["error_message"]
    assert polled["data"]["training_run"]["checkpoint_handle"] is None


def test_read_training_run_log_supports_offsets_and_eof(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    context = build_default_context(artifact_root=tmp_path / "artifacts")
    tools = UserTools(context=context)
    train_dataset_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle
    compiled = tools.compile_training_request(
        train_dataset_handle=train_dataset_handle,
        model_key="kbf",
        run_name="log_run",
    )

    started = tools.start_training_run(
        compiled_request_handle=compiled["data"]["summary"]["handle"],
        artifact_root=str(tmp_path / "outputs"),
    )
    handle = started["data"]["summary"]["handle"]
    polled = _poll_user_training_run(tools, handle)
    assert polled["data"]["training_run"]["status"] == "SUCCEEDED"

    first = tools.read_training_run_log(
        training_run_handle=handle,
        offset=0,
        max_bytes=12,
    )
    second = tools.read_training_run_log(
        training_run_handle=handle,
        offset=first["data"]["next_offset"],
        max_bytes=65536,
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert "fake trainer" in first["data"]["text"] + second["data"]["text"]
    assert first["data"]["next_offset"] > 0
    assert first["data"]["eof"] is False
    assert second["data"]["eof"] is True


def test_describe_training_run_reconciles_stale_running_pid(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset = context.facade.register_dataset_file(path=str(dataset_path))
    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="stale_run",
        ),
    )
    compiled_summary = context.facade.register_compiled_training_request(compiled_request=compiled)
    run = context.facade.register_training_run(
        compiled_request_handle=compiled_summary.handle,
        status=TrainingRunStatus.RUNNING,
        created_at="2026-04-18T00:00:00+00:00",
        started_at="2026-04-18T00:00:01+00:00",
        model_ref=compiled.model_ref,
        train_dataset_handle=dataset.handle,
        valid_dataset_handle=None,
        reference_profile=compiled.profile.key,
        checkpoint_handle=None,
        artifact_root=str(tmp_path / "outputs"),
        run_name="stale_run",
        pid=999999,
        log_path=str(tmp_path / "outputs" / "stale_run" / "training.log"),
        config_path=str(tmp_path / "outputs" / "stale_run.yaml"),
        run_root=str(tmp_path / "outputs" / "stale_run"),
        artifacts={},
        metrics={},
    )

    described = DemoTools(context=context).describe_training_run(training_run_handle=run.handle)

    assert described["ok"] is True
    assert described["data"]["training_run"]["status"] == "FAILED"
    assert described["data"]["training_run"]["error_type"] == "InfrastructureError"
    assert "terminal state" in described["data"]["training_run"]["error_message"]


def test_describe_training_run_reconciles_stale_queued_pid(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path, with_control=False)

    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset = context.facade.register_dataset_file(path=str(dataset_path))
    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=dataset.handle,
            model_key="kbf",
            run_name="stale_queued_run",
        ),
    )
    compiled_summary = context.facade.register_compiled_training_request(compiled_request=compiled)
    run = context.facade.register_training_run(
        compiled_request_handle=compiled_summary.handle,
        status=TrainingRunStatus.QUEUED,
        created_at="2026-04-18T00:00:00+00:00",
        model_ref=compiled.model_ref,
        train_dataset_handle=dataset.handle,
        valid_dataset_handle=None,
        reference_profile=compiled.profile.key,
        checkpoint_handle=None,
        artifact_root=str(tmp_path / "outputs"),
        run_name="stale_queued_run",
        pid=999999,
        log_path=str(tmp_path / "outputs" / "stale_queued_run" / "training.log"),
        config_path=str(tmp_path / "outputs" / "stale_queued_run.yaml"),
        run_root=str(tmp_path / "outputs" / "stale_queued_run"),
        artifacts={},
        metrics={},
    )

    described = DemoTools(context=context).describe_training_run(training_run_handle=run.handle)

    assert described["ok"] is True
    assert described["data"]["training_run"]["status"] == "FAILED"
    assert described["data"]["training_run"]["error_type"] == "InfrastructureError"
    assert "before recording a running state" in described["data"]["training_run"]["error_message"]
