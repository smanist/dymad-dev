"""Durable subprocess entrypoint for async training runs."""

from __future__ import annotations

import argparse
import importlib
import os
import traceback

from dymad.agent.exec.context import build_default_context
from dymad.agent.exec.workflow import _execute_training_run, _utc_now
from dymad.agent.store.object_store import TrainingRunStatus


def _apply_bootstrap_hook() -> None:
    hook = os.environ.get("DYMAD_TRAINING_WORKER_BOOTSTRAP")
    if not hook:
        return
    module_name, _, attr_name = hook.partition(":")
    if not module_name or not attr_name:
        raise ValueError("DYMAD_TRAINING_WORKER_BOOTSTRAP must be in '<module>:<callable>' form")
    module = importlib.import_module(module_name)
    bootstrap = getattr(module, attr_name)
    bootstrap()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-handle", required=True)
    args = parser.parse_args(argv)

    context = build_default_context(artifact_root=args.artifact_root)
    facade = context.facade
    run = facade.get_training_run(args.run_handle)

    if run.status in {TrainingRunStatus.SUCCEEDED, TrainingRunStatus.FAILED}:
        return 0

    started_at = run.started_at or _utc_now()
    facade.update_training_run(
        args.run_handle,
        status=TrainingRunStatus.RUNNING,
        started_at=started_at,
        error_type=None,
        error_message=None,
    )

    try:
        _apply_bootstrap_hook()
        compiled_request = facade.get_compiled_training_request(run.compiled_request_handle)
        result = _execute_training_run(
            facade=facade,
            model_ref=compiled_request.model_ref,
            train_dataset_handle=compiled_request.train_dataset_handle,
            valid_dataset_handle=compiled_request.valid_dataset_handle,
            reference_profile=compiled_request.reference_profile,
            run_name=compiled_request.effective_run_name,
            effective_config=compiled_request.effective_config,
            artifact_root=run.artifact_root,
            seed=compiled_request.seed,
            device=compiled_request.device,
            max_workers=compiled_request.max_workers,
        )
        facade.update_training_run(
            args.run_handle,
            status=TrainingRunStatus.SUCCEEDED,
            started_at=started_at,
            finished_at=_utc_now(),
            checkpoint_handle=result.checkpoint_summary.handle,
            artifacts=result.artifacts,
            metrics=result.metrics,
            error_type=None,
            error_message=None,
        )
        return 0
    except Exception as exc:
        traceback.print_exc()
        facade.update_training_run(
            args.run_handle,
            status=TrainingRunStatus.FAILED,
            started_at=started_at,
            finished_at=_utc_now(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
