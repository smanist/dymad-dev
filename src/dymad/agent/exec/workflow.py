"""Exec workflows over facade operations."""

from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
import torch
import yaml

from dymad.agent.exec.state import (
    AnalysisRunResult,
    DatasetInspection,
    DescribeTrainingRunResult,
    EvaluateModelResult,
    PredictionWorkflowPlan,
    ReadTrainingRunLogResult,
    SpectralWorkflowPlan,
    StartModelTrainingResult,
    StartTrainingRunResult,
)
from dymad.agent.exec.training_profiles import profile_config, resolve_profile_name
from dymad.agent.facade.operations import FacadeOperations
from dymad.agent.registry import SUPPORTED_EVALUATION_METRICS, resolve_model_capability
from dymad.agent.store.object_store import (
    CompiledAnalysisRequestRecord,
    CompiledTrainingRequestRecord,
    ObjectSummary,
    TrainingRunStatus,
)
from dymad.utils.misc import _normalize_legacy_training_config
from dymad.utils.plot import plot_trajectory

if TYPE_CHECKING:
    from dymad.agent.exec.context import ExecutionContext
    from dymad.sako.adapter import SpectralAnalysisAdapter, SpectralEigensystem, SpectralRuntime
    from dymad.sako.snapshot import SpectralSnapshot


logger = logging.getLogger(__name__)
_INFRASTRUCTURE_ERROR_TYPE = "InfrastructureError"


def _resolve_model_ref(model_ref: str):
    module_name, _, attr_name = model_ref.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"model_ref must be in '<module>:<name>' form, got: {model_ref}")
    module = importlib.import_module(module_name)
    try:
        model = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"model_ref target not found: {model_ref}") from exc
    return model


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _validate_user_config(config: dict[str, Any] | None, *, prefix: tuple[str, ...] = ()) -> None:
    if not isinstance(config, dict):
        return
    for key, value in config.items():
        path = prefix + (key,)
        if path[0] == "path":
            raise ValueError("config.path is reserved and cannot be overridden")
        if path in {("data", "path"), ("data_valid", "path")}:
            raise ValueError(f"{'.'.join(path)} is reserved and cannot be overridden")
        _validate_user_config(value, prefix=path)


def _default_run_name(model_ref: str) -> str:
    suffix = uuid4().hex[:8]
    return f"{model_ref.split(':', 1)[1].lower()}_{suffix}"


def _validate_run_name(run_name: str) -> str:
    candidate = run_name.strip()
    if not candidate:
        raise ValueError("run_name cannot be empty")
    if candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
        raise ValueError("run_name must be a simple file-name component without path separators")
    return candidate


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_request(device: str | None) -> torch.device | None:
    if device in (None, "", "auto"):
        return None
    return torch.device(device)


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_pid_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _requested_model_variant_ref(
    *,
    model_ref: str,
    capability,
    dataset_kind: str,
) -> str | None:
    normalized = model_ref.strip().lower()
    for variant in capability.variants:
        if normalized not in {
            variant.key.lower(),
            variant.name.lower(),
            variant.model_ref.lower(),
        }:
            continue
        if variant.dataset_kind != dataset_kind:
            raise ValueError(
                f"model_ref '{model_ref}' expects graph={variant.dataset_kind == 'graph'} "
                f"but dataset kind is '{dataset_kind}'"
            )
        return variant.model_ref
    return None


def _build_manager(dataset_record, *, default_device: str = "cpu"):
    from dymad.io import TrajectoryManager, TrajectoryManagerGraph

    metadata = {
        "config": {
            "data": {
                "path": dataset_record.path,
                "double_precision": False,
            }
        }
    }
    manager_cls = TrajectoryManagerGraph if dataset_record.kind == "graph" else TrajectoryManager
    return manager_cls(metadata, data_key="train", device=torch.device(default_device))


def _inspect_dataset_record(dataset_record) -> DatasetInspection:
    manager = _build_manager(dataset_record)
    manager.prepare_data()
    data = np.load(dataset_record.path, allow_pickle=True)
    try:
        keys = tuple(sorted(data.files))
    finally:
        data.close()
    lengths = [len(item) for item in manager.x]
    unique_lengths = sorted(set(lengths))
    return DatasetInspection(
        dataset_handle=dataset_record.handle,
        format=dataset_record.format,
        kind=dataset_record.kind,
        keys=keys,
        n_trajectories=len(manager.x),
        n_steps=unique_lengths[0] if len(unique_lengths) == 1 else None,
        is_ragged=len(unique_lengths) > 1,
        state_dim=int(manager.metadata["n_state_features"]),
        control_dim=int(manager.metadata["n_control_features"]),
        parameter_dim=int(manager.metadata["n_parameters"]),
        has_time="t" in keys,
        has_graph=dataset_record.kind == "graph",
        n_nodes=int(manager.metadata["n_nodes"]) if dataset_record.kind == "graph" else None,
    )


def _effective_config(
    *,
    model_ref: str,
    dataset_record,
    valid_dataset_record,
    reference_profile: str | None,
    user_config: dict[str, Any] | None,
    run_name: str,
) -> tuple[str, dict[str, Any]]:
    _validate_user_config(user_config or {})
    profile_name = resolve_profile_name(
        model_ref=model_ref,
        dataset_kind=dataset_record.kind,
        reference_profile=reference_profile,
    )
    config = profile_config(profile_name)
    config = _deep_merge(config, user_config or {})
    config.setdefault("data", {})
    config["data"]["path"] = dataset_record.path
    if valid_dataset_record is None:
        config.pop("data_valid", None)
    else:
        config["data_valid"] = {
            "path": valid_dataset_record.path,
            "double_precision": bool(config["data"].get("double_precision", False)),
        }
    config.setdefault("model", {})
    config["model"]["name"] = run_name
    config.pop("path", None)
    _normalize_legacy_training_config(config)
    return profile_name, config


def _select_trainer(config: dict[str, Any]):
    from dymad.training import (
        LinearTrainer,
        NODETrainer,
        OneStepTrainer,
        StackedTrainer,
        WeakFormTrainer,
    )

    phases = config.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("effective config must contain at least one phase")
    optimizer_phases = [
        phase
        for phase in phases
        if isinstance(phase, dict) and (phase.get("type") == "optimizer" or "trainer" in phase)
    ]
    if not optimizer_phases:
        raise ValueError("effective config must contain at least one optimizer phase")
    if len(optimizer_phases) > 1:
        return StackedTrainer, "stacked"
    trainer = optimizer_phases[0].get("trainer")
    if trainer == "NODE":
        return NODETrainer, "node"
    if trainer == "Weak":
        return WeakFormTrainer, "weak_form"
    if trainer == "Linear":
        return LinearTrainer, "linear"
    if trainer == "OneStep":
        return OneStepTrainer, "one_step"
    raise ValueError(f"unsupported trainer kind in phases: {trainer}")


def _write_training_config(config: dict[str, Any], *, artifact_root: Path, run_name: str) -> Path:
    config_path = artifact_root / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _load_training_metrics(summary_path: Path) -> dict[str, float | None]:
    with np.load(summary_path, allow_pickle=True) as npz:
        best_valid = npz["best_valid_loss"].item()
        return {
            "best_valid_total": (
                float(best_valid["valid_total"]) if "valid_total" in best_valid else None
            ),
            "final_valid_loss": float(npz["final_valid_loss"]),
            "avg_epoch_time": float(npz["avg_epoch_time"]),
        }


def _default_training_artifacts() -> dict[str, str | None]:
    return {
        "checkpoint_path": None,
        "training_summary_path": None,
        "history_plot_path": None,
        "prediction_plot_path": None,
        "cv_results_path": None,
        "cv_plot_path": None,
    }


@dataclass(frozen=True)
class _ExecutedTrainingRun:
    checkpoint_summary: ObjectSummary
    artifacts: dict[str, str | None]
    metrics: dict[str, float | None]
    reference_profile: str
    trainer_kind: str


def _execute_training_run(
    *,
    facade: FacadeOperations,
    model_ref: str,
    train_dataset_handle: str,
    valid_dataset_handle: str | None,
    reference_profile: str,
    run_name: str,
    effective_config: dict[str, Any],
    artifact_root: str,
    seed: int | None,
    device: str,
    max_workers: int,
) -> _ExecutedTrainingRun:
    model = _resolve_model_ref(model_ref)
    artifact_root_path = _ensure_dir(artifact_root)
    trainer_cls, trainer_kind = _select_trainer(effective_config)
    config_path = _write_training_config(
        effective_config,
        artifact_root=artifact_root_path,
        run_name=run_name,
    )

    _set_seed(seed)
    trainer = trainer_cls(
        str(config_path),
        model,
        device=_device_from_request(device),
        max_workers=max_workers,
    )
    trainer.train()

    run_root = artifact_root_path / run_name
    checkpoint_path = run_root / f"{run_name}.pt"
    summary_path = run_root / f"{run_name}_summary.npz"
    history_plot_path = run_root / f"{run_name}_history.png"
    prediction_plot_path = run_root / f"{run_name}_prediction.png"
    cv_results_path = run_root / f"{run_name}_cv.npz"
    cv_plot_path = run_root / "cv_results.png"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"training did not produce checkpoint: {checkpoint_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"training did not produce summary: {summary_path}")

    checkpoint_summary = facade.register_checkpoint(
        model_ref=model_ref,
        checkpoint_path=str(checkpoint_path),
        device=str(trainer.device),
    )
    return _ExecutedTrainingRun(
        checkpoint_summary=checkpoint_summary,
        artifacts={
            "checkpoint_path": str(checkpoint_path),
            "training_summary_path": str(summary_path),
            "history_plot_path": str(history_plot_path) if history_plot_path.is_file() else None,
            "prediction_plot_path": (
                str(prediction_plot_path) if prediction_plot_path.is_file() else None
            ),
            "cv_results_path": str(cv_results_path) if cv_results_path.is_file() else None,
            "cv_plot_path": str(cv_plot_path) if cv_plot_path.is_file() else None,
        },
        metrics=_load_training_metrics(summary_path),
        reference_profile=reference_profile,
        trainer_kind=trainer_kind,
    )


def _trajectory_payload(manager, index: int) -> dict[str, Any]:
    from dymad.io import TrajectoryManagerGraph

    payload = {
        "x": manager.x[index],
        "t": manager.t[index],
        "u": None if manager.metadata["n_control_features"] == 0 else manager.u[index],
        "p": None if manager.metadata["n_parameters"] == 0 else manager.p[index],
    }
    if isinstance(manager, TrajectoryManagerGraph):
        payload.update(
            {
                "ei": manager.ei[index],
                "ew": None if manager.metadata["n_edge_weights"] == 0 else manager.ew[index],
                "ea": None if manager.metadata["n_edge_features"] == 0 else manager.ea[index],
            }
        )
    return payload


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _plot_indices(errors: np.ndarray, *, selection: str, max_plots: int) -> list[int]:
    if max_plots <= 0 or errors.size == 0:
        return []
    if selection == "best":
        ordered = np.argsort(errors)
    elif selection == "worst":
        ordered = np.argsort(errors)[::-1]
    elif selection == "median":
        median = float(np.median(errors))
        ordered = np.argsort(np.abs(errors - median))
    else:
        raise ValueError(f"unsupported plot_selection: {selection}")
    return [int(index) for index in ordered[:max_plots]]


@dataclass
class CompatibilityExecutor:
    """Plans typed-handle flows and executes train/evaluate compatibility workflows."""

    facade: FacadeOperations
    context_provider: Callable[[], ExecutionContext] | None = None

    def _active_context(self) -> ExecutionContext:
        if self.context_provider is None:
            raise RuntimeError("CompatibilityExecutor requires an execution context provider")
        return self.context_provider()

    def inspect_dataset(self, *, dataset_handle: str) -> DatasetInspection:
        return _inspect_dataset_record(self.facade.get_dataset(dataset_handle))

    def run_analysis_request(
        self,
        *,
        compiled_request_handle: str,
        artifact_root: str,
    ) -> AnalysisRunResult:
        compiled_request = self.facade.get_compiled_analysis_request(compiled_request_handle)
        return self._run_compiled_analysis_request(
            compiled_request=compiled_request,
            artifact_root=artifact_root,
        )

    def start_training_run(
        self,
        *,
        compiled_request_handle: str,
        artifact_root: str,
    ) -> StartTrainingRunResult:
        compiled_request = self.facade.get_compiled_training_request(compiled_request_handle)
        return self._start_training_run_record(
            compiled_request_handle=compiled_request_handle,
            compiled_request=compiled_request,
            artifact_root=artifact_root,
        )

    def start_model_training(
        self,
        *,
        train_dataset_handle: str,
        valid_dataset_handle: str | None = None,
        model_ref: str,
        reference_profile: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        artifact_root: str,
        seed: int | None = None,
        device: str = "auto",
        max_workers: int = 1,
    ) -> StartModelTrainingResult:
        dataset_record = self.facade.get_dataset(train_dataset_handle)
        valid_record = (
            None if valid_dataset_handle is None else self.facade.get_dataset(valid_dataset_handle)
        )
        if valid_record is not None and valid_record.kind != dataset_record.kind:
            raise ValueError("train and valid datasets must have the same kind")

        capability = resolve_model_capability(model_ref)
        requested_variant_model_ref = _requested_model_variant_ref(
            model_ref=model_ref,
            capability=capability,
            dataset_kind=dataset_record.kind,
        )
        from dymad.agent.compiler import TrainingRequest, compile_training_request

        compiled_request = compile_training_request(
            facade=self.facade,
            request=TrainingRequest(
                train_dataset_handle=train_dataset_handle,
                model_key=capability.key,
                valid_dataset_handle=valid_dataset_handle,
                reference_profile=reference_profile,
                overrides=config,
                run_name=run_name,
                seed=seed,
                device=device,
                max_workers=max_workers,
            ),
        )
        if requested_variant_model_ref is not None:
            _, effective_config = _effective_config(
                model_ref=requested_variant_model_ref,
                dataset_record=dataset_record,
                valid_dataset_record=valid_record,
                reference_profile=compiled_request.request.reference_profile,
                user_config=cast(dict[str, Any] | None, compiled_request.request.overrides),
                run_name=compiled_request.effective_run_name,
            )
            _, trainer_kind = _select_trainer(effective_config)
            compiled_request = replace(
                compiled_request,
                model_ref=requested_variant_model_ref,
                effective_config=effective_config,
                trainer_kind=trainer_kind,
            )
        compiled_request_summary = self.facade.register_compiled_training_request(
            compiled_request=compiled_request
        )
        launch = self._start_training_run_record(
            compiled_request_handle=compiled_request_summary.handle,
            compiled_request=self.facade.get_compiled_training_request(
                compiled_request_summary.handle
            ),
            artifact_root=artifact_root,
        )
        return StartModelTrainingResult(
            summary=launch.summary,
            training_run=launch.training_run,
            compiled_request_summary=compiled_request_summary,
        )

    def describe_training_run(
        self,
        *,
        training_run_handle: str,
    ) -> DescribeTrainingRunResult:
        run = self.facade.refresh_training_run(training_run_handle)
        if (
            run.status in {TrainingRunStatus.QUEUED, TrainingRunStatus.RUNNING}
            and run.finished_at is None
            and run.pid is not None
            and not _is_pid_running(run.pid)
        ):
            message = (
                "training worker exited before recording a running state"
                if run.status is TrainingRunStatus.QUEUED
                else "training worker exited without recording a terminal state"
            )
            run = self.facade.update_training_run(
                training_run_handle,
                status=TrainingRunStatus.FAILED,
                finished_at=_utc_now(),
                error_type=_INFRASTRUCTURE_ERROR_TYPE,
                error_message=message,
            )
        return DescribeTrainingRunResult(
            summary=self.facade.describe_object(training_run_handle),
            training_run=run,
        )

    def read_training_run_log(
        self,
        *,
        training_run_handle: str,
        offset: int = 0,
        max_bytes: int = 65536,
    ) -> ReadTrainingRunLogResult:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        run = self.describe_training_run(training_run_handle=training_run_handle).training_run
        if run.log_path is None:
            return ReadTrainingRunLogResult(text="", next_offset=offset, eof=True)
        log_path = Path(run.log_path)
        if not log_path.is_file():
            return ReadTrainingRunLogResult(text="", next_offset=offset, eof=True)
        size = log_path.stat().st_size
        next_offset = min(offset, size)
        with log_path.open("rb") as fh:
            fh.seek(next_offset)
            chunk = fh.read(max_bytes)
        updated_offset = next_offset + len(chunk)
        return ReadTrainingRunLogResult(
            text=chunk.decode("utf-8", errors="replace"),
            next_offset=updated_offset,
            eof=updated_offset >= size,
        )

    def _start_training_run_record(
        self,
        *,
        compiled_request_handle: str,
        compiled_request: CompiledTrainingRequestRecord,
        artifact_root: str,
    ) -> StartTrainingRunResult:
        artifact_root_path = _ensure_dir(artifact_root)
        run_root = artifact_root_path / compiled_request.effective_run_name
        run_root.mkdir(parents=True, exist_ok=True)
        config_path = artifact_root_path / f"{compiled_request.effective_run_name}.yaml"
        log_path = run_root / "training.log"
        log_path.touch(exist_ok=True)

        run_summary = self.facade.register_training_run(
            compiled_request_handle=compiled_request_handle,
            status=TrainingRunStatus.QUEUED,
            created_at=_utc_now(),
            model_ref=compiled_request.model_ref,
            train_dataset_handle=compiled_request.train_dataset_handle,
            valid_dataset_handle=compiled_request.valid_dataset_handle,
            reference_profile=compiled_request.reference_profile,
            checkpoint_handle=None,
            artifact_root=str(artifact_root_path),
            run_name=compiled_request.effective_run_name,
            artifacts=_default_training_artifacts(),
            metrics={},
        )

        context = self._active_context()
        store_root = context.store.artifact_store_root
        if store_root is None:
            raise RuntimeError("artifact store root is unavailable for training worker launch")

        try:
            with log_path.open("ab") as log_stream:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "dymad.agent.exec.training_worker",
                        "--artifact-root",
                        store_root,
                        "--run-handle",
                        run_summary.handle,
                    ],
                    cwd=str(Path.cwd()),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            run = self.facade.update_training_run(
                run_summary.handle,
                pid=process.pid,
                log_path=str(log_path),
                config_path=str(config_path),
                run_root=str(run_root),
            )
        except Exception as exc:
            run = self.facade.update_training_run(
                run_summary.handle,
                status=TrainingRunStatus.FAILED,
                finished_at=_utc_now(),
                log_path=str(log_path),
                config_path=str(config_path),
                run_root=str(run_root),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        return StartTrainingRunResult(
            summary=self.facade.describe_object(run_summary.handle),
            training_run=run,
        )

    def _run_compiled_analysis_request(
        self,
        *,
        compiled_request: CompiledAnalysisRequestRecord,
        artifact_root: str,
    ) -> AnalysisRunResult:
        if compiled_request.workflow_key == "spectral_koopman":
            checkpoint = self.facade.get_checkpoint(cast(str, compiled_request.checkpoint_handle))
            from dymad.sako.base import SpectralAnalysis

            model_class = _resolve_model_ref(checkpoint.model_ref)
            params = dict(compiled_request.parameters)
            analysis = SpectralAnalysis(
                model_class,
                checkpoint.checkpoint_path,
                dt=float(params.get("dt", 1.0)),
                forder=params.get("forder", "full"),
                reps=float(params.get("reps", 1e-10)),
                etol=float(params.get("etol", 1e-13)),
                remove_one=bool(params.get("remove_one", True)),
                exec_context=self._active_context(),
            )
            root = _ensure_dir(Path(artifact_root).expanduser() / "analyses")
            token = uuid4().hex[:12]
            summary_path = root / f"{token}_spectral_summary.json"
            summary_payload = {
                "workflow_key": "spectral_koopman",
                "checkpoint_handle": checkpoint.handle,
                "checkpoint_path": checkpoint.checkpoint_path,
                "n_eigs": int(len(np.asarray(analysis._wd))),
                "n_eigs_full": int(len(np.asarray(analysis._wd_full))),
                "obs_dim": int(analysis._ctx.snapshot.obs_dim),
                "sample_count": int(analysis._ctx.snapshot.sample_count),
            }
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return AnalysisRunResult(
                workflow_key="spectral_koopman",
                artifacts={"summary_path": str(summary_path)},
                summary=summary_payload,
            )

        if compiled_request.workflow_key == "vortex_transform_modes":
            from dymad.agent.exec.vortex_analysis import (
                compute_vortex_mode_analysis,
                persist_vortex_mode_analysis,
            )

            train_dataset = self.facade.get_dataset(
                compiled_request.dataset_handles["train_dataset_handle"]
            )
            test_dataset = self.facade.get_dataset(
                compiled_request.dataset_handles["test_dataset_handle"]
            )
            params = dict(compiled_request.parameters)
            analysis = compute_vortex_mode_analysis(
                config_path=cast(str, params["config_path"]),
                train_dataset_path=train_dataset.path,
                test_dataset_path=test_dataset.path,
                index=int(params.get("index", 5)),
                nx=int(params.get("nx", 199)),
                ny=int(params.get("ny", 449)),
            )
            persisted = persist_vortex_mode_analysis(analysis, artifact_root=artifact_root)
            return AnalysisRunResult(
                workflow_key="vortex_transform_modes",
                artifacts={
                    "output_path": persisted.output_path,
                    "summary_path": persisted.summary_path,
                },
                summary={
                    "workflow_key": "vortex_transform_modes",
                    "rel_dx_error": persisted.rel_dx_error,
                    "rel_dz_error": persisted.rel_dz_error,
                    "index": persisted.index,
                    "nx": persisted.nx,
                    "ny": persisted.ny,
                },
            )

        raise ValueError(f"unsupported analysis workflow: {compiled_request.workflow_key}")

    def evaluate_model(
        self,
        *,
        checkpoint_handle: str,
        test_dataset_handle: str,
        metric: str,
        artifact_root: str,
        plot_selection: str = "median",
        max_plots: int = 1,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> EvaluateModelResult:
        if metric not in SUPPORTED_EVALUATION_METRICS:
            raise ValueError(f"unsupported evaluation metric: {metric}")
        checkpoint = self.facade.get_checkpoint(checkpoint_handle)
        dataset = self.facade.get_dataset(test_dataset_handle)
        model = _resolve_model_ref(checkpoint.model_ref)
        if bool(getattr(model, "GRAPH", False)) != (dataset.kind == "graph"):
            raise ValueError(
                f"checkpoint model '{checkpoint.model_ref}' expects graph="
                f"{bool(getattr(model, 'GRAPH', False))} but dataset kind is '{dataset.kind}'"
            )

        manager = _build_manager(dataset)
        manager.prepare_data()
        from dymad.io import TrajectoryManagerGraph, load_model

        _, predict_fn = cast(
            tuple[Any, Callable[..., np.ndarray]],
            load_model(
                model,
                checkpoint.checkpoint_path,
                context=self._active_context(),
            ),
        )

        predictions: list[np.ndarray] = []
        errors: list[float] = []
        for index in range(len(manager.x)):
            payload = _trajectory_payload(manager, index)
            kwargs = dict(predict_kwargs or {})
            kwargs.setdefault("device", "cpu")
            pred = cast(
                np.ndarray,
                predict_fn(
                    payload["x"],
                    payload["t"],
                    u=payload["u"],
                    p=payload["p"],
                    **(
                        {}
                        if not isinstance(manager, TrajectoryManagerGraph)
                        else {"ei": payload["ei"], "ew": payload["ew"], "ea": payload["ea"]}
                    ),
                    **kwargs,
                ),
            )
            predictions.append(pred)
            errors.append(_rmse(pred, payload["x"]))

        error_array = np.asarray(errors, dtype=float)
        metrics = {
            "rmse_mean": float(np.mean(error_array)),
            "rmse_std": float(np.std(error_array)),
            "rmse_median": float(np.median(error_array)),
            "rmse_min": float(np.min(error_array)),
            "rmse_max": float(np.max(error_array)),
            "n_test_trajectories": float(len(error_array)),
        }

        eval_root = _ensure_dir(Path(artifact_root).expanduser() / "evaluations")
        eval_token = uuid4().hex[:12]
        metrics_path = eval_root / f"{eval_token}_metrics.json"
        metrics_payload = {
            "metric": metric,
            "aggregate": metrics,
            "per_trajectory_rmse": [float(value) for value in error_array.tolist()],
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        plot_paths: list[str] = []
        plot_skipped_reason = None
        if dataset.kind == "graph":
            plot_skipped_reason = "graph plotting unsupported in v1"
        else:
            for rank, index in enumerate(
                _plot_indices(error_array, selection=plot_selection, max_plots=max_plots)
            ):
                payload = _trajectory_payload(manager, index)
                model_name = f"{eval_token}_{plot_selection}_{rank}"
                try:
                    plot_trajectory(
                        np.array([payload["x"], predictions[index]]),
                        payload["t"],
                        model_name=model_name,
                        us=payload["u"],
                        labels=["Truth", "Prediction"],
                        ifclose=True,
                        prefix=str(eval_root),
                    )
                except Exception:
                    logger.warning(
                        "Skipping evaluation plot '%s' due to plotting failure.",
                        model_name,
                        exc_info=True,
                    )
                    if plot_skipped_reason is None:
                        plot_skipped_reason = "plotting failed"
                    continue
                plot_paths.append(str(eval_root / f"{model_name}_prediction.png"))

        evaluation_summary = self.facade.register_evaluation(
            checkpoint_handle=checkpoint_handle,
            test_dataset_handle=test_dataset_handle,
            metric=metric,
            metrics_path=str(metrics_path),
            plot_paths=plot_paths,
        )
        return EvaluateModelResult(
            evaluation_summary=evaluation_summary,
            artifacts={
                "metrics_path": str(metrics_path),
                "plot_paths": plot_paths,
            },
            metrics=metrics,
            plot_skipped_reason=plot_skipped_reason,
        )

    def plan_checkpoint_prediction(
        self,
        *,
        model_ref: str,
        checkpoint_path: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> PredictionWorkflowPlan:
        checkpoint = self.facade.register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
        )
        request = self.facade.prepare_prediction_request(
            checkpoint_handle=checkpoint.handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )
        return PredictionWorkflowPlan(
            checkpoint_handle=checkpoint.handle,
            prediction_handle=request.handle,
            entrypoint="dymad.io.checkpoint.load_model",
            notes=(
                "This skeleton intentionally records boundary state only.",
                "Numerical model behavior remains in legacy io/models modules.",
            ),
        )

    def materialize_checkpoint_prediction(
        self,
        *,
        plan: PredictionWorkflowPlan,
        model_class: type[Any],
    ) -> tuple[Any, Callable[..., Any]]:
        raise NotImplementedError(
            "Checkpoint materialization is no longer routed through CompatibilityExecutor. "
            "Use dymad.io.load_model for now; executor-native materialization is pending."
        )

    def plan_spectral_analysis(
        self,
        *,
        model_ref: str,
        checkpoint_path: str,
        snapshot: SpectralSnapshot,
    ) -> SpectralWorkflowPlan:
        checkpoint = self.facade.register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
        )
        snapshot_summary = self.facade.register_spectral_snapshot(
            checkpoint_handle=checkpoint.handle,
            snapshot=snapshot,
        )
        return SpectralWorkflowPlan(
            checkpoint_handle=checkpoint.handle,
            spectral_snapshot_handle=snapshot_summary.handle,
            entrypoint="dymad.sako.SpectralAnalysis",
            notes=(
                "Spectral snapshot is persisted and resolved through facade/store handles.",
                "Numerical kernels still execute through the adapter compatibility layer.",
            ),
        )

    def materialize_spectral_adapter(
        self,
        *,
        plan: SpectralWorkflowPlan,
        eigensystem: SpectralEigensystem,
        runtime: SpectralRuntime | None = None,
        reps: float = 1e-10,
        etol: float = 1e-13,
    ) -> SpectralAnalysisAdapter:
        from dymad.sako.adapter import SpectralAnalysisAdapter

        snapshot_record = self.facade.get_spectral_snapshot(plan.spectral_snapshot_handle)
        if snapshot_record.checkpoint_handle != plan.checkpoint_handle:
            raise ValueError("plan checkpoint/spectral handles are inconsistent")
        return SpectralAnalysisAdapter(
            snapshot=snapshot_record.snapshot,
            eigensystem=eigensystem,
            runtime=runtime,
            reps=reps,
            etol=etol,
        )
