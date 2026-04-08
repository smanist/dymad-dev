"""Exec workflows over facade operations."""

from __future__ import annotations

import copy
import importlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from dymad.agent.exec.state import (
    DatasetInspection,
    EvaluateModelResult,
    PredictionWorkflowPlan,
    SpectralWorkflowPlan,
    TrainModelResult,
)
from dymad.agent.exec.training_profiles import profile_config, resolve_profile_name
from dymad.agent.facade.operations import FacadeOperations
from dymad.utils.misc import _normalize_legacy_training_config
from dymad.utils.plot import plot_trajectory

if TYPE_CHECKING:
    from dymad.agent.exec.context import ExecutionContext
    from dymad.sako.adapter import SpectralAnalysisAdapter, SpectralEigensystem, SpectralRuntime
    from dymad.sako.snapshot import SpectralSnapshot


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
    from dymad.training import LinearTrainer, NODETrainer, StackedTrainer, WeakFormTrainer

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
    context_provider: Callable[[], "ExecutionContext"] | None = None

    def _active_context(self) -> "ExecutionContext":
        if self.context_provider is None:
            raise RuntimeError("CompatibilityExecutor requires an execution context provider")
        return self.context_provider()

    def inspect_dataset(self, *, dataset_handle: str) -> DatasetInspection:
        return _inspect_dataset_record(self.facade.get_dataset(dataset_handle))

    def train_model(
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
    ) -> TrainModelResult:
        model = _resolve_model_ref(model_ref)
        dataset_record = self.facade.get_dataset(train_dataset_handle)
        valid_record = (
            None if valid_dataset_handle is None else self.facade.get_dataset(valid_dataset_handle)
        )
        if bool(getattr(model, "GRAPH", False)) != (dataset_record.kind == "graph"):
            raise ValueError(
                f"model_ref '{model_ref}' expects graph={bool(getattr(model, 'GRAPH', False))} "
                f"but dataset kind is '{dataset_record.kind}'"
            )
        if valid_record is not None and valid_record.kind != dataset_record.kind:
            raise ValueError("train and valid datasets must have the same kind")

        active_run_name = _validate_run_name(run_name or _default_run_name(model_ref))
        profile_name, effective_config = _effective_config(
            model_ref=model_ref,
            dataset_record=dataset_record,
            valid_dataset_record=valid_record,
            reference_profile=reference_profile,
            user_config=config,
            run_name=active_run_name,
        )
        trainer_cls, trainer_kind = _select_trainer(effective_config)
        artifact_root_path = _ensure_dir(artifact_root)
        config_path = _write_training_config(
            effective_config,
            artifact_root=artifact_root_path,
            run_name=active_run_name,
        )

        _set_seed(seed)
        trainer = trainer_cls(
            str(config_path),
            model,
            device=_device_from_request(device),
            max_workers=max_workers,
        )
        trainer.train()

        run_root = artifact_root_path / active_run_name
        checkpoint_path = run_root / f"{active_run_name}.pt"
        summary_path = run_root / f"{active_run_name}_summary.npz"
        history_plot_path = run_root / f"{active_run_name}_history.png"
        prediction_plot_path = run_root / f"{active_run_name}_prediction.png"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"training did not produce checkpoint: {checkpoint_path}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"training did not produce summary: {summary_path}")

        checkpoint_summary = self.facade.register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=str(checkpoint_path),
            device=str(trainer.device),
        )
        run_summary = self.facade.register_training_run(
            model_ref=model_ref,
            train_dataset_handle=train_dataset_handle,
            valid_dataset_handle=valid_dataset_handle,
            reference_profile=profile_name,
            checkpoint_handle=checkpoint_summary.handle,
            artifact_root=str(artifact_root_path),
            run_name=active_run_name,
        )

        return TrainModelResult(
            run_summary=run_summary,
            checkpoint_summary=checkpoint_summary,
            artifacts={
                "checkpoint_path": str(checkpoint_path),
                "training_summary_path": str(summary_path),
                "history_plot_path": str(history_plot_path) if history_plot_path.is_file() else None,
                "prediction_plot_path": (
                    str(prediction_plot_path) if prediction_plot_path.is_file() else None
                ),
            },
            metrics=_load_training_metrics(summary_path),
            reference_profile=profile_name,
            trainer_kind=trainer_kind,
        )

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
        if metric != "rollout_rmse":
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

        _, predict_fn = load_model(
            model,
            checkpoint.checkpoint_path,
            context=self._active_context(),
        )

        predictions: list[np.ndarray] = []
        errors: list[float] = []
        for index in range(len(manager.x)):
            payload = _trajectory_payload(manager, index)
            kwargs = dict(predict_kwargs or {})
            kwargs.setdefault("device", "cpu")
            pred = predict_fn(
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
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

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
                plot_trajectory(
                    np.array([payload["x"], predictions[index]]),
                    payload["t"],
                    model_name=model_name,
                    us=payload["u"],
                    labels=["Truth", "Prediction"],
                    ifclose=True,
                    prefix=str(eval_root),
                )
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
