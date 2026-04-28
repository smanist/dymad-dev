from __future__ import annotations

import copy
import logging
import random
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from dymad.core import GraphSeries, GraphTrainerBatch, RegularSeries, RegularTrainerBatch
from dymad.core.transform_builder import build_transform_module
from dymad.losses import LOSS_MAP
from dymad.numerics import denoise, denoising_metrics, generate_weak_weights
from dymad.training.batch_adapter import RuntimeBatch, TrainerBatch, batch_to_runtime
from dymad.training.execution_services import ExecutionServices
from dymad.training.ls_update import LSUpdater, _comp_linear_eval_ct, _comp_linear_eval_dt
from dymad.training.phase_runtime import (
    ArtifactRegistry,
    EvaluationArtifact,
    ExportArtifact,
    LinearSolveRecord,
    LinearSolveReportArtifact,
    ModelArtifact,
    OptimizerStateArtifact,
    PhaseContext,
    PhaseRecord,
    PhaseResult,
    TrainerState,
    TrainingHistoryArtifact,
)
from dymad.utils import make_scheduler, plot_hist, plot_trajectory

logger = logging.getLogger(__name__)


def _safe_plot(logger: logging.Logger, *, label: str, fn) -> bool:
    try:
        fn()
    except Exception:
        logger.warning("Skipping %s due to plotting failure.", label, exc_info=True)
        return False
    return True


class PhaseSpecValidationError(ValueError):
    """Raised when a phase spec or normalized legacy config is invalid."""


@dataclass(frozen=True)
class BasePhaseSpec:
    name: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerPhaseSpec(BasePhaseSpec):
    trainer: str = "NODE"
    reset_optimizer: bool = False

    def __init__(
        self,
        name: str,
        trainer: str,
        config: dict[str, Any],
        *,
        reset_optimizer: bool = False,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "optimizer")
        object.__setattr__(self, "trainer", trainer)
        object.__setattr__(self, "reset_optimizer", reset_optimizer)
        object.__setattr__(self, "config", config)


@dataclass(frozen=True)
class LinearSolvePhaseSpec(BasePhaseSpec):
    method: str = "full"
    params: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    reset_optimizer: bool = True

    def __init__(
        self,
        name: str,
        *,
        method: str,
        params: Any = None,
        kwargs: dict[str, Any] | None = None,
        reset_optimizer: bool = True,
        config: dict[str, Any] | None = None,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "linear_solve")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "kwargs", {} if kwargs is None else kwargs)
        object.__setattr__(self, "reset_optimizer", reset_optimizer)
        object.__setattr__(self, "config", {} if config is None else config)


@dataclass(frozen=True)
class DataPhaseSpec(BasePhaseSpec):
    operation: str = "context"

    def __init__(
        self, name: str, *, operation: str = "context", config: dict[str, Any] | None = None
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "data")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "config", {} if config is None else config)


@dataclass(frozen=True)
class AnalysisPhaseSpec(BasePhaseSpec):
    split: str = "valid"
    evaluate_all: bool = False

    def __init__(
        self,
        name: str,
        *,
        split: str = "valid",
        evaluate_all: bool = False,
        config: dict[str, Any] | None = None,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "analysis")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "evaluate_all", evaluate_all)
        object.__setattr__(self, "config", {} if config is None else config)


@dataclass(frozen=True)
class ExportPhaseSpec(BasePhaseSpec):
    export_kind: str = "best_model"

    def __init__(self, name: str, *, export_kind: str, config: dict[str, Any] | None = None):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "export")
        object.__setattr__(self, "export_kind", export_kind)
        object.__setattr__(self, "config", {} if config is None else config)


PhaseSpec = (
    OptimizerPhaseSpec | LinearSolvePhaseSpec | DataPhaseSpec | AnalysisPhaseSpec | ExportPhaseSpec
)


def _dataset_len(dataset: Sequence[TrainerBatch] | None) -> int:
    return 0 if dataset is None else len(dataset)


def _dataset_collate_fn(dataset: Sequence[TrainerBatch]):
    if not dataset:
        return RegularTrainerBatch.collate_series
    first = dataset[0]
    if isinstance(first, GraphSeries):
        return GraphTrainerBatch.collate_series
    if isinstance(first, RegularSeries):
        return RegularTrainerBatch.collate_series
    raise TypeError(f"Unsupported dataset item type '{type(first)}' for dataloader collation.")


def _build_dataset_loader(
    dataset: Sequence[TrainerBatch] | None,
    *,
    config: dict[str, Any],
) -> DataLoader[TrainerBatch] | None:
    if dataset is None:
        return None
    dl_cfg = copy.deepcopy(config.get("dataloader", {}))
    batch_size = int(dl_cfg.get("batch_size", 1))
    shuffle = bool(dl_cfg.get("shuffle", True))
    return DataLoader(
        cast(Any, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_dataset_collate_fn(dataset),
    )


def _dataset_first(
    dataset: Sequence[TrainerBatch] | None,
) -> tuple[TrainerBatch | None, dict[str, Any] | None]:
    if dataset is None or len(dataset) == 0:
        return None, None
    return dataset[0], None


def _require_loader(loader: DataLoader[TrainerBatch] | None) -> DataLoader[TrainerBatch]:
    if loader is None:
        raise ValueError("Expected dataloader to be initialized.")
    return loader


def _require_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        raise ValueError("Expected metadata to be initialized.")
    return metadata


def _require_loss_class(loss_key: str) -> type[torch.nn.Module]:
    loss_class = LOSS_MAP.get(loss_key)
    if loss_class is None:
        raise KeyError(f"Unknown loss '{loss_key}'.")
    return cast(type[torch.nn.Module], loss_class)


def _determine_chop_step(window: int, step: int | float) -> int:
    if isinstance(step, int):
        return step
    if isinstance(step, float):
        stp = int(window * step)
        return min(max(stp, 1), window)
    raise ValueError(f"Invalid step type: {type(step)}. Expected int or float.")


def _normalize_legacy_optimizer_name(trainer: str) -> str:
    if trainer not in {"NODE", "Weak", "Linear", "OneStep"}:
        raise PhaseSpecValidationError(
            f"Unsupported trainer '{trainer}'. Expected one of NODE, Weak, Linear, OneStep."
        )
    return trainer


def _optimizer_spec_from_legacy(
    entry: dict[str, Any], index: int, suffix: str = ""
) -> OptimizerPhaseSpec:
    cfg = copy.deepcopy(entry)
    trainer = _normalize_legacy_optimizer_name(cfg.pop("trainer"))
    reset_optimizer = bool(cfg.pop("reset_optimizer", False))
    name = cfg.get("name", f"phase_{index}")
    if suffix:
        name = f"{name}_{suffix}"
    return OptimizerPhaseSpec(
        name=name,
        trainer=trainer,
        config=cfg,
        reset_optimizer=reset_optimizer,
    )


def _raise_legacy_ls_update_error() -> None:
    raise PhaseSpecValidationError(
        "'ls_update' is deprecated and no longer supported. "
        "Use explicit 'type: linear_solve' phases, optionally inside a 'repeat' block."
    )


def _with_repeat_name(entry: dict[str, Any], index: int, suffix: str) -> dict[str, Any]:
    cloned = copy.deepcopy(entry)
    if "repeat" in cloned:
        repeat_cfg = copy.deepcopy(cloned["repeat"])
        repeat_cfg.setdefault("name", f"repeat_{index}_{suffix}")
        cloned["repeat"] = repeat_cfg
        return cloned
    base_name = cloned.get("name", f"phase_{index}")
    cloned["name"] = f"{base_name}_{suffix}"
    return cloned


def _warn_if_repeat_contains_terminal_phase(spec: PhaseSpec, repeat_name: str) -> None:
    if isinstance(spec, AnalysisPhaseSpec):
        warnings.warn(
            f"Repeat block '{repeat_name}' contains an analysis phase. "
            "That is allowed, but analysis is usually more meaningful at the top level.",
            UserWarning,
            stacklevel=3,
        )
    if isinstance(spec, ExportPhaseSpec):
        warnings.warn(
            f"Repeat block '{repeat_name}' contains an export phase. "
            "That is allowed, but export is usually more meaningful at the top level.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_explicit_phase(entry: dict[str, Any], index: int) -> PhaseSpec:
    phase_type = entry.get("type")
    if phase_type == "optimizer":
        trainer = _normalize_legacy_optimizer_name(entry["trainer"])
        return OptimizerPhaseSpec(
            name=entry.get("name", f"phase_{index}"),
            trainer=trainer,
            reset_optimizer=entry.get("reset_optimizer", False),
            config={
                k: copy.deepcopy(v)
                for k, v in entry.items()
                if k not in {"type", "name", "trainer", "reset_optimizer"}
            },
        )
    if phase_type == "linear_solve":
        return LinearSolvePhaseSpec(
            name=entry.get("name", f"phase_{index}"),
            method=entry.get("method", "full"),
            params=entry.get("params"),
            kwargs=copy.deepcopy(entry.get("kwargs", {})),
            reset_optimizer=entry.get("reset_optimizer", True),
            config={
                k: copy.deepcopy(v)
                for k, v in entry.items()
                if k not in {"type", "name", "method", "params", "kwargs", "reset_optimizer"}
            },
        )
    if phase_type == "data":
        return DataPhaseSpec(
            name=entry.get("name", f"phase_{index}"),
            operation=entry.get("operation", "context"),
            config={
                k: copy.deepcopy(v)
                for k, v in entry.items()
                if k not in {"type", "name", "operation"}
            },
        )
    if phase_type == "analysis":
        return AnalysisPhaseSpec(
            name=entry.get("name", f"phase_{index}"),
            split=entry.get("split", "valid"),
            evaluate_all=entry.get("evaluate_all", False),
            config={
                k: copy.deepcopy(v)
                for k, v in entry.items()
                if k not in {"type", "name", "split", "evaluate_all"}
            },
        )
    if phase_type == "export":
        return ExportPhaseSpec(
            name=entry.get("name", f"phase_{index}"),
            export_kind=entry.get("export_kind", "best_model"),
            config={
                k: copy.deepcopy(v)
                for k, v in entry.items()
                if k not in {"type", "name", "export_kind"}
            },
        )
    raise PhaseSpecValidationError(f"Unsupported explicit phase type '{phase_type}'.")


def _normalize_phase_entry(entry: dict[str, Any], index: int) -> list[PhaseSpec]:
    if entry.get("ls_update") is not None:
        _raise_legacy_ls_update_error()
    if "repeat" in entry:
        return _normalize_repeat_block(entry, index)
    if "type" in entry:
        return [_normalize_explicit_phase(entry, index)]
    if "trainer" not in entry:
        raise PhaseSpecValidationError(
            f"Phase entry {index} must define 'trainer', 'type', or 'repeat'."
        )
    return [_optimizer_spec_from_legacy(entry, index)]


def _normalize_repeat_block(entry: dict[str, Any], index: int) -> list[PhaseSpec]:
    if set(entry.keys()) != {"repeat"}:
        invalid = ", ".join(sorted(key for key in entry.keys() if key != "repeat"))
        raise PhaseSpecValidationError(
            f"Repeat phase entry {index} may only contain the 'repeat' key; got extra keys: {invalid}."
        )

    repeat_cfg = entry["repeat"]
    if not isinstance(repeat_cfg, dict):
        raise PhaseSpecValidationError(f"Repeat phase entry {index} must map to a dictionary.")

    if "times" not in repeat_cfg:
        raise PhaseSpecValidationError(f"Repeat phase entry {index} must define 'times'.")
    times = int(repeat_cfg["times"])
    if times <= 0:
        raise PhaseSpecValidationError(
            f"Repeat phase entry {index} must define a positive 'times' value."
        )

    raw_phases = repeat_cfg.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise PhaseSpecValidationError(
            f"Repeat phase entry {index} must define a non-empty 'phases' list."
        )

    repeat_name = repeat_cfg.get("name", f"repeat_{index}")
    specs: list[PhaseSpec] = []
    for iteration in range(times):
        suffix = f"{repeat_name}_{iteration}"
        for nested_index, nested_entry in enumerate(raw_phases):
            named_entry = _with_repeat_name(nested_entry, nested_index, suffix)
            nested_specs = _normalize_phase_entry(named_entry, nested_index)
            for spec in nested_specs:
                _warn_if_repeat_contains_terminal_phase(spec, repeat_name)
            specs.extend(nested_specs)
    return specs


def normalize_phase_specs(config: dict[str, Any]) -> list[PhaseSpec]:
    raw_phases = copy.deepcopy(config.get("phases"))
    if raw_phases is None:
        raise PhaseSpecValidationError("Training config must contain 'phases'.")

    specs: list[PhaseSpec] = []
    for index, entry in enumerate(raw_phases):
        specs.extend(_normalize_phase_entry(entry, index))

    if not any(spec.kind == "analysis" for spec in specs):
        specs.append(AnalysisPhaseSpec(name="analysis"))
    export_kinds = {spec.export_kind for spec in specs if isinstance(spec, ExportPhaseSpec)}
    if "best_model" not in export_kinds:
        specs.append(ExportPhaseSpec(name="export_best_model", export_kind="best_model"))
    if "run_checkpoint" not in export_kinds:
        specs.append(ExportPhaseSpec(name="export_run_checkpoint", export_kind="run_checkpoint"))
    if "summary" not in export_kinds:
        specs.append(ExportPhaseSpec(name="export_summary", export_kind="summary"))
    return specs


class BasePhase:
    def __init__(
        self,
        *,
        spec: PhaseSpec,
        config: dict[str, Any],
        model_class: type,
        dtype: torch.dtype,
        execution_services: ExecutionServices,
    ):
        self.spec = spec
        self.config = copy.deepcopy(config)
        self.model_class = model_class
        self.dtype = dtype
        self.execution_services = execution_services
        self.device = execution_services.device

    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        raise NotImplementedError

    def replay_context(
        self,
        *,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        logger: logging.Logger,
    ) -> tuple[PhaseContext, ArtifactRegistry]:
        return phase_context, artifacts

    def _ensure_model_artifact(
        self,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
    ) -> ModelArtifact:
        model_artifact = artifacts.get("model")
        if model_artifact is not None:
            return model_artifact
        model = self.model_class(
            self.config["model"],
            phase_context.train_md,
            dtype=self.dtype,
            device=self.device,
        ).to(self.device)
        if self.config.get("data", {}).get("double_precision", False):
            model = model.double()
        model_artifact = ModelArtifact(
            model=model,
            config=copy.deepcopy(self.config),
            train_md=copy.deepcopy(phase_context.train_md or {}),
            valid_md=copy.deepcopy(phase_context.valid_md or {}),
            dtype=self.dtype,
        )
        artifacts.put("model", model_artifact)
        return model_artifact

    def _ensure_history_artifact(self, artifacts: ArtifactRegistry) -> TrainingHistoryArtifact:
        history = artifacts.get("history")
        if history is None:
            history = TrainingHistoryArtifact()
            artifacts.put("history", history)
        return history

    def _ensure_export_artifact(self, artifacts: ArtifactRegistry) -> ExportArtifact:
        export_artifact = artifacts.get("exports")
        if export_artifact is None:
            export_artifact = ExportArtifact()
            artifacts.put("exports", export_artifact)
        return export_artifact

    def _select_export_model_state_dict(
        self,
        model_artifact: ModelArtifact,
        history: TrainingHistoryArtifact | None = None,
    ) -> dict[str, Any]:
        if history is not None and history.best_model_state_dict is not None:
            return copy.deepcopy(history.best_model_state_dict)
        return copy.deepcopy(model_artifact.model.state_dict())

    def _prediction_settings(self) -> tuple[str, dict[str, Any]]:
        phases = self.config.get("phases", [])
        for phase_cfg in phases:
            if not isinstance(phase_cfg, dict):
                continue
            phase_type = phase_cfg.get("type")
            trainer_name = phase_cfg.get("trainer")
            if phase_type == "optimizer" or trainer_name in {"NODE", "Weak", "Linear", "OneStep"}:
                return (
                    phase_cfg.get("ode_method", "dopri5"),
                    copy.deepcopy(phase_cfg.get("ode_args", {})),
                )
        return "dopri5", {}

    def _select_plot_sample(self, phase_context: PhaseContext):
        if phase_context.valid_set is not None and _dataset_len(phase_context.valid_set) > 0:
            return phase_context.valid_set[0], phase_context.valid_md
        if phase_context.train_set is not None and _dataset_len(phase_context.train_set) > 0:
            return phase_context.train_set[0], phase_context.train_md
        return None, None

    def _inverse_transform_tensor(
        self,
        tensor: torch.Tensor,
        transform_config: dict[str, Any] | list[dict[str, Any]] | None,
        transform_state: dict[str, Any] | None,
    ) -> np.ndarray:
        module = build_transform_module(transform_config, transform_state)
        restored = module.inverse_batch([tensor.detach().cpu()])[0]
        return restored.detach().cpu().numpy()

    def _export_prediction_plot(
        self,
        *,
        model_artifact: ModelArtifact,
        history: TrainingHistoryArtifact,
        phase_context: PhaseContext,
        run_name: str,
        logger: logging.Logger,
        state_dict: dict[str, Any] | None = None,
    ) -> str | None:
        if bool(getattr(model_artifact.model, "GRAPH", False)):
            logger.info("Skipping per-run prediction plot for graph model '%s'.", run_name)
            return None

        sample, sample_md = self._select_plot_sample(phase_context)
        if sample is None or sample_md is None:
            logger.info("Skipping per-run prediction plot for '%s': no sample available.", run_name)
            return None

        runtime = cast(Any, batch_to_runtime(sample))
        if getattr(runtime, "x", None) is None or getattr(runtime, "t", None) is None:
            logger.info(
                "Skipping per-run prediction plot for '%s': sample has no regular trajectory payload.",
                run_name,
            )
            return None

        plot_cfg = copy.deepcopy(self.config.get("plotting", {}))
        if not plot_cfg.get("prediction", True):
            logger.info(
                "Skipping per-run prediction plot for '%s': plotting.prediction is false.", run_name
            )
            return None

        xidx = plot_cfg.get("xidx")
        uidx = plot_cfg.get("uidx")
        max_state_dims = int(plot_cfg.get("max_state_dims", 16))
        max_control_dims = int(plot_cfg.get("max_control_dims", 8))
        raw_state_dims = sample_md.get("n_state_features")
        raw_control_dims = sample_md.get("n_control_features")
        if xidx is None and raw_state_dims is not None and int(raw_state_dims) > max_state_dims:
            logger.info(
                "Skipping per-run prediction plot for '%s': raw state dimension %d exceeds max_state_dims=%d.",
                run_name,
                int(raw_state_dims),
                max_state_dims,
            )
            return None
        if (
            uidx is None
            and raw_control_dims is not None
            and int(raw_control_dims) > max_control_dims
        ):
            logger.info(
                "Skipping per-run prediction plot for '%s': raw control dimension %d exceeds max_control_dims=%d.",
                run_name,
                int(raw_control_dims),
                max_control_dims,
            )
            return None

        time_tensor = runtime.t[0] if runtime.t.ndim > 1 else runtime.t
        state_tensor = runtime.x[0] if runtime.x.ndim > 2 else runtime.x
        control_tensor = None
        if getattr(runtime, "u", None) is not None:
            control_tensor = runtime.u[0] if runtime.u.ndim > 2 else runtime.u

        model = model_artifact.model
        export_state = (
            copy.deepcopy(state_dict)
            if state_dict is not None
            else self._select_export_model_state_dict(model_artifact, history)
        )
        original_state = copy.deepcopy(model.state_dict())
        ode_method, ode_args = self._prediction_settings()

        try:
            model.load_state_dict(export_state)
            model.eval()
            with torch.no_grad():
                prediction = cast(Any, model).predict(
                    runtime.initial_state(),
                    runtime,
                    runtime.t,
                    method=ode_method,
                    **ode_args,
                )
        finally:
            model.load_state_dict(original_state)

        if prediction.ndim > 2 and prediction.shape[0] == 1:
            prediction = prediction[0]

        truth_np = self._inverse_transform_tensor(
            state_tensor,
            self.config.get("transform_x"),
            sample_md.get("transform_x_state"),
        )
        pred_np = self._inverse_transform_tensor(
            prediction,
            self.config.get("transform_x"),
            sample_md.get("transform_x_state"),
        )
        control_np = None
        if control_tensor is not None:
            control_np = self._inverse_transform_tensor(
                control_tensor,
                self.config.get("transform_u"),
                sample_md.get("transform_u_state"),
            )

        time_np = cast(torch.Tensor, time_tensor).detach().cpu().numpy()
        plot_len = min(len(time_np), truth_np.shape[0], pred_np.shape[0])
        if control_np is not None:
            plot_len = min(plot_len, control_np.shape[0])

        wrote_plot = _safe_plot(
            logger,
            label=f"prediction plot '{run_name}'",
            fn=lambda: plot_trajectory(
                np.array([truth_np[:plot_len], pred_np[:plot_len]]),
                time_np[:plot_len],
                model_name=run_name,
                us=None if control_np is None else control_np[:plot_len],
                labels=["Truth", "Prediction"],
                ifclose=True,
                prefix=self.execution_services.checkpoint_prefix,
                xidx=xidx,
                uidx=uidx,
            ),
        )
        if not wrote_plot:
            return None
        return self.execution_services.checkpoint_file(f"{run_name}_prediction.png")

    def _write_progress_plots(
        self,
        *,
        model_artifact: ModelArtifact,
        history: TrainingHistoryArtifact,
        phase_context: PhaseContext,
        run_name: str,
        logger: logging.Logger,
        hist_entries: list[dict[str, list]],
        crit_name: str | None,
        state_dict: dict[str, Any] | None = None,
    ) -> None:
        _safe_plot(
            logger,
            label=f"history plot '{run_name}'",
            fn=lambda: plot_hist(
                copy.deepcopy(hist_entries),
                copy.deepcopy(history.crit),
                crit_name,
                run_name,
                ifclose=True,
                prefix=self.execution_services.checkpoint_prefix,
            ),
        )
        self._export_prediction_plot(
            model_artifact=model_artifact,
            history=history,
            phase_context=phase_context,
            run_name=run_name,
            logger=logger,
            state_dict=state_dict,
        )

    def _build_phase_record(
        self,
        trainer_state: TrainerState,
        metrics: dict[str, float],
        artifacts: ArtifactRegistry,
        *,
        started_epoch: int,
    ) -> PhaseRecord:
        return PhaseRecord(
            name=self.spec.name,
            kind=self.spec.kind,
            started_epoch=started_epoch,
            completed_epoch=trainer_state.epoch,
            metrics=copy.deepcopy(metrics),
            artifact_keys=sorted(list(artifacts.keys())),
        )


class BaseOptimizerPhase(BasePhase):
    def _build_optimizer_artifact(
        self,
        model: torch.nn.Module,
        artifacts: ArtifactRegistry,
    ) -> OptimizerStateArtifact:
        spec = cast(OptimizerPhaseSpec, self.spec)
        phase_cfg = spec.config
        lr = float(phase_cfg.get("learning_rate", 1e-3))
        gamma = float(phase_cfg.get("decay_rate", 0.999))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        prior = artifacts.get("optimizer_state")
        if isinstance(prior, OptimizerStateArtifact) and not spec.reset_optimizer:
            try:
                optimizer.load_state_dict(prior.optimizer.state_dict())
            except ValueError:
                pass
            else:
                # Reuse moments/step counters while still honoring the new phase lr.
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
        schedulers = [
            make_scheduler(torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma))
        ]
        criteria, criteria_weights, criteria_names = self._build_criteria()
        artifact = OptimizerStateArtifact(
            optimizer=optimizer,
            schedulers=schedulers,
            criteria=criteria,
            criteria_weights=criteria_weights,
            criteria_names=criteria_names,
            owner_phase=self.spec.name,
        )
        artifacts.put("optimizer_state", artifact)
        return artifact

    def _build_criteria(self) -> tuple[list[torch.nn.Module], list[float], list[str]]:
        crit_dict = copy.deepcopy(self.config.get("criterion", {}))
        crit_dict.update(copy.deepcopy(self.spec.config.get("criterion", {})))

        criteria: list[torch.nn.Module] = []
        names = ["dynamics"]
        weights: list[float] = []

        if "dynamics" in crit_dict:
            crit_cfg = crit_dict["dynamics"]
            loss_class = _require_loss_class(crit_cfg.get("type", "mse"))
            criteria.append(loss_class(**crit_cfg.get("params", {})))
            weights.append(crit_cfg.get("weight", 1.0))
        else:
            criteria.append(torch.nn.MSELoss(reduction="mean"))
            weights.append(1.0)

        if "recon" in crit_dict:
            crit_cfg = crit_dict["recon"]
            loss_class = _require_loss_class(crit_cfg.get("type", "mse"))
            criteria.append(loss_class(**crit_cfg.get("params", {})))
            weights.append(crit_cfg.get("weight", 1.0))
            names.append("recon")

        for key in crit_dict:
            if key in {"dynamics", "recon"}:
                continue
            crit_cfg = crit_dict[key]
            loss_class = _require_loss_class(crit_cfg.get("type", "mse"))
            criteria.append(loss_class(**crit_cfg.get("params", {})))
            weights.append(crit_cfg.get("weight", 1.0))
            names.append(key)

        prediction_cfg = copy.deepcopy(self.config.get("prediction_criterion", {}))
        prediction_cfg.update(copy.deepcopy(self.spec.config.get("prediction_criterion", {})))
        if prediction_cfg:
            key = prediction_cfg.get("type", "mse")
            loss_class = _require_loss_class(key)
            criteria.append(loss_class(**prediction_cfg.get("params", {})))
            names.append(key)
        else:
            criteria.append(criteria[0])
            names.append(str(names[0]))

        return criteria, weights, names

    @staticmethod
    def _aggregate_losses(loss_list: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        if not loss_list:
            raise ValueError("loss_list must not be empty")
        total = torch.zeros_like(loss_list[0])
        for loss, weight in zip(loss_list, weights, strict=False):
            total = total + loss * weight
        return total

    @staticmethod
    def _average_loss_lists(loss_lists: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        if not loss_lists:
            raise ValueError("loss_lists must not be empty")
        n_items = len(loss_lists)
        averaged: list[torch.Tensor] = []
        for loss_terms in zip(*loss_lists, strict=False):
            total = torch.zeros_like(loss_terms[0])
            for loss in loss_terms:
                total = total + loss
            averaged.append(total / n_items)
        return averaged

    def _additional_criteria_evaluation(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        x_hat,
        predictions,
        batch: TrainerBatch | RuntimeBatch,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> list[torch.Tensor]:
        loss_list: list[torch.Tensor] = []
        if len(optimizer_state.criteria_weights) < 2:
            return loss_list

        runtime = cast(Any, batch)
        if not (hasattr(runtime, "is_uniform_length") and hasattr(runtime, "iter_series")):
            runtime = cast(Any, batch_to_runtime(cast(TrainerBatch, batch)))

        if runtime is None:
            raise ValueError("Runtime conversion failed.")

        if getattr(runtime, "is_uniform_length", False) is None:
            raise ValueError("Runtime length metadata is unavailable.")

        if hasattr(runtime, "is_uniform_length") and hasattr(runtime, "iter_series"):
            pass
        else:
            runtime = cast(Any, batch_to_runtime(cast(TrainerBatch, batch)))

        if hasattr(runtime, "is_uniform_length") and not runtime.is_uniform_length:
            if x_hat is not None or predictions is not None:
                raise ValueError(
                    "Ragged runtime collections require per-sample criteria evaluation."
                )
            nested_losses = [
                self._additional_criteria_evaluation(
                    model,
                    optimizer_state,
                    None,
                    None,
                    item,
                    ode_method,
                    ode_args,
                )
                for item in runtime.iter_series()
            ]
            return self._average_loss_lists(nested_losses)

        if optimizer_state.criteria_names[1] == "recon":
            if x_hat is None:
                latent = cast(Any, model).encoder(runtime)
                x_hat = cast(Any, model).decoder(latent, runtime)
            recon_loss = optimizer_state.criteria[1](runtime.x, x_hat.view(*runtime.x.shape))
            loss_list.append(recon_loss)
            next_index = 2
        else:
            next_index = 1

        preds = predictions
        if preds is None and len(optimizer_state.criteria) - 1 > next_index:
            init_states = runtime.x[:, 0, :]
            ts = runtime.t.to(self.device)
            preds = cast(Any, model).predict(
                init_states, runtime, ts, method=ode_method, **ode_args
            )

        for idx in range(next_index, len(optimizer_state.criteria) - 1):
            loss_list.append(optimizer_state.criteria[idx](preds, runtime.x))

        return loss_list

    def _evaluate_loader(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        dataloader: DataLoader,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> tuple[float, list[float]]:
        model.eval()
        loader = _require_loader(cast(DataLoader[TrainerBatch] | None, dataloader))
        total = 0.0
        items = [0.0 for _ in optimizer_state.criteria_weights]
        with torch.no_grad():
            for batch in loader:
                losses = self._compute_losses(model, optimizer_state, batch, ode_method, ode_args)
                agg = self._aggregate_losses(losses, optimizer_state.criteria_weights)
                total += agg.item()
                items = [acc + value.item() for acc, value in zip(items, losses, strict=False)]
        return total / len(loader), [value / len(loader) for value in items]

    def _evaluate_prediction_criterion_single(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        truth: TrainerBatch,
        *,
        method: str,
        ode_args: dict[str, Any] | None = None,
    ) -> float:
        runtime = batch_to_runtime(truth)
        with torch.no_grad():
            runtime_any = cast(Any, runtime)
            x_truth = runtime_any.x
            x0 = runtime_any.x[:, 0, :]
            ts = runtime_any.t
            x_pred = cast(Any, model).predict(
                x0, runtime_any, ts, method=method, **(ode_args or {})
            )
            return optimizer_state.criteria[-1](x_pred, x_truth).item()

    def _evaluate_prediction_criterion(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        dataset,
        *,
        method: str,
        ode_args: dict[str, Any] | None = None,
        evaluate_all: bool = False,
    ) -> float:
        if evaluate_all:
            samples = dataset
        else:
            samples = [random.choice(dataset)]
        values = [
            self._evaluate_prediction_criterion_single(
                model,
                optimizer_state,
                item,
                method=method,
                ode_args=ode_args,
            )
            for item in samples
        ]
        return sum(values) / len(values)

    def _update_prediction_history(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        phase_context: PhaseContext,
        history: TrainingHistoryArtifact,
        *,
        epoch: int,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> None:
        train_crit = self._evaluate_prediction_criterion(
            model,
            optimizer_state,
            cast(list[TrainerBatch], phase_context.train_set or []),
            method=ode_method,
            ode_args=ode_args,
        )
        valid_crit = self._evaluate_prediction_criterion(
            model,
            optimizer_state,
            cast(list[TrainerBatch], phase_context.valid_set or []),
            method=ode_method,
            ode_args=ode_args,
        )
        history.crit.append([epoch, train_crit, valid_crit])

    def _maybe_update_best(
        self,
        model: torch.nn.Module,
        trainer_state: TrainerState,
        history: TrainingHistoryArtifact,
        local_hist: dict[str, list],
    ) -> bool:
        if local_hist["valid_total"][-1] < trainer_state.best_loss["valid_total"]:
            trainer_state.best_loss = {key: value[-1] for key, value in local_hist.items() if value}
            trainer_state.convergence_epoch = local_hist["epoch"][-1] + 1
            history.best_loss = copy.deepcopy(trainer_state.best_loss)
            history.best_model_state_dict = copy.deepcopy(model.state_dict())
            history.convergence_epoch = trainer_state.convergence_epoch
            return True
        return False

    def _compute_losses(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        batch: TrainerBatch,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> list[torch.Tensor]:
        raise NotImplementedError

    def _train_epoch(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        phase_context: PhaseContext,
        ode_method: str,
        ode_args: dict[str, Any],
        phase_logger: logging.Logger,
    ) -> tuple[float, list[float], bool]:
        model.train()
        train_loader = _require_loader(phase_context.train_loader)
        total = 0.0
        items = [0.0 for _ in optimizer_state.criteria_weights]
        for batch in train_loader:
            optimizer_state.optimizer.zero_grad(set_to_none=True)
            losses = self._compute_losses(model, optimizer_state, batch, ode_method, ode_args)
            agg = self._aggregate_losses(losses, optimizer_state.criteria_weights)
            agg.backward()
            optimizer_state.optimizer.step()
            total += agg.item()
            items = [acc + value.item() for acc, value in zip(items, losses, strict=False)]

        avg_total = total / len(train_loader)
        avg_items = [value / len(train_loader) for value in items]

        converged = False
        for scheduler in optimizer_state.schedulers:
            flag, changed = scheduler.step(eploss=avg_total)
            converged = converged or flag
            if changed:
                phase_logger.info("Resetting best loss after scheduler transition.")
        min_lr = float(self.spec.config.get("min_learning_rate", 1e-6))
        if min_lr > 0.0:
            for param_group in optimizer_state.optimizer.param_groups:
                if param_group["lr"] < min_lr:
                    param_group["lr"] = min_lr
        return avg_total, avg_items, converged

    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        started_epoch = trainer_state.epoch
        model_artifact = self._ensure_model_artifact(phase_context, artifacts)
        history = self._ensure_history_artifact(artifacts)
        optimizer_state = self._build_optimizer_artifact(model_artifact.model, artifacts)
        self._customize_optimizer_artifact(
            optimizer_state, model_artifact.model, phase_context, logger
        )

        n_epochs = int(self.spec.config.get("n_epochs", 1))
        save_interval = int(self.spec.config.get("save_interval", 10))
        log_interval = int(self.spec.config.get("log_interval", save_interval))
        ode_method = self.spec.config.get("ode_method", "dopri5")
        ode_args = copy.deepcopy(self.spec.config.get("ode_args", {}))

        local_hist = {"epoch": []}
        local_hist.update({f"train_{name}": [] for name in optimizer_state.criteria_names[:-1]})
        local_hist.update({f"valid_{name}": [] for name in optimizer_state.criteria_names[:-1]})
        local_hist["train_total"] = []
        local_hist["valid_total"] = []

        for local_epoch in range(n_epochs):
            epoch_start = time.time()
            train_total, train_items, converged = self._train_epoch(
                model_artifact.model,
                optimizer_state,
                phase_context,
                ode_method,
                ode_args,
                logger,
            )
            valid_total, valid_items = self._evaluate_loader(
                model_artifact.model,
                optimizer_state,
                _require_loader(phase_context.valid_loader),
                ode_method,
                ode_args,
            )
            trainer_state.epoch += 1
            history.epoch_times.append(time.time() - epoch_start)
            trainer_state.converged = trainer_state.converged or converged

            local_hist["epoch"].append(trainer_state.epoch - 1)
            for name, train_value, valid_value in zip(
                optimizer_state.criteria_names[:-1], train_items, valid_items, strict=False
            ):
                local_hist[f"train_{name}"].append(train_value)
                local_hist[f"valid_{name}"].append(valid_value)
            local_hist["train_total"].append(train_total)
            local_hist["valid_total"].append(valid_total)
            self._maybe_update_best(model_artifact.model, trainer_state, history, local_hist)

            should_log = (
                local_epoch == 0
                or trainer_state.converged
                or local_epoch == n_epochs - 1
                or (log_interval > 0 and trainer_state.epoch % log_interval == 0)
            )
            if should_log:
                current_lr = float(optimizer_state.optimizer.param_groups[0]["lr"])
                logger.info(
                    "Epoch %d/%d | train_total=%.4e | valid_total=%.4e | best_valid=%.4e | lr=%.2e",
                    local_epoch + 1,
                    n_epochs,
                    train_total,
                    valid_total,
                    trainer_state.best_loss["valid_total"],
                    current_lr,
                )

            if (
                trainer_state.epoch % save_interval == 0
                or trainer_state.converged
                or local_epoch == n_epochs - 1
            ):
                self._update_prediction_history(
                    model_artifact.model,
                    optimizer_state,
                    phase_context,
                    history,
                    epoch=trainer_state.epoch - 1,
                    ode_method=ode_method,
                    ode_args=ode_args,
                )
                self._write_progress_plots(
                    model_artifact=model_artifact,
                    history=history,
                    phase_context=phase_context,
                    run_name=run_name,
                    logger=logger,
                    hist_entries=[*history.hist, copy.deepcopy(local_hist)],
                    crit_name=optimizer_state.criteria_names[-1],
                    state_dict=model_artifact.model.state_dict(),
                )
                if trainer_state.converged:
                    break

        history.hist.append(local_hist)
        artifacts.put("model", model_artifact)
        artifacts.put("optimizer_state", optimizer_state)
        metrics = copy.deepcopy(trainer_state.best_loss)
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=started_epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=phase_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )

    def _customize_optimizer_artifact(
        self,
        optimizer_state: OptimizerStateArtifact,
        model: torch.nn.Module,
        phase_context: PhaseContext,
        phase_logger: logging.Logger,
    ) -> None:
        return


class NodeOptimizerPhase(BaseOptimizerPhase):
    def _customize_optimizer_artifact(
        self,
        optimizer_state: OptimizerStateArtifact,
        model: torch.nn.Module,
        phase_context: PhaseContext,
        phase_logger: logging.Logger,
    ) -> None:
        sweep_lengths = self.spec.config.get("sweep_lengths", [None])
        epoch_step = self.spec.config.get("sweep_epoch_step", self.spec.config.get("n_epochs", 1))
        sweep_tols = self.spec.config.get("sweep_tols")
        sweep_mode = self.spec.config.get("sweep_mode", "skip")
        optimizer_state.schedulers.append(
            make_scheduler(
                scheduler_type="sweep",
                sweep_lengths=sweep_lengths,
                sweep_tols=sweep_tols,
                epoch_step=epoch_step,
                mode=sweep_mode,
            )
        )

    def _compute_losses(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        batch: TrainerBatch,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> list[torch.Tensor]:
        batch_any = cast(Any, batch)
        if getattr(batch_any, "is_ragged", False):
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch_any.iter_single_batches()
                ]
            )

        num_steps = optimizer_state.schedulers[1].get_length()
        if num_steps is None:
            runtime = cast(Any, batch_to_runtime(batch))
            num_steps = runtime.x.size(1)

        chop_mode = self.spec.config.get("chop_mode", "initial")
        chop_step = self.spec.config.get("chop_step", 1.0)
        if chop_mode == "initial":
            if hasattr(batch_any, "truncate"):
                runtime_batch = batch_any.truncate(num_steps).to(self.device)
                runtime = cast(Any, batch_to_runtime(runtime_batch))
            else:
                runtime = cast(Any, batch_to_runtime(batch)).truncate(num_steps).to(self.device)
        else:
            step = _determine_chop_step(num_steps, chop_step)
            if hasattr(batch_any, "window"):
                runtime_batch = batch_any.window(num_steps, step).to(self.device)
                runtime = cast(Any, batch_to_runtime(runtime_batch))
            else:
                runtime = cast(Any, batch_to_runtime(batch)).unfold(num_steps, step).to(self.device)

        init_states = runtime.x[:, 0, :]
        ts = runtime.t[:, :num_steps].to(self.device)
        predictions = cast(Any, model).predict(
            init_states, runtime, ts, method=ode_method, **ode_args
        )
        losses = [optimizer_state.criteria[0](predictions, runtime.x)]
        losses.extend(
            self._additional_criteria_evaluation(
                model,
                optimizer_state,
                None,
                predictions,
                runtime,
                ode_method,
                ode_args,
            )
        )
        return losses


class WeakFormOptimizerPhase(BaseOptimizerPhase):
    def _customize_optimizer_artifact(
        self,
        optimizer_state: OptimizerStateArtifact,
        model: torch.nn.Module,
        phase_context: PhaseContext,
        phase_logger: logging.Logger,
    ) -> None:
        params = self.spec.config["weak_form_params"]
        dtype = next(model.parameters()).dtype
        train_md = _require_metadata(phase_context.train_md)
        C, D = generate_weak_weights(
            dt=train_md["dt_and_n_steps"][0][0],
            n_integration_points=params["N"],
            poly_order=params["ordpol"],
            int_rule_order=params["ordint"],
        )
        optimizer_state._weak_C = torch.tensor(C.T, dtype=dtype, device=self.device)
        optimizer_state._weak_D = torch.tensor(D.T, dtype=dtype, device=self.device)
        optimizer_state._weak_N = params["N"]
        optimizer_state._weak_dN = params["dN"]

    def _compute_losses(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        batch: TrainerBatch,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> list[torch.Tensor]:
        batch_any = cast(Any, batch)
        if getattr(batch_any, "is_ragged", False):
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch_any.iter_single_batches()
                ]
            )

        runtime_batch = batch_any.to(self.device)
        runtime = cast(Any, batch_to_runtime(runtime_batch))
        latent = cast(Any, model).encoder(runtime)
        latent_dot = cast(Any, model).dynamics(latent, runtime)
        x_hat = cast(Any, model).decoder(latent, runtime)
        z_windows = latent.unfold(1, optimizer_state._weak_N, optimizer_state._weak_dN)
        z_dot_windows = latent_dot.unfold(1, optimizer_state._weak_N, optimizer_state._weak_dN)
        true_weak = z_windows @ optimizer_state._weak_C
        pred_weak = z_dot_windows @ optimizer_state._weak_D
        losses = [optimizer_state.criteria[0](pred_weak, true_weak)]
        losses.extend(
            self._additional_criteria_evaluation(
                model,
                optimizer_state,
                x_hat,
                None,
                runtime_batch,
                ode_method,
                ode_args,
            )
        )
        return losses


class LinearRegressionPhase(BaseOptimizerPhase):
    def _create_ls_updater(
        self,
        model: torch.nn.Module,
        phase_context: PhaseContext,
    ) -> LSUpdater:
        train_md = _require_metadata(phase_context.train_md)
        return LSUpdater(
            method=self.spec.config.get("method", "full"),
            model=model,
            dt=train_md["dt_and_n_steps"][0][0],
            params=self.spec.config.get("params"),
            **copy.deepcopy(self.spec.config.get("kwargs", {})),
        )

    def _compute_losses(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        batch: TrainerBatch,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> list[torch.Tensor]:
        batch_any = cast(Any, batch)
        if getattr(batch_any, "is_ragged", False):
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch_any.iter_single_batches()
                ]
            )
        runtime_batch = batch_any.to(self.device)
        updater = optimizer_state._linear_updater
        if updater is None:
            raise ValueError("Linear updater is not initialized.")
        losses = [updater.eval_batch(model, runtime_batch, optimizer_state.criteria[0])]
        losses.extend(
            self._additional_criteria_evaluation(
                model,
                optimizer_state,
                None,
                None,
                batch_to_runtime(runtime_batch),
                ode_method,
                ode_args,
            )
        )
        return losses

    def _customize_optimizer_artifact(
        self,
        optimizer_state: OptimizerStateArtifact,
        model: torch.nn.Module,
        phase_context: PhaseContext,
        phase_logger: logging.Logger,
    ) -> None:
        optimizer_state._linear_updater = self._create_ls_updater(model, phase_context)

    def _train_epoch(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        phase_context: PhaseContext,
        ode_method: str,
        ode_args: dict[str, Any],
        phase_logger: logging.Logger,
    ) -> tuple[float, list[float], bool]:
        updater = optimizer_state._linear_updater
        if updater is None:
            raise ValueError("Linear updater is not initialized.")
        avg_loss, _ = updater.update(model, _require_loader(phase_context.train_loader))
        items = [float(avg_loss)] + [0.0] * (len(optimizer_state.criteria_weights) - 1)
        return float(avg_loss), items, False


class OneStepOptimizerPhase(BaseOptimizerPhase):
    def _customize_optimizer_artifact(
        self,
        optimizer_state: OptimizerStateArtifact,
        model: torch.nn.Module,
        phase_context: PhaseContext,
        phase_logger: logging.Logger,
    ) -> None:
        del model, phase_logger
        train_md = _require_metadata(phase_context.train_md)
        optimizer_state._one_step_dt = float(train_md["dt_and_n_steps"][0][0])
        optimizer_state._one_step_kwargs = {}
        kwargs_cfg = self.spec.config.get("kwargs", {})
        if isinstance(kwargs_cfg, dict):
            optimizer_state._one_step_kwargs.update(copy.deepcopy(kwargs_cfg))
        if "order" in self.spec.config:
            optimizer_state._one_step_kwargs["order"] = self.spec.config["order"]

    def _compute_losses(
        self,
        model: torch.nn.Module,
        optimizer_state: OptimizerStateArtifact,
        batch: TrainerBatch,
        ode_method: str,
        ode_args: dict[str, Any],
    ) -> list[torch.Tensor]:
        batch_any = cast(Any, batch)
        if getattr(batch_any, "is_ragged", False):
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch_any.iter_single_batches()
                ]
            )

        dt = optimizer_state._one_step_dt
        if dt is None:
            raise ValueError("One-step target time step is not initialized.")

        runtime_batch = batch_any.to(self.device)
        if getattr(model, "CONT", False):
            predictions, targets = _comp_linear_eval_ct(
                model,
                runtime_batch,
                dt=dt,
                **optimizer_state._one_step_kwargs,
            )
        else:
            predictions, targets = _comp_linear_eval_dt(
                model,
                runtime_batch,
                dt=dt,
                **optimizer_state._one_step_kwargs,
            )
        losses = [optimizer_state.criteria[0](predictions, targets)]
        losses.extend(
            self._additional_criteria_evaluation(
                model,
                optimizer_state,
                None,
                None,
                runtime_batch,
                ode_method,
                ode_args,
            )
        )
        return losses


class LinearSolvePhase(BasePhase):
    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        spec = cast(LinearSolvePhaseSpec, self.spec)
        started_epoch = trainer_state.epoch
        model_artifact = self._ensure_model_artifact(phase_context, artifacts)
        self._ensure_history_artifact(artifacts)
        optimizer_state = artifacts.get("optimizer_state")
        train_md = _require_metadata(phase_context.train_md)
        updater = LSUpdater(
            method=spec.method,
            model=model_artifact.model,
            dt=train_md["dt_and_n_steps"][0][0],
            params=spec.params,
            **spec.kwargs,
        )
        loss, params = updater.update(
            model_artifact.model, _require_loader(phase_context.train_loader)
        )
        updated_names: list[str] = []
        if isinstance(optimizer_state, OptimizerStateArtifact) and spec.reset_optimizer:
            param_to_name = {param: name for name, param in model_artifact.model.named_parameters()}
            for param in params:
                updated_names.append(param_to_name.get(param, "<unnamed>"))
                optimizer_state.optimizer.state.pop(param, None)
        report = artifacts.get("linear_solve_report")
        if report is None:
            report = LinearSolveReportArtifact()
            artifacts.put("linear_solve_report", report)
        report.records.append(
            LinearSolveRecord(
                phase_name=self.spec.name,
                method=updater.method,
                loss=float(loss),
                updated_parameters=updated_names,
            )
        )
        metrics = {"linear_solve_loss": float(loss)}
        if isinstance(optimizer_state, OptimizerStateArtifact):
            valid_total = 0.0
            valid_loader = _require_loader(phase_context.valid_loader)
            with torch.no_grad():
                for batch in valid_loader:
                    valid_total += updater.eval_batch(
                        model_artifact.model, batch.to(self.device), optimizer_state.criteria[0]
                    ).item()
            valid_total /= len(valid_loader)
            metrics["valid_total"] = valid_total
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=started_epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=phase_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )


class ContextDataPhase(BasePhase):
    @staticmethod
    def _series_signal_array(series: TrainerBatch) -> np.ndarray:
        if isinstance(series, RegularSeries):
            return series.state.detach().cpu().numpy()
        if isinstance(series, GraphSeries):
            return series.node_state.detach().cpu().numpy()
        raise TypeError(f"Unsupported dataset item type '{type(series)}' for smoothing.")

    @staticmethod
    def _format_metrics_for_log(metrics: dict[str, float]) -> str:
        split_metrics: dict[str, dict[str, float]] = {"train": {}, "valid": {}}
        other_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if key.startswith("train_"):
                split_metrics["train"][key.removeprefix("train_")] = value
            elif key.startswith("valid_"):
                split_metrics["valid"][key.removeprefix("valid_")] = value
            else:
                other_metrics[key] = value

        metric_order = [
            "size",
            "delta_rmse",
            "delta_mae",
            "delta_max_abs",
            "delta_rel_rmse",
            "roughness_before",
            "roughness_after",
            "roughness_delta",
            "roughness_ratio",
        ]
        metric_names = [
            name
            for name in metric_order
            if name in split_metrics["train"] or name in split_metrics["valid"]
        ]
        extra_metric_names = sorted(
            (set(split_metrics["train"]) | set(split_metrics["valid"])) - set(metric_names)
        )
        metric_names.extend(extra_metric_names)

        lines: list[str] = []
        if metric_names:
            metric_width = max(len("metric"), *(len(name) for name in metric_names))
            value_width = 14
            header = (
                f"{'metric':<{metric_width}}  {'train':>{value_width}}  {'valid':>{value_width}}"
            )
            separator = f"{'-' * metric_width}  {'-' * value_width}  {'-' * value_width}"
            lines.extend([header, separator])
            for name in metric_names:
                train_value = split_metrics["train"].get(name)
                valid_value = split_metrics["valid"].get(name)
                train_text = "-" if train_value is None else f"{train_value:.4e}"
                valid_text = "-" if valid_value is None else f"{valid_value:.4e}"
                lines.append(
                    f"{name:<{metric_width}}  {train_text:>{value_width}}  {valid_text:>{value_width}}"
                )

        if other_metrics:
            if lines:
                lines.append("")
            lines.extend(f"{key}={value:.4e}" for key, value in sorted(other_metrics.items()))

        return "\n".join(lines)

    def _resolve_smoothing_method(self) -> str:
        spec = cast(DataPhaseSpec, self.spec)
        method = str(spec.config.get("method", "savgol")).lower()
        if method != "savgol":
            raise PhaseSpecValidationError(
                f"Unsupported data smoothing method '{method}'. Expected 'savgol'."
            )
        return method

    def _resolve_smoothing_splits(self) -> tuple[str, ...]:
        spec = cast(DataPhaseSpec, self.spec)
        raw_splits = spec.config.get("splits", ("train", "valid"))
        if isinstance(raw_splits, str):
            splits = (raw_splits,)
        elif isinstance(raw_splits, Sequence):
            splits = tuple(str(split) for split in raw_splits)
        else:
            raise PhaseSpecValidationError(
                f"Data phase '{self.spec.name}' expects 'splits' to be a string or sequence."
            )
        invalid = [split for split in splits if split not in {"train", "valid"}]
        if invalid:
            raise PhaseSpecValidationError(
                f"Data phase '{self.spec.name}' has invalid splits: {', '.join(invalid)}."
            )
        return splits

    def _resolve_savgol_config(self) -> dict[str, Any]:
        cfg = cast(DataPhaseSpec, self.spec).config
        window_length = int(cfg.get("window_length", 7))
        polyorder = int(cfg.get("polyorder", 3))
        deriv = int(cfg.get("deriv", 0))
        delta = float(cfg.get("delta", 1.0))
        mode = str(cfg.get("mode", "interp"))
        cval = float(cfg.get("cval", 0.0))
        if window_length <= 0 or window_length % 2 == 0:
            raise PhaseSpecValidationError(
                f"Data phase '{self.spec.name}' requires an odd, positive window_length."
            )
        if polyorder < 0 or polyorder >= window_length:
            raise PhaseSpecValidationError(
                f"Data phase '{self.spec.name}' requires polyorder < window_length."
            )
        if deriv < 0:
            raise PhaseSpecValidationError(
                f"Data phase '{self.spec.name}' requires deriv to be non-negative."
            )
        return {
            "window_length": window_length,
            "polyorder": polyorder,
            "deriv": deriv,
            "delta": delta,
            "mode": mode,
            "cval": cval,
        }

    def _smooth_tensor(
        self,
        tensor: torch.Tensor,
        *,
        method: str,
        denoise_cfg: dict[str, Any],
    ) -> torch.Tensor:
        if tensor.shape[0] < denoise_cfg["window_length"]:
            raise PhaseSpecValidationError(
                f"Data phase '{self.spec.name}' requires every selected trajectory to have at least "
                f"{denoise_cfg['window_length']} steps."
            )
        return cast(torch.Tensor, denoise(tensor, method=method, axis=0, **denoise_cfg))

    def _smooth_series(
        self,
        series: TrainerBatch,
        *,
        method: str,
        denoise_cfg: dict[str, Any],
    ) -> TrainerBatch:
        if isinstance(series, RegularSeries):
            return series.with_state(
                self._smooth_tensor(series.state, method=method, denoise_cfg=denoise_cfg)
            )
        if isinstance(series, GraphSeries):
            return replace(
                series,
                node_state=self._smooth_tensor(
                    series.node_state, method=method, denoise_cfg=denoise_cfg
                ),
                meta=dict(series.meta),
            )
        raise TypeError(f"Unsupported dataset item type '{type(series)}' for smoothing.")

    def _append_phase_history(
        self,
        metadata: dict[str, Any] | None,
        *,
        method: str,
        savgol_cfg: dict[str, Any],
        metrics: dict[str, float],
        split: str,
        dataset: Sequence[TrainerBatch],
    ) -> dict[str, Any]:
        updated = {} if metadata is None else copy.deepcopy(metadata)
        history = list(updated.get("data_phase_history", []))
        history.append(
            {
                "phase": self.spec.name,
                "operation": "smooth",
                "method": method,
                "split": split,
                "num_trajectories": len(dataset),
                "window_length": savgol_cfg["window_length"],
                "polyorder": savgol_cfg["polyorder"],
                "deriv": savgol_cfg["deriv"],
                "metrics": copy.deepcopy(metrics),
            }
        )
        updated["data_phase_history"] = history
        return updated

    def _compute_smoothing_metrics(
        self,
        *,
        original: Sequence[TrainerBatch],
        smoothed: Sequence[TrainerBatch],
        split: str,
    ) -> dict[str, float]:
        if len(original) != len(smoothed):
            raise ValueError(
                f"Smoothing metrics require matching dataset lengths for split '{split}'."
            )
        metrics = denoising_metrics(
            original=[self._series_signal_array(series) for series in original],
            denoised=[self._series_signal_array(series) for series in smoothed],
        )
        return {f"{split}_{key}": value for key, value in metrics.items()}

    def _apply_smoothing(
        self,
        *,
        phase_context: PhaseContext,
    ) -> tuple[PhaseContext, dict[str, float]]:
        method = self._resolve_smoothing_method()
        splits = self._resolve_smoothing_splits()
        denoise_cfg = self._resolve_savgol_config()
        metrics: dict[str, float] = {}

        train_set = phase_context.train_set
        train_md = phase_context.train_md
        if "train" in splits:
            if train_set is None:
                raise PhaseSpecValidationError(
                    f"Data phase '{self.spec.name}' cannot smooth the train split because it is missing."
                )
            smoothed_train = [
                self._smooth_series(series, method=method, denoise_cfg=denoise_cfg)
                for series in train_set
            ]
            train_metrics = self._compute_smoothing_metrics(
                original=train_set,
                smoothed=cast(list[TrainerBatch], smoothed_train),
                split="train",
            )
            metrics.update(train_metrics)
            train_set = cast(list[TrainerBatch], smoothed_train)
            train_md = self._append_phase_history(
                phase_context.train_md,
                method=method,
                savgol_cfg=denoise_cfg,
                metrics=train_metrics,
                split="train",
                dataset=train_set,
            )

        valid_set = phase_context.valid_set
        valid_md = phase_context.valid_md
        if "valid" in splits:
            if valid_set is None:
                raise PhaseSpecValidationError(
                    f"Data phase '{self.spec.name}' cannot smooth the valid split because it is missing."
                )
            smoothed_valid = [
                self._smooth_series(series, method=method, denoise_cfg=denoise_cfg)
                for series in valid_set
            ]
            valid_metrics = self._compute_smoothing_metrics(
                original=valid_set,
                smoothed=cast(list[TrainerBatch], smoothed_valid),
                split="valid",
            )
            metrics.update(valid_metrics)
            valid_set = cast(list[TrainerBatch], smoothed_valid)
            valid_md = self._append_phase_history(
                phase_context.valid_md,
                method=method,
                savgol_cfg=denoise_cfg,
                metrics=valid_metrics,
                split="valid",
                dataset=valid_set,
            )

        return (
            PhaseContext(
                train_set=train_set,
                valid_set=valid_set,
                train_loader=(
                    _build_dataset_loader(train_set, config=self.config)
                    if "train" in splits
                    else phase_context.train_loader
                ),
                valid_loader=(
                    _build_dataset_loader(valid_set, config=self.config)
                    if "valid" in splits
                    else phase_context.valid_loader
                ),
                train_md=train_md,
                valid_md=valid_md,
            ),
            metrics,
        )

    def _apply_operation(
        self,
        *,
        phase_context: PhaseContext,
    ) -> tuple[PhaseContext, dict[str, float]]:
        spec = cast(DataPhaseSpec, self.spec)
        if spec.operation == "context":
            metrics = {
                "train_size": float(_dataset_len(phase_context.train_set)),
                "valid_size": float(_dataset_len(phase_context.valid_set)),
            }
            return phase_context, metrics
        if spec.operation == "smooth":
            updated_context, metrics = self._apply_smoothing(phase_context=phase_context)
            metrics.update(
                {
                    "train_size": float(_dataset_len(updated_context.train_set)),
                    "valid_size": float(_dataset_len(updated_context.valid_set)),
                }
            )
            return updated_context, metrics
        raise PhaseSpecValidationError(f"Unsupported data phase operation '{spec.operation}'.")

    def replay_context(
        self,
        *,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        logger: logging.Logger,
    ) -> tuple[PhaseContext, ArtifactRegistry]:
        updated_context, _ = self._apply_operation(phase_context=phase_context)
        logger.info("Replayed completed data phase '%s' (%s).", self.spec.name, self.spec.kind)
        return updated_context, artifacts

    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        del run_name
        updated_context, metrics = self._apply_operation(phase_context=phase_context)
        logger.info(
            "Data phase '%s' completed:\n%s",
            self.spec.name,
            self._format_metrics_for_log(metrics),
        )
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=trainer_state.epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=updated_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )


class ValidationAnalysisPhase(BasePhase):
    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        spec = cast(AnalysisPhaseSpec, self.spec)
        model_artifact = artifacts.require("model", ModelArtifact)
        optimizer_state = artifacts.get("optimizer_state")
        if not isinstance(optimizer_state, OptimizerStateArtifact):
            criteria: list[torch.nn.Module] = [torch.nn.MSELoss(), torch.nn.MSELoss()]
            optimizer_state = OptimizerStateArtifact(
                optimizer=torch.optim.Adam(model_artifact.model.parameters(), lr=1e-3),
                criteria=criteria,
                criteria_weights=[1.0],
                criteria_names=["dynamics", "mse"],
            )
        history = self._ensure_history_artifact(artifacts)
        metric_name = optimizer_state.criteria_names[-1]
        dataset = phase_context.valid_set if spec.split == "valid" else phase_context.train_set
        evaluator = NodeOptimizerPhase(
            spec=OptimizerPhaseSpec(name=f"{spec.name}_eval", trainer="NODE", config={}),
            config=self.config,
            model_class=self.model_class,
            dtype=self.dtype,
            execution_services=self.execution_services,
        )
        ode_method, ode_args = evaluator._prediction_settings()
        value = evaluator._evaluate_prediction_criterion(
            model_artifact.model,
            optimizer_state,
            cast(list[TrainerBatch], dataset or []),
            method=ode_method,
            ode_args=ode_args,
            evaluate_all=spec.evaluate_all,
        )
        evaluation = EvaluationArtifact(
            metrics={metric_name: value},
            split=spec.split,
            criterion_name=metric_name,
        )
        artifacts.put("evaluation", evaluation)
        metrics = {metric_name: value}
        if not history.crit:
            history.crit.append([trainer_state.epoch, value, value])
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=trainer_state.epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=phase_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )


class BestModelExportPhase(BasePhase):
    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        model_artifact = artifacts.require("model", ModelArtifact)
        history = self._ensure_history_artifact(artifacts)
        optimizer_state = artifacts.get("optimizer_state")
        export_state = self._select_export_model_state_dict(model_artifact, history)
        payload = {
            "config": self.config,
            "device": self.device,
            "epoch": trainer_state.epoch,
            "best_loss": copy.deepcopy(trainer_state.best_loss),
            "hist": copy.deepcopy(history.hist),
            "crit": copy.deepcopy(history.crit),
            "epoch_times": copy.deepcopy(history.epoch_times),
            "converged": trainer_state.converged,
            "model_state_dict": export_state,
            "train_md": copy.deepcopy(model_artifact.train_md),
            "valid_md": copy.deepcopy(model_artifact.valid_md),
        }
        if isinstance(optimizer_state, OptimizerStateArtifact):
            payload["criteria_weights"] = copy.deepcopy(optimizer_state.criteria_weights)
            payload["criteria_names"] = copy.deepcopy(optimizer_state.criteria_names)
        output_path = self.execution_services.checkpoint_file(f"{run_name}.pt")
        torch.save(payload, output_path)
        exports = self._ensure_export_artifact(artifacts)
        exports.outputs["best_model"] = output_path
        metrics = {"exports_written": float(len(exports.outputs))}
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=trainer_state.epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=phase_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )


class RunCheckpointExportPhase(BasePhase):
    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        output_path = self.execution_services.checkpoint_file(f"{run_name}_run_checkpoint.pt")
        payload = {
            "schema": "dymad.training.run_checkpoint.v1",
            "trainer_state": trainer_state.checkpoint_payload(),
            "artifacts": artifacts.checkpoint_payload(),
        }
        torch.save(payload, output_path)
        exports = self._ensure_export_artifact(artifacts)
        exports.outputs["run_checkpoint"] = output_path
        metrics = {"exports_written": float(len(exports.outputs))}
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=trainer_state.epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=phase_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )


class SummaryExportPhase(BasePhase):
    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        history = self._ensure_history_artifact(artifacts)
        evaluation = artifacts.get("evaluation")
        crit_epoch = np.array([])
        crits = np.array([])
        if history.crit:
            tmp = np.array(history.crit).T
            crit_epoch, crits = tmp[0], tmp[1:]
        local_hist = (
            history.hist[-1] if history.hist else {"train_total": [np.nan], "valid_total": [np.nan]}
        )
        results = {
            "model_name": run_name,
            "total_training_time": float(sum(history.epoch_times)),
            "avg_epoch_time": float(np.mean(history.epoch_times)) if history.epoch_times else 0.0,
            "final_train_loss": local_hist["train_total"][-1],
            "final_valid_loss": local_hist["valid_total"][-1],
            "best_valid_loss": copy.deepcopy(trainer_state.best_loss),
            "phase_records": copy.deepcopy(trainer_state.phase_records),
            "phase_metrics": {
                record.name: copy.deepcopy(record.metrics) for record in trainer_state.phase_records
            },
            "convergence_epoch": trainer_state.convergence_epoch,
            "hist": copy.deepcopy(history.hist),
            "crit_name": None if evaluation is None else evaluation.criterion_name,
            "crit_epoch": crit_epoch,
            "crits": crits,
        }
        output_path = self.execution_services.checkpoint_file(f"{run_name}_summary.npz")
        np.savez_compressed(output_path, **results)
        exports = self._ensure_export_artifact(artifacts)
        exports.outputs["summary"] = output_path
        if _safe_plot(
            logger,
            label=f"history plot '{run_name}'",
            fn=lambda: plot_hist(
                copy.deepcopy(history.hist),
                copy.deepcopy(history.crit),
                None if evaluation is None else evaluation.criterion_name,
                run_name,
                ifclose=True,
                prefix=self.execution_services.checkpoint_prefix,
            ),
        ):
            exports.outputs["history_plot"] = self.execution_services.checkpoint_file(
                f"{run_name}_history.png"
            )
        prediction_path = self._export_prediction_plot(
            model_artifact=artifacts.require("model", ModelArtifact),
            history=history,
            phase_context=phase_context,
            run_name=run_name,
            logger=logger,
        )
        if prediction_path is not None:
            exports.outputs["prediction_plot"] = prediction_path
        metrics = {"exports_written": float(len(exports.outputs))}
        record = self._build_phase_record(
            trainer_state, metrics, artifacts, started_epoch=trainer_state.epoch
        )
        trainer_state.phase_records.append(record)
        return PhaseResult(
            name=self.spec.name,
            kind=self.spec.kind,
            trainer_state=trainer_state,
            phase_context=phase_context,
            artifacts=artifacts,
            metrics=metrics,
            record=record,
        )


def build_phase(
    spec: PhaseSpec,
    *,
    config: dict[str, Any],
    model_class: type,
    dtype: torch.dtype,
    execution_services: ExecutionServices,
) -> BasePhase:
    if isinstance(spec, OptimizerPhaseSpec):
        if spec.trainer == "NODE":
            return NodeOptimizerPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
        if spec.trainer == "Weak":
            return WeakFormOptimizerPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
        if spec.trainer == "Linear":
            return LinearRegressionPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
        if spec.trainer == "OneStep":
            return OneStepOptimizerPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
    if isinstance(spec, LinearSolvePhaseSpec):
        return LinearSolvePhase(
            spec=spec,
            config=config,
            model_class=model_class,
            dtype=dtype,
            execution_services=execution_services,
        )
    if isinstance(spec, DataPhaseSpec):
        return ContextDataPhase(
            spec=spec,
            config=config,
            model_class=model_class,
            dtype=dtype,
            execution_services=execution_services,
        )
    if isinstance(spec, AnalysisPhaseSpec):
        return ValidationAnalysisPhase(
            spec=spec,
            config=config,
            model_class=model_class,
            dtype=dtype,
            execution_services=execution_services,
        )
    if isinstance(spec, ExportPhaseSpec):
        if spec.export_kind == "best_model":
            return BestModelExportPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
        if spec.export_kind == "run_checkpoint":
            return RunCheckpointExportPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
        if spec.export_kind == "summary":
            return SummaryExportPhase(
                spec=spec,
                config=config,
                model_class=model_class,
                dtype=dtype,
                execution_services=execution_services,
            )
    raise PhaseSpecValidationError(f"Unsupported phase spec '{spec}'.")
