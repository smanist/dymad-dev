from __future__ import annotations

import copy
import logging
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dymad.core.transform_builder import build_transform_module
from dymad.losses import LOSS_MAP
from dymad.numerics import generate_weak_weights
from dymad.training.batch_adapter import RuntimeBatch, TrainerBatch, batch_to_runtime
from dymad.training.execution_services import ExecutionServices
from dymad.training.ls_update import LSUpdater
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

    def __init__(self, name: str, trainer: str, config: dict[str, Any]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "optimizer")
        object.__setattr__(self, "trainer", trainer)
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


def _determine_chop_step(window: int, step: int | float) -> int:
    if isinstance(step, int):
        return step
    if isinstance(step, float):
        stp = int(window * step)
        return min(max(stp, 1), window)
    raise ValueError(f"Invalid step type: {type(step)}. Expected int or float.")


def _normalize_legacy_optimizer_name(trainer: str) -> str:
    if trainer not in {"NODE", "Weak", "Linear"}:
        raise PhaseSpecValidationError(
            f"Unsupported trainer '{trainer}'. Expected one of NODE, Weak, Linear."
        )
    return trainer


def _optimizer_spec_from_legacy(
    entry: dict[str, Any], index: int, suffix: str = ""
) -> OptimizerPhaseSpec:
    cfg = copy.deepcopy(entry)
    trainer = _normalize_legacy_optimizer_name(cfg.pop("trainer"))
    name = cfg.get("name", f"phase_{index}")
    if suffix:
        name = f"{name}_{suffix}"
    return OptimizerPhaseSpec(name=name, trainer=trainer, config=cfg)


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
            config={
                k: copy.deepcopy(v)
                for k, v in entry.items()
                if k not in {"type", "name", "trainer"}
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
            if phase_type == "optimizer" or trainer_name in {"NODE", "Weak", "Linear"}:
                return (
                    phase_cfg.get("ode_method", "dopri5"),
                    copy.deepcopy(phase_cfg.get("ode_args", {})),
                )
        return "dopri5", {}

    def _select_plot_sample(self, phase_context: PhaseContext):
        if phase_context.valid_set is not None and len(phase_context.valid_set) > 0:
            return phase_context.valid_set[0], phase_context.valid_md
        if phase_context.train_set is not None and len(phase_context.train_set) > 0:
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

        runtime = batch_to_runtime(sample)
        if getattr(runtime, "x", None) is None or getattr(runtime, "t", None) is None:
            logger.info(
                "Skipping per-run prediction plot for '%s': sample has no regular trajectory payload.",
                run_name,
            )
            return None

        plot_cfg = copy.deepcopy(self.config.get("plotting", {}))
        if not plot_cfg.get("prediction", True):
            logger.info("Skipping per-run prediction plot for '%s': plotting.prediction is false.", run_name)
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
        if uidx is None and raw_control_dims is not None and int(raw_control_dims) > max_control_dims:
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
                prediction = model.predict(
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

        time_np = time_tensor.detach().cpu().numpy()
        plot_len = min(len(time_np), truth_np.shape[0], pred_np.shape[0])
        if control_np is not None:
            plot_len = min(plot_len, control_np.shape[0])

        plot_trajectory(
            np.array([truth_np[:plot_len], pred_np[:plot_len]]),
            time_np[:plot_len],
            model_name=run_name,
            us=None if control_np is None else control_np[:plot_len],
            labels=["Truth", "Prediction"],
            ifclose=True,
            prefix=self.execution_services.checkpoint_prefix,
            xidx=xidx,
            uidx=uidx,
        )
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
        plot_hist(
            copy.deepcopy(hist_entries),
            copy.deepcopy(history.crit),
            crit_name,
            run_name,
            ifclose=True,
            prefix=self.execution_services.checkpoint_prefix,
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
        phase_cfg = self.spec.config
        lr = float(phase_cfg.get("learning_rate", 1e-3))
        gamma = float(phase_cfg.get("decay_rate", 0.999))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        prior = artifacts.get("optimizer_state")
        if isinstance(prior, OptimizerStateArtifact):
            try:
                optimizer.load_state_dict(prior.optimizer.state_dict())
            except ValueError:
                pass
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
            loss_class = LOSS_MAP[crit_cfg.get("type", "mse")]
            criteria.append(loss_class(**crit_cfg.get("params", {})))
            weights.append(crit_cfg.get("weight", 1.0))
        else:
            criteria.append(torch.nn.MSELoss(reduction="mean"))
            weights.append(1.0)

        if "recon" in crit_dict:
            crit_cfg = crit_dict["recon"]
            loss_class = LOSS_MAP.get(crit_cfg.get("type", "mse"))
            criteria.append(loss_class(**crit_cfg.get("params", {})))
            weights.append(crit_cfg.get("weight", 1.0))
            names.append("recon")

        for key in crit_dict:
            if key in {"dynamics", "recon"}:
                continue
            crit_cfg = crit_dict[key]
            loss_class = LOSS_MAP.get(crit_cfg.get("type", "mse"))
            criteria.append(loss_class(**crit_cfg.get("params", {})))
            weights.append(crit_cfg.get("weight", 1.0))
            names.append(key)

        prediction_cfg = copy.deepcopy(self.config.get("prediction_criterion", {}))
        prediction_cfg.update(copy.deepcopy(self.spec.config.get("prediction_criterion", {})))
        if prediction_cfg:
            key = prediction_cfg.get("type", "mse")
            loss_class = LOSS_MAP.get(key)
            criteria.append(loss_class(**prediction_cfg.get("params", {})))
            names.append(key)
        else:
            criteria.append(criteria[0])
            names.append(str(names[0]))

        return criteria, weights, names

    @staticmethod
    def _aggregate_losses(loss_list: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        total: torch.Tensor | float = 0.0
        for loss, weight in zip(loss_list, weights, strict=False):
            total = total + loss * weight
        return total

    @staticmethod
    def _average_loss_lists(loss_lists: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        if not loss_lists:
            raise ValueError("loss_lists must not be empty")
        n_items = len(loss_lists)
        return [sum(loss_terms) / n_items for loss_terms in zip(*loss_lists, strict=False)]

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

        if hasattr(batch, "is_uniform_length") and hasattr(batch, "iter_series"):
            runtime = batch
        else:
            runtime = batch_to_runtime(batch)

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
                latent = model.encoder(runtime)
                x_hat = model.decoder(latent, runtime)
            recon_loss = optimizer_state.criteria[1](runtime.x, x_hat.view(*runtime.x.shape))
            loss_list.append(recon_loss)
            next_index = 2
        else:
            next_index = 1

        preds = predictions
        if preds is None and len(optimizer_state.criteria) - 1 > next_index:
            init_states = runtime.x[:, 0, :]
            ts = runtime.t.to(self.device)
            preds = model.predict(init_states, runtime, ts, method=ode_method, **ode_args)

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
        total = 0.0
        items = [0.0 for _ in optimizer_state.criteria_weights]
        with torch.no_grad():
            for batch in dataloader:
                losses = self._compute_losses(model, optimizer_state, batch, ode_method, ode_args)
                agg = self._aggregate_losses(losses, optimizer_state.criteria_weights)
                total += agg.item()
                items = [acc + value.item() for acc, value in zip(items, losses, strict=False)]
        return total / len(dataloader), [value / len(dataloader) for value in items]

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
            x_truth = runtime.x
            x0 = runtime.x[:, 0, :]
            ts = runtime.t
            x_pred = model.predict(x0, runtime, ts, method=method, **(ode_args or {}))
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
            phase_context.train_set,
            method=ode_method,
            ode_args=ode_args,
        )
        valid_crit = self._evaluate_prediction_criterion(
            model,
            optimizer_state,
            phase_context.valid_set,
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
        total = 0.0
        items = [0.0 for _ in optimizer_state.criteria_weights]
        for batch in phase_context.train_loader:
            optimizer_state.optimizer.zero_grad(set_to_none=True)
            losses = self._compute_losses(model, optimizer_state, batch, ode_method, ode_args)
            agg = self._aggregate_losses(losses, optimizer_state.criteria_weights)
            agg.backward()
            optimizer_state.optimizer.step()
            total += agg.item()
            items = [acc + value.item() for acc, value in zip(items, losses, strict=False)]

        avg_total = total / len(phase_context.train_loader)
        avg_items = [value / len(phase_context.train_loader) for value in items]

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
                phase_context.valid_loader,
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
        if hasattr(batch, "is_ragged") and batch.is_ragged:
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch.iter_single_batches()
                ]
            )

        num_steps = optimizer_state.schedulers[1].get_length()
        if num_steps is None:
            runtime = batch_to_runtime(batch)
            num_steps = runtime.x.size(1)

        chop_mode = self.spec.config.get("chop_mode", "initial")
        chop_step = self.spec.config.get("chop_step", 1.0)
        if chop_mode == "initial":
            if hasattr(batch, "truncate"):
                runtime_batch = batch.truncate(num_steps).to(self.device)
                runtime = batch_to_runtime(runtime_batch)
            else:
                runtime = batch_to_runtime(batch).truncate(num_steps).to(self.device)
        else:
            step = _determine_chop_step(num_steps, chop_step)
            if hasattr(batch, "window"):
                runtime_batch = batch.window(num_steps, step).to(self.device)
                runtime = batch_to_runtime(runtime_batch)
            else:
                runtime = batch_to_runtime(batch).unfold(num_steps, step).to(self.device)

        init_states = runtime.x[:, 0, :]
        ts = runtime.t[:, :num_steps].to(self.device)
        predictions = model.predict(init_states, runtime, ts, method=ode_method, **ode_args)
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
        C, D = generate_weak_weights(
            dt=phase_context.train_md["dt_and_n_steps"][0][0],
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
        if hasattr(batch, "is_ragged") and batch.is_ragged:
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch.iter_single_batches()
                ]
            )

        runtime_batch = batch.to(self.device)
        runtime = batch_to_runtime(runtime_batch)
        latent = model.encoder(runtime)
        latent_dot = model.dynamics(latent, runtime)
        x_hat = model.decoder(latent, runtime)
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
        return LSUpdater(
            method=self.spec.config.get("method", "full"),
            model=model,
            dt=phase_context.train_md["dt_and_n_steps"][0][0],
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
        if hasattr(batch, "is_ragged") and batch.is_ragged:
            return self._average_loss_lists(
                [
                    self._compute_losses(model, optimizer_state, sample, ode_method, ode_args)
                    for sample in batch.iter_single_batches()
                ]
            )
        runtime_batch = batch.to(self.device)
        updater = optimizer_state._linear_updater
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
        avg_loss, _ = optimizer_state._linear_updater.update(model, phase_context.train_loader)
        items = [float(avg_loss)] + [0.0] * (len(optimizer_state.criteria_weights) - 1)
        return float(avg_loss), items, False


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
        started_epoch = trainer_state.epoch
        model_artifact = self._ensure_model_artifact(phase_context, artifacts)
        self._ensure_history_artifact(artifacts)
        optimizer_state = artifacts.get("optimizer_state")
        updater = LSUpdater(
            method=self.spec.method,
            model=model_artifact.model,
            dt=phase_context.train_md["dt_and_n_steps"][0][0],
            params=self.spec.params,
            **self.spec.kwargs,
        )
        loss, params = updater.update(model_artifact.model, phase_context.train_loader)
        updated_names: list[str] = []
        if isinstance(optimizer_state, OptimizerStateArtifact) and self.spec.reset_optimizer:
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
            with torch.no_grad():
                for batch in phase_context.valid_loader:
                    valid_total += updater.eval_batch(
                        model_artifact.model, batch.to(self.device), optimizer_state.criteria[0]
                    ).item()
            valid_total /= len(phase_context.valid_loader)
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
    def execute(
        self,
        *,
        trainer_state: TrainerState,
        phase_context: PhaseContext,
        artifacts: ArtifactRegistry,
        run_name: str,
        logger: logging.Logger,
    ) -> PhaseResult:
        metrics = {
            "train_size": float(len(phase_context.train_set or [])),
            "valid_size": float(len(phase_context.valid_set or [])),
        }
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
        model_artifact = artifacts.require("model", ModelArtifact)
        optimizer_state = artifacts.get("optimizer_state")
        if not isinstance(optimizer_state, OptimizerStateArtifact):
            criteria = [torch.nn.MSELoss()]
            optimizer_state = OptimizerStateArtifact(
                optimizer=torch.optim.Adam(model_artifact.model.parameters(), lr=1e-3),
                criteria=criteria + criteria,
                criteria_weights=[1.0],
                criteria_names=["dynamics", "mse"],
            )
        history = self._ensure_history_artifact(artifacts)
        metric_name = optimizer_state.criteria_names[-1]
        dataset = phase_context.valid_set if self.spec.split == "valid" else phase_context.train_set
        evaluator = NodeOptimizerPhase(
            spec=OptimizerPhaseSpec(name=f"{self.spec.name}_eval", trainer="NODE", config={}),
            config=self.config,
            model_class=self.model_class,
            dtype=self.dtype,
            execution_services=self.execution_services,
        )
        ode_method, ode_args = evaluator._prediction_settings()
        value = evaluator._evaluate_prediction_criterion(
            model_artifact.model,
            optimizer_state,
            dataset,
            method=ode_method,
            ode_args=ode_args,
            evaluate_all=self.spec.evaluate_all,
        )
        evaluation = EvaluationArtifact(
            metrics={metric_name: value},
            split=self.spec.split,
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
        plot_hist(
            copy.deepcopy(history.hist),
            copy.deepcopy(history.crit),
            None if evaluation is None else evaluation.criterion_name,
            run_name,
            ifclose=True,
            prefix=self.execution_services.checkpoint_prefix,
        )
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
