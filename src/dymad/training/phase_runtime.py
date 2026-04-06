from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from dymad.training.execution_services import ExecutionServices


class TrainingCheckpointError(ValueError):
    """Raised when a typed training checkpoint cannot be loaded."""


@dataclass
class ModelArtifact:
    model: torch.nn.Module
    config: dict[str, Any]
    train_md: dict[str, Any]
    valid_md: dict[str, Any]
    dtype: torch.dtype


@dataclass
class OptimizerStateArtifact:
    optimizer: torch.optim.Optimizer
    schedulers: list[Any] = field(default_factory=list)
    criteria: list[torch.nn.Module] = field(default_factory=list)
    criteria_weights: list[float] = field(default_factory=list)
    criteria_names: list[str] = field(default_factory=list)
    owner_phase: str = ""


@dataclass
class TrainingHistoryArtifact:
    hist: list[Any] = field(default_factory=list)
    crit: list[Any] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_loss: dict[str, float] = field(default_factory=lambda: {"valid_total": float("inf")})
    best_model_state_dict: dict[str, Any] | None = None
    convergence_epoch: int | None = None


@dataclass
class LinearSolveRecord:
    phase_name: str
    method: str
    loss: float
    updated_parameters: list[str] = field(default_factory=list)


@dataclass
class LinearSolveReportArtifact:
    records: list[LinearSolveRecord] = field(default_factory=list)


@dataclass
class EvaluationArtifact:
    metrics: dict[str, float] = field(default_factory=dict)
    split: str = "valid"
    criterion_name: str = "total"


@dataclass
class ExportArtifact:
    outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class ArtifactRegistry:
    """Typed intermediate artifacts shared across phases."""

    _artifacts: dict[str, Any] = field(default_factory=dict)

    def put(self, key: str, artifact: Any) -> Any:
        self._artifacts[key] = artifact
        return artifact

    def get(self, key: str, default: Any = None) -> Any:
        return self._artifacts.get(key, default)

    def require(self, key: str, expected_type: type | tuple[type, ...] | None = None) -> Any:
        if key not in self._artifacts:
            raise KeyError(f"Missing required artifact '{key}'.")
        artifact = self._artifacts[key]
        if expected_type is not None and not isinstance(artifact, expected_type):
            raise TypeError(f"Artifact '{key}' must be {expected_type}, got {type(artifact)}.")
        return artifact

    def keys(self) -> Iterable[str]:
        return self._artifacts.keys()

    def checkpoint_payload(self) -> dict[str, Any]:
        return dict(self._artifacts)

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any] | None) -> ArtifactRegistry:
        return cls(_artifacts={} if payload is None else dict(payload))


@dataclass
class PhaseRecord:
    name: str
    kind: str
    started_epoch: int
    completed_epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_keys: list[str] = field(default_factory=list)


@dataclass
class TrainerState:
    """Checkpointable training state."""

    config: dict[str, Any] | None
    execution_services: ExecutionServices | None = None
    device: torch.device | None = None
    epoch: int = 0
    best_loss: dict[str, float] = field(default_factory=lambda: {"valid_total": float("inf")})
    converged: bool = False
    convergence_epoch: int | None = None
    phase_cursor: int = 0
    phase_records: list[PhaseRecord] = field(default_factory=list)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": copy.deepcopy(self.config),
            "device": self.device,
            "epoch": self.epoch,
            "best_loss": copy.deepcopy(self.best_loss),
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
            "phase_cursor": self.phase_cursor,
            "phase_records": copy.deepcopy(self.phase_records),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: dict[str, Any],
        *,
        execution_services: ExecutionServices | None = None,
    ) -> TrainerState:
        return cls(
            config=payload.get("config"),
            execution_services=execution_services,
            device=payload.get("device"),
            epoch=payload.get("epoch", 0),
            best_loss=copy.deepcopy(payload.get("best_loss", {"valid_total": float("inf")})),
            converged=payload.get("converged", False),
            convergence_epoch=payload.get("convergence_epoch"),
            phase_cursor=payload.get("phase_cursor", 0),
            phase_records=copy.deepcopy(payload.get("phase_records", [])),
        )


@dataclass
class PhaseContext:
    """Live phase context for one run."""

    train_set: Dataset | None = None
    valid_set: Dataset | None = None
    train_loader: DataLoader | None = None
    valid_loader: DataLoader | None = None
    train_md: dict[str, Any] | None = None
    valid_md: dict[str, Any] | None = None


@dataclass
class PhaseResult:
    """Typed phase outcome."""

    name: str
    kind: str
    trainer_state: TrainerState
    phase_context: PhaseContext
    artifacts: ArtifactRegistry
    metrics: dict[str, float] = field(default_factory=dict)
    record: PhaseRecord | None = None

    def get_metric(self, metric_name: str) -> float:
        key = f"valid_{metric_name}"
        if key in self.trainer_state.best_loss:
            return self.trainer_state.best_loss[key]
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        raise KeyError(f"Metric '{metric_name}' not found in phase result '{self.name}'.")


def build_initial_trainer_state(
    config: dict[str, Any],
    *,
    execution_services: ExecutionServices,
) -> TrainerState:
    return TrainerState(
        config=execution_services.apply_to_config(config),
        execution_services=execution_services,
        device=execution_services.device,
    )
