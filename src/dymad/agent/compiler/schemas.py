"""Typed request and result records for user-mode training compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dymad.agent.registry import DatasetKind, ModelCapability, ProfileCapability

DiagnosticLevel = Literal["warning"]


class TrainingCompileValidationError(ValueError):
    """Raised when a training request cannot be compiled safely."""

    def __init__(self, message: str, *, field_path: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.field_path = field_path


@dataclass(frozen=True)
class CompileDiagnostic:
    level: DiagnosticLevel
    code: str
    message: str
    field_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingRequest:
    train_dataset_handle: str
    model_key: str
    valid_dataset_handle: str | None = None
    reference_profile: str | None = None
    overrides: dict[str, Any] | str | None = None
    run_name: str | None = None
    seed: int | None = None
    device: str = "auto"
    max_workers: int = 1


@dataclass(frozen=True)
class CompiledTrainingRequest:
    request: TrainingRequest
    model: ModelCapability
    profile: ProfileCapability
    model_ref: str
    train_dataset_kind: DatasetKind
    valid_dataset_kind: DatasetKind | None
    effective_run_name: str
    effective_config: dict[str, Any]
    trainer_kind: str
    warnings: tuple[CompileDiagnostic, ...] = field(default_factory=tuple)
