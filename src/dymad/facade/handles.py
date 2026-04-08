"""Typed handle validators for facade-managed objects."""

from __future__ import annotations

import re
from dataclasses import dataclass


class HandleValidationError(ValueError):
    """Raised when a handle has the wrong prefix or shape."""


_CHECKPOINT_RE = re.compile(r"^chk_[a-z0-9]{6,}$")
_DATASET_RE = re.compile(r"^ds_[a-z0-9]{6,}$")
_TRAINING_RUN_RE = re.compile(r"^run_[a-z0-9]{6,}$")
_EVALUATION_RE = re.compile(r"^eval_[a-z0-9]{6,}$")
_PREDICTION_RE = re.compile(r"^pred_[a-z0-9]{6,}$")
_SPECTRAL_SNAPSHOT_RE = re.compile(r"^specsnap_[a-z0-9]{6,}$")


@dataclass(frozen=True)
class CheckpointHandle:
    value: str

    def __post_init__(self) -> None:
        if not _CHECKPOINT_RE.match(self.value):
            raise HandleValidationError(f"invalid checkpoint handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> CheckpointHandle:
        return cls(raw)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DatasetHandle:
    value: str

    def __post_init__(self) -> None:
        if not _DATASET_RE.match(self.value):
            raise HandleValidationError(f"invalid dataset handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> DatasetHandle:
        return cls(raw)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TrainingRunHandle:
    value: str

    def __post_init__(self) -> None:
        if not _TRAINING_RUN_RE.match(self.value):
            raise HandleValidationError(f"invalid training run handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> TrainingRunHandle:
        return cls(raw)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EvaluationHandle:
    value: str

    def __post_init__(self) -> None:
        if not _EVALUATION_RE.match(self.value):
            raise HandleValidationError(f"invalid evaluation handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> EvaluationHandle:
        return cls(raw)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PredictionHandle:
    value: str

    def __post_init__(self) -> None:
        if not _PREDICTION_RE.match(self.value):
            raise HandleValidationError(f"invalid prediction handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> PredictionHandle:
        return cls(raw)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SpectralSnapshotHandle:
    value: str

    def __post_init__(self) -> None:
        if not _SPECTRAL_SNAPSHOT_RE.match(self.value):
            raise HandleValidationError(f"invalid spectral snapshot handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> SpectralSnapshotHandle:
        return cls(raw)

    def __str__(self) -> str:
        return self.value
