"""Typed handle validators for facade-managed objects."""

from __future__ import annotations

from dataclasses import dataclass
import re


class HandleValidationError(ValueError):
    """Raised when a handle has the wrong prefix or shape."""


_CHECKPOINT_RE = re.compile(r"^chk_[a-z0-9]{6,}$")
_PREDICTION_RE = re.compile(r"^pred_[a-z0-9]{6,}$")


@dataclass(frozen=True)
class CheckpointHandle:
    value: str

    def __post_init__(self) -> None:
        if not _CHECKPOINT_RE.match(self.value):
            raise HandleValidationError(f"invalid checkpoint handle: {self.value}")

    @classmethod
    def parse(cls, raw: str) -> "CheckpointHandle":
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
    def parse(cls, raw: str) -> "PredictionHandle":
        return cls(raw)

    def __str__(self) -> str:
        return self.value
