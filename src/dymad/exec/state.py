"""Execution-layer state for compatibility planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionWorkflowPlan:
    checkpoint_handle: str
    prediction_handle: str
    entrypoint: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SpectralWorkflowPlan:
    checkpoint_handle: str
    spectral_snapshot_handle: str
    entrypoint: str
    notes: tuple[str, ...]
