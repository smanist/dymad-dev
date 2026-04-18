"""Execution-layer state for compatibility planning."""

from __future__ import annotations

from dataclasses import dataclass

from dymad.agent.store.object_store import ObjectSummary


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


@dataclass(frozen=True)
class DatasetInspection:
    dataset_handle: str
    format: str
    kind: str
    keys: tuple[str, ...]
    n_trajectories: int
    n_steps: int | None
    is_ragged: bool
    state_dim: int
    control_dim: int
    parameter_dim: int
    has_time: bool
    has_graph: bool
    n_nodes: int | None = None


@dataclass(frozen=True)
class TrainModelResult:
    run_summary: ObjectSummary
    checkpoint_summary: ObjectSummary
    artifacts: dict[str, str | None]
    metrics: dict[str, float | None]
    reference_profile: str
    trainer_kind: str


@dataclass(frozen=True)
class EvaluateModelResult:
    evaluation_summary: ObjectSummary
    artifacts: dict[str, str | list[str]]
    metrics: dict[str, float]
    plot_skipped_reason: str | None


@dataclass(frozen=True)
class AnalysisRunResult:
    workflow_key: str
    artifacts: dict[str, str]
    summary: dict[str, float | int | str | bool | None]
