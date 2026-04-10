"""Execution-layer state for compatibility planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dymad.agent.store.object_store import ObjectSummary, TrainingRunRecord


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
class DatasetCompatibility:
    dataset_handle: str
    dataset_kind: str
    model_ref: str
    model_name: str
    expected_graph: bool
    expected_dataset_kind: str
    is_compatible: bool
    reason: str | None


@dataclass(frozen=True)
class ModelFamilyDescription:
    model_ref: str
    name: str
    time_domain: str
    graph_mode: str
    recipe_kind: str
    rollout_family: str
    default_predictor: str
    allowed_predictors: tuple[str, ...]
    expects_graph_data: bool


@dataclass(frozen=True)
class ReferenceProfileDescription:
    profile_name: str
    dataset_kind: str | None
    model_refs: tuple[str, ...]
    model_defaults: dict[str, Any]
    default_phases: list[dict[str, Any]]


@dataclass(frozen=True)
class TrainingConfigValidationResult:
    is_valid: bool
    compatibility: DatasetCompatibility
    reference_profile: str | None
    trainer_kind: str | None
    run_name: str | None
    normalized_config: dict[str, Any] | None
    rejection_reason: str | None


@dataclass(frozen=True)
class MaterializedTrainingConfigResult:
    config_path: str
    compatibility: DatasetCompatibility
    reference_profile: str
    trainer_kind: str
    run_name: str
    normalized_config: dict[str, Any]


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
    prediction_summary: ObjectSummary
    artifacts: dict[str, str | list[str]]
    metrics: dict[str, Any]
    plot_skipped_reason: str | None


@dataclass(frozen=True)
class PredictCheckpointResult:
    prediction_summary: ObjectSummary
    artifacts: dict[str, str]
    selected_indices: list[int]


@dataclass(frozen=True)
class ComputeRolloutMetricsResult:
    evaluation_summary: ObjectSummary
    prediction_summary: ObjectSummary
    artifacts: dict[str, str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PlotRolloutsResult:
    prediction_summary: ObjectSummary
    artifacts: dict[str, list[str]]
    plot_skipped_reason: str | None


@dataclass(frozen=True)
class TrainingRunInspection:
    run_summary: ObjectSummary
    run_record: TrainingRunRecord


@dataclass(frozen=True)
class TrainingArtifactsListing:
    run_summary: ObjectSummary
    run_record: TrainingRunRecord
    paths: dict[str, str]
    exists: dict[str, bool]
