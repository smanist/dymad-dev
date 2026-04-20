"""Typed capability records for agent-facing registry access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dymad.models.model_spec import GraphMode, TimeDomain

DatasetKind = Literal["regular", "graph"]
WorkflowKind = Literal["training"]
AnalysisSupportLevel = Literal["supported", "experimental"]
AnalysisImplementation = Literal["library", "script_backed"]


@dataclass(frozen=True)
class ModelVariantCapability:
    key: str
    name: str
    model_ref: str
    dataset_kind: DatasetKind
    time_domain: TimeDomain
    graph_mode: GraphMode


@dataclass(frozen=True)
class ModelCapability:
    key: str
    name: str
    summary: str
    aliases: tuple[str, ...]
    dataset_kinds: tuple[DatasetKind, ...]
    default_model_ref_by_dataset_kind: dict[DatasetKind, str]
    variants: tuple[ModelVariantCapability, ...]


@dataclass(frozen=True)
class ProfileCapability:
    key: str
    name: str
    dataset_kind: DatasetKind
    model_keys: tuple[str, ...]
    implementation_model_refs: tuple[str, ...]
    aliases: tuple[str, ...]
    config: dict[str, Any]


@dataclass(frozen=True)
class TrainingWorkflowCapability:
    key: str
    name: str
    workflow_kind: WorkflowKind
    model_key: str
    dataset_kind: DatasetKind
    default_model_ref: str
    default_profile: str | None
    profile_keys: tuple[str, ...]


@dataclass(frozen=True)
class TrainingPhaseEntrySchema:
    key: str
    summary: str
    accepted_shape: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    enum_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allows_additional_keys: bool = False
    notes: tuple[str, ...] = ()
    example: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainingCapabilityExample:
    name: str
    user_request: str
    overrides: dict[str, Any]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingCVSchema:
    supported: bool
    workflow_kind: str
    allowed_keys: tuple[str, ...]
    default_metric: str
    param_grid_value_forms: tuple[str, ...]
    search_schema: dict[str, Any] = field(default_factory=dict)
    selection_schema: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingCapabilityDetail:
    capability: TrainingWorkflowCapability
    allowed_override_top_level_keys: tuple[str, ...]
    runtime_owned_override_paths: tuple[str, ...]
    allowed_data_override_keys: tuple[str, ...]
    runtime_owned_model_keys: tuple[str, ...]
    phase_entry_schemas: tuple[TrainingPhaseEntrySchema, ...]
    cv_schema: TrainingCVSchema
    translation_guidance: tuple[str, ...]
    constraint_notes: tuple[str, ...]
    auto_appended_phases: tuple[str, ...]
    examples: tuple[TrainingCapabilityExample, ...]


@dataclass(frozen=True)
class AnalysisCapability:
    key: str
    name: str
    summary: str
    support_level: AnalysisSupportLevel
    implementation: AnalysisImplementation
    requires_checkpoint: bool
    dataset_input_keys: tuple[str, ...]
    parameter_schema: dict[str, Any]


@dataclass(frozen=True)
class EvaluationCapability:
    key: str
    name: str
    summary: str
    dataset_kinds: tuple[DatasetKind, ...]
    supported_metrics: tuple[str, ...]
    parameter_schema: dict[str, Any]
    notes: tuple[str, ...] = ()
