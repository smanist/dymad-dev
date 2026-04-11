"""Typed capability records for agent-facing registry access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dymad.models.model_spec import GraphMode, TimeDomain

DatasetKind = Literal["regular", "graph"]
WorkflowKind = Literal["training"]


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
