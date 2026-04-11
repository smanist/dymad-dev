"""Model capability registry built from predefined typed model specs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import cast

from dymad.agent.registry.types import DatasetKind, ModelCapability, ModelVariantCapability
from dymad.models import collections


@dataclass(frozen=True)
class _ModelFamilyDefinition:
    key: str
    name: str
    summary: str
    aliases: tuple[str, ...]
    variant_names: tuple[str, ...]
    default_variant_by_dataset_kind: dict[DatasetKind, str]


_MODEL_FAMILY_DEFINITIONS: tuple[_ModelFamilyDefinition, ...] = (
    _ModelFamilyDefinition(
        key="kbf",
        name="KBF",
        summary="Koopman bilinear form models.",
        aliases=("koopman_bilinear_form",),
        variant_names=("KBF", "DKBF", "GKBF", "DGKBF"),
        default_variant_by_dataset_kind={"regular": "KBF", "graph": "GKBF"},
    ),
    _ModelFamilyDefinition(
        key="km",
        name="KM",
        summary="Kernel machine models.",
        aliases=("kernel_machine",),
        variant_names=("KM", "DKM", "GKM", "DGKM"),
        default_variant_by_dataset_kind={"regular": "KM", "graph": "GKM"},
    ),
    _ModelFamilyDefinition(
        key="kmsk",
        name="KMSK",
        summary="Kernel machine models with skip connections.",
        aliases=("kernel_machine_skip",),
        variant_names=("DKMSK", "DGKMSK"),
        default_variant_by_dataset_kind={"regular": "DKMSK", "graph": "DGKMSK"},
    ),
    _ModelFamilyDefinition(
        key="kmm",
        name="KMM",
        summary="Kernel machine on manifold models.",
        aliases=("kernel_machine_manifold",),
        variant_names=("KMM",),
        default_variant_by_dataset_kind={"regular": "KMM"},
    ),
    _ModelFamilyDefinition(
        key="ldm",
        name="LDM",
        summary="Latent dynamics models.",
        aliases=("latent_dynamics_model",),
        variant_names=("LDM", "DLDM", "GLDM", "DGLDM", "LDMG", "DLDMG"),
        default_variant_by_dataset_kind={"regular": "LDM", "graph": "GLDM"},
    ),
    _ModelFamilyDefinition(
        key="lti",
        name="LTI",
        summary="Linear time-invariant models.",
        aliases=("linear_time_invariant",),
        variant_names=("LTI", "DLTI", "GLTI", "DGLTI"),
        default_variant_by_dataset_kind={"regular": "LTI", "graph": "GLTI"},
    ),
    _ModelFamilyDefinition(
        key="sdm",
        name="SDM",
        summary="Sequential dynamics models.",
        aliases=("sequential_dynamics_model",),
        variant_names=("DSDM", "DSDMG"),
        default_variant_by_dataset_kind={"regular": "DSDM", "graph": "DSDMG"},
    ),
)


def _dataset_kind_for_variant(variant) -> DatasetKind:
    return "regular" if variant.typed_spec().graph_mode == "none" else "graph"


def _variant_capability(variant_name: str) -> ModelVariantCapability:
    variant = getattr(collections, variant_name)
    spec = variant.typed_spec()
    return ModelVariantCapability(
        key=variant_name.lower(),
        name=variant_name,
        model_ref=f"{collections.__name__}:{variant_name}",
        dataset_kind=_dataset_kind_for_variant(variant),
        time_domain=spec.time_domain,
        graph_mode=spec.graph_mode,
    )


def _model_capability(definition: _ModelFamilyDefinition) -> ModelCapability:
    variants = tuple(_variant_capability(name) for name in definition.variant_names)
    dataset_kinds = cast(
        tuple[DatasetKind, ...],
        tuple(
            kind for kind in ("regular", "graph") if any(v.dataset_kind == kind for v in variants)
        ),
    )
    default_model_ref_by_dataset_kind = cast(
        dict[DatasetKind, str],
        {
            dataset_kind: f"{collections.__name__}:{variant_name}"
            for dataset_kind, variant_name in definition.default_variant_by_dataset_kind.items()
        },
    )
    return ModelCapability(
        key=definition.key,
        name=definition.name,
        summary=definition.summary,
        aliases=definition.aliases,
        dataset_kinds=dataset_kinds,
        default_model_ref_by_dataset_kind=default_model_ref_by_dataset_kind,
        variants=variants,
    )


@lru_cache(maxsize=1)
def _model_capabilities() -> tuple[ModelCapability, ...]:
    return tuple(_model_capability(definition) for definition in _MODEL_FAMILY_DEFINITIONS)


def list_model_capabilities() -> tuple[ModelCapability, ...]:
    return _model_capabilities()


@lru_cache(maxsize=1)
def _model_alias_index() -> dict[str, ModelCapability]:
    index: dict[str, ModelCapability] = {}
    for capability in _model_capabilities():
        keys = {
            capability.key,
            capability.name,
            *capability.aliases,
        }
        keys.update(variant.name for variant in capability.variants)
        keys.update(variant.model_ref for variant in capability.variants)
        for key in keys:
            index[key.strip().lower()] = capability
    return index


def resolve_model_capability(key_or_alias: str) -> ModelCapability:
    normalized = key_or_alias.strip().lower()
    if not normalized:
        raise ValueError("model capability key cannot be empty")
    capability = _model_alias_index().get(normalized)
    if capability is None:
        supported = ", ".join(cap.key for cap in _model_capabilities())
        raise ValueError(
            f"unknown model capability '{key_or_alias}'. supported capabilities: {supported}"
        )
    return capability
