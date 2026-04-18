"""Training workflow and profile registry built from current profile metadata."""

from __future__ import annotations

import copy
from collections import defaultdict
from functools import lru_cache
from typing import Any, cast

from dymad.agent.registry.models import list_model_capabilities, resolve_model_capability
from dymad.agent.registry.types import (
    DatasetKind,
    ProfileCapability,
    TrainingWorkflowCapability,
)


def _base_profile(*, graph: bool, run_name: str, model: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": {
            "path": "",
            "double_precision": False,
        },
        "transform_x": {"type": "identity"},
        "transform_u": {"type": "identity"},
        "split": {"train_frac": 0.75},
        "dataloader": {"batch_size": 16},
        "model": {
            "name": run_name,
            **model,
        },
        "criterion": {
            "dynamics": {"weight": 1.0},
            "recon": {"weight": 1.0},
        },
        "plotting": {
            "prediction": not graph,
            "max_state_dims": 16,
            "max_control_dims": 8,
        },
        "phases": [
            {
                "type": "optimizer",
                "name": "WeakForm",
                "trainer": "Weak",
                "n_epochs": 25,
                "save_interval": 5,
                "load_checkpoint": False,
                "learning_rate": 5e-3,
                "decay_rate": 0.999,
                "weak_form_params": {
                    "N": 13,
                    "dN": 2,
                    "ordpol": 2,
                    "ordint": 2,
                },
            }
        ],
    }


_PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    "kbf-regular-default": _base_profile(
        graph=False,
        run_name="kbf_regular",
        model={
            "encoder_layers": 1,
            "decoder_layers": 1,
            "hidden_dimension": 32,
            "koopman_dimension": 4,
            "const_term": True,
            "activation": "none",
            "weight_init": "xavier_uniform",
            "input_order": "cubic",
        },
    ),
    "ldm-regular-default": _base_profile(
        graph=False,
        run_name="ldm_regular",
        model={
            "encoder_layers": 0,
            "processor_layers": 1,
            "decoder_layers": 0,
            "hidden_dimension": 32,
            "activation": "none",
            "weight_init": "xavier_uniform",
            "input_order": "cubic",
        },
    ),
    "lti-regular-default": _base_profile(
        graph=False,
        run_name="lti_regular",
        model={
            "encoder_layers": 1,
            "decoder_layers": 1,
            "hidden_dimension": 32,
            "koopman_dimension": 4,
            "const_term": True,
            "activation": "none",
            "weight_init": "xavier_uniform",
            "input_order": "cubic",
        },
    ),
    "kbf-graph-default": _base_profile(
        graph=True,
        run_name="kbf_graph",
        model={
            "encoder_layers": 1,
            "decoder_layers": 1,
            "hidden_dimension": 32,
            "koopman_dimension": 3,
            "const_term": True,
            "autoencoder_type": "cat",
            "gcl": "sage",
            "activation": "none",
            "weight_init": "xavier_uniform",
            "input_order": "cubic",
        },
    ),
    "ldm-graph-default": _base_profile(
        graph=True,
        run_name="ldm_graph",
        model={
            "encoder_layers": 1,
            "decoder_layers": 1,
            "hidden_dimension": 32,
            "autoencoder_type": "cat",
            "gcl": "sage",
            "activation": "none",
            "weight_init": "xavier_uniform",
            "input_order": "cubic",
        },
    ),
    "lti-graph-default": _base_profile(
        graph=True,
        run_name="lti_graph",
        model={
            "encoder_layers": 1,
            "decoder_layers": 1,
            "hidden_dimension": 32,
            "koopman_dimension": 3,
            "const_term": True,
            "autoencoder_type": "cat",
            "gcl": "sage",
            "activation": "none",
            "weight_init": "xavier_uniform",
            "input_order": "cubic",
        },
    ),
}

_PROFILE_ALIASES: dict[tuple[str, DatasetKind], str] = {
    ("dymad.models.collections:KBF", "regular"): "kbf-regular-default",
    ("dymad.models.collections:DKBF", "regular"): "kbf-regular-default",
    ("dymad.models.collections:LDM", "regular"): "ldm-regular-default",
    ("dymad.models.collections:DLDM", "regular"): "ldm-regular-default",
    ("dymad.models.collections:LTI", "regular"): "lti-regular-default",
    ("dymad.models.collections:DLTI", "regular"): "lti-regular-default",
    ("dymad.models.collections:GKBF", "graph"): "kbf-graph-default",
    ("dymad.models.collections:DGKBF", "graph"): "kbf-graph-default",
    ("dymad.models.collections:GLDM", "graph"): "ldm-graph-default",
    ("dymad.models.collections:DGLDM", "graph"): "ldm-graph-default",
    ("dymad.models.collections:GLTI", "graph"): "lti-graph-default",
    ("dymad.models.collections:DGLTI", "graph"): "lti-graph-default",
}


def profile_registry_payload() -> dict[str, dict[str, Any]]:
    return {key: copy.deepcopy(value) for key, value in _PROFILE_REGISTRY.items()}


def profile_alias_mapping() -> dict[tuple[str, DatasetKind], str]:
    return dict(_PROFILE_ALIASES)


@lru_cache(maxsize=1)
def _profile_capabilities() -> tuple[ProfileCapability, ...]:
    alias_groups: dict[str, list[tuple[str, DatasetKind]]] = defaultdict(list)
    for alias_key, profile_key in _PROFILE_ALIASES.items():
        alias_groups[profile_key].append(alias_key)

    capabilities: list[ProfileCapability] = []
    for profile_key in sorted(_PROFILE_REGISTRY):
        alias_entries = alias_groups.get(profile_key, [])
        model_refs = tuple(sorted(model_ref for model_ref, _ in alias_entries))
        dataset_kinds = {dataset_kind for _, dataset_kind in alias_entries}
        if len(dataset_kinds) > 1:
            raise ValueError(f"profile '{profile_key}' maps to multiple dataset kinds")
        dataset_kind = cast(DatasetKind, next(iter(dataset_kinds)) if dataset_kinds else "regular")
        model_keys = tuple(
            sorted(
                {
                    resolve_model_capability(model_ref).key
                    for model_ref, _dataset_kind in alias_entries
                }
            )
        )
        capability = ProfileCapability(
            key=profile_key,
            name=profile_key,
            dataset_kind=dataset_kind,
            model_keys=model_keys,
            implementation_model_refs=model_refs,
            aliases=(profile_key,),
            config=copy.deepcopy(_PROFILE_REGISTRY[profile_key]),
        )
        capabilities.append(capability)
    return tuple(capabilities)


def list_profile_capabilities() -> tuple[ProfileCapability, ...]:
    return _profile_capabilities()


@lru_cache(maxsize=1)
def _profile_capability_index() -> dict[str, ProfileCapability]:
    return {capability.key: capability for capability in _profile_capabilities()}


def available_profiles() -> list[str]:
    return sorted(_PROFILE_REGISTRY)


def profile_config(profile_name: str) -> dict[str, Any]:
    capability = _profile_capability_index().get(profile_name)
    if capability is None:
        raise ValueError(f"unknown profile: {profile_name}")
    return copy.deepcopy(capability.config)


def resolve_profile_name(
    *,
    model_ref: str,
    dataset_kind: DatasetKind,
    reference_profile: str | None,
) -> str:
    if reference_profile is not None:
        if reference_profile not in _PROFILE_REGISTRY:
            supported = ", ".join(available_profiles())
            raise ValueError(
                f"unsupported reference_profile '{reference_profile}'. supported profiles: {supported}"
            )
        return reference_profile
    profile = _PROFILE_ALIASES.get((model_ref, dataset_kind))
    if profile is None:
        supported = ", ".join(available_profiles())
        raise ValueError(
            f"no inferred reference profile for model_ref='{model_ref}' and dataset kind "
            f"'{dataset_kind}'. supported profiles: {supported}"
        )
    return profile


@lru_cache(maxsize=1)
def _training_capabilities() -> tuple[TrainingWorkflowCapability, ...]:
    profile_keys_by_model_kind: dict[tuple[str, DatasetKind], list[str]] = defaultdict(list)
    for profile in _profile_capabilities():
        for model_key in profile.model_keys:
            profile_keys_by_model_kind[(model_key, profile.dataset_kind)].append(profile.key)

    capabilities: list[TrainingWorkflowCapability] = []
    for model in list_model_capabilities():
        for dataset_kind in model.dataset_kinds:
            profile_keys = tuple(
                sorted(profile_keys_by_model_kind.get((model.key, dataset_kind), []))
            )
            default_profile = profile_keys[0] if profile_keys else None
            capabilities.append(
                TrainingWorkflowCapability(
                    key=f"train-{model.key}-{dataset_kind}",
                    name=f"Train {model.name} on {dataset_kind} datasets",
                    workflow_kind="training",
                    model_key=model.key,
                    dataset_kind=dataset_kind,
                    default_model_ref=model.default_model_ref_by_dataset_kind[dataset_kind],
                    default_profile=default_profile,
                    profile_keys=profile_keys,
                )
            )
    return tuple(capabilities)


def list_training_capabilities(
    *, dataset_kind: DatasetKind | None = None
) -> tuple[TrainingWorkflowCapability, ...]:
    capabilities = _training_capabilities()
    if dataset_kind is None:
        return capabilities
    return tuple(cap for cap in capabilities if cap.dataset_kind == dataset_kind)
