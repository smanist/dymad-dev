"""Explicit reference profiles for MCP-driven training runs."""

from __future__ import annotations

import copy
from typing import Any


def _base_profile(*, graph: bool, run_name: str, model: dict[str, Any]) -> dict[str, Any]:
    profile = {
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
    return profile


PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
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


PROFILE_ALIASES: dict[tuple[str, str], str] = {
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


def available_profiles() -> list[str]:
    return sorted(PROFILE_REGISTRY)


def resolve_profile_name(
    *,
    model_ref: str,
    dataset_kind: str,
    reference_profile: str | None,
) -> str:
    if reference_profile is not None:
        if reference_profile not in PROFILE_REGISTRY:
            supported = ", ".join(available_profiles())
            raise ValueError(
                f"unsupported reference_profile '{reference_profile}'. supported profiles: {supported}"
            )
        return reference_profile
    profile = PROFILE_ALIASES.get((model_ref, dataset_kind))
    if profile is None:
        supported = ", ".join(available_profiles())
        raise ValueError(
            f"no inferred reference profile for model_ref='{model_ref}' and dataset kind "
            f"'{dataset_kind}'. supported profiles: {supported}"
        )
    return profile


def profile_config(profile_name: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(PROFILE_REGISTRY[profile_name])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile_name}") from exc
