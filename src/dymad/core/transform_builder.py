"""Explicit transform-construction boundary for typed and legacy paths."""

from __future__ import annotations

from typing import Any

import torch

from dymad.core.torch_transforms import (
    AddOneTransform,
    ComposeTransform,
    DelayEmbeddingTransform,
    IdentityTransform,
    LiftTransform,
    ScalerTransform,
)
from dymad.core.transform_module import LegacyTransformModuleAdapter, NDRTransformModuleAdapter, TransformModule
from dymad.transform import make_transform
from dymad.transform.base import Transform
from dymad.transform.collection import TRN_MAP

_TYPE_ALIASES = {
    "diffmap": "dm",
    "diffmapvb": "vbdm",
    "isomap": "isomap",
}


def build_legacy_transform(config) -> Transform:
    """Construct a legacy transform behind one explicit compatibility boundary."""
    return make_transform(_normalize_stage_configs(config))


def build_transform_module(config, state_dict: dict[str, Any] | None = None) -> TransformModule:
    """Build a typed transform module from legacy config/state inputs."""
    stages = _normalize_stage_configs(config)
    if not stages:
        module = IdentityTransform()
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    if _is_compose_state(state_dict):
        child_states = list(state_dict.get("children", []))
    elif len(stages) == 1:
        child_states = [state_dict]
    else:
        child_states = [None for _ in stages]

    if len(child_states) != len(stages):
        child_states = [None for _ in stages]

    children = [
        _build_stage_module(stage_cfg, child_state)
        for stage_cfg, child_state in zip(stages, child_states)
    ]
    return ComposeTransform(children)


def export_transform_state(module: TransformModule) -> dict[str, Any]:
    """Export a typed transform module in the legacy-compatible checkpoint format."""
    if isinstance(module, ComposeTransform):
        return {
            "type": "Compose",
            "names": [_legacy_type_name(child) for child in module.transforms],
            "delay": module.delay,
            "children": [export_transform_state(child) for child in module.transforms],
            "inp": module.input_dim,
            "out": module.output_dim,
        }

    if isinstance(module, (LegacyTransformModuleAdapter, NDRTransformModuleAdapter)):
        return module.legacy_transform.state_dict()

    if isinstance(module, IdentityTransform):
        return {"inp": module.input_dim, "out": module.output_dim}

    if isinstance(module, AddOneTransform):
        return {"inp": module.input_dim, "out": module.output_dim}

    if isinstance(module, ScalerTransform):
        return {
            "mode": module.mode,
            "off": module.offset.detach().cpu().numpy(),
            "scl": module.scale.detach().cpu().numpy(),
            "inp": module.input_dim,
            "out": module.output_dim,
        }

    if isinstance(module, DelayEmbeddingTransform):
        return {
            "delay": module.delay,
            "inp": module.input_dim,
            "out": module.output_dim,
        }

    if isinstance(module, LiftTransform):
        return {
            "inp": module.input_dim,
            "out": module.output_dim,
            "C": None,
            "fobs": module.fobs,
            "finv": module.finv,
            "fargs": dict(module.kwargs),
        }

    raise TypeError(f"Legacy export is not implemented for {type(module).__name__}")


def _normalize_stage_configs(config) -> list[dict[str, Any]]:
    if config is None:
        return []
    if isinstance(config, dict):
        config = [config]
    normalized = []
    for stage in config:
        stage_dict = dict(stage)
        stage_type = stage_dict.get("type")
        if stage_type is not None:
            stage_dict["type"] = _canonicalize_type(stage_type)
        normalized.append(stage_dict)
    return normalized


def _canonicalize_type(stage_type: str) -> str:
    lowered = str(stage_type).lower()
    return _TYPE_ALIASES.get(lowered, lowered)


def _is_compose_state(state_dict: dict[str, Any] | None) -> bool:
    if not isinstance(state_dict, dict):
        return False
    state_type = str(state_dict.get("type", "")).lower()
    return "children" in state_dict or state_type == "compose"


def _build_stage_module(stage_cfg: dict[str, Any], state_dict: dict[str, Any] | None) -> TransformModule:
    stage_type = _canonicalize_type(stage_cfg.get("type", ""))
    kwargs = dict(stage_cfg)
    kwargs.pop("type", None)

    if stage_type in {"", "identity"}:
        module = IdentityTransform()
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    if stage_type == "add_one":
        module = AddOneTransform()
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    if stage_type == "scaler":
        module = ScalerTransform(mode=kwargs.get("mode", "01"))
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    if stage_type == "delay":
        module = DelayEmbeddingTransform(delay=int(kwargs.get("delay", 1)))
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    if stage_type == "lift" and _can_build_native_lift(kwargs, state_dict):
        module = LiftTransform(**kwargs)
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    legacy_transform = _instantiate_legacy_stage(stage_type, kwargs)
    if state_dict is not None:
        legacy_transform.load_state_dict(state_dict)

    if stage_type in {"dm", "vbdm", "isomap"}:
        return NDRTransformModuleAdapter(legacy_transform)
    return LegacyTransformModuleAdapter(
        legacy_transform,
        invertibility="approximate",
        supports_gradients="false",
    )


def _instantiate_legacy_stage(stage_type: str, kwargs: dict[str, Any]) -> Transform:
    if stage_type not in TRN_MAP:
        raise ValueError(f"Unknown transform type: {stage_type}")
    return TRN_MAP[stage_type](**kwargs)


def _can_build_native_lift(kwargs: dict[str, Any], state_dict: dict[str, Any] | None) -> bool:
    fobs = kwargs.get("fobs")
    if fobs not in {"poly", "mixed"}:
        return False
    if state_dict is None:
        return True
    return state_dict.get("C") is None


def _load_native_stage_state(module: TransformModule, state_dict: dict[str, Any]) -> None:
    if isinstance(module, IdentityTransform):
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        return

    if isinstance(module, AddOneTransform):
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        return

    if isinstance(module, ScalerTransform):
        off = torch.as_tensor(state_dict["off"], dtype=torch.get_default_dtype())
        scl = torch.as_tensor(state_dict["scl"], dtype=torch.get_default_dtype())
        module.offset = off
        module.scale = scl
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        return

    if isinstance(module, DelayEmbeddingTransform):
        module.delay = int(state_dict["delay"])
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        return

    if isinstance(module, LiftTransform):
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        module._feature_sizes = module._infer_feature_sizes()
        return

    raise TypeError(f"Native state loading is not implemented for {type(module).__name__}")


def _legacy_type_name(module: TransformModule) -> str:
    if isinstance(module, (LegacyTransformModuleAdapter, NDRTransformModuleAdapter)):
        return str(module.legacy_transform)
    if isinstance(module, IdentityTransform):
        return "identity"
    if isinstance(module, AddOneTransform):
        return "add_one"
    if isinstance(module, ScalerTransform):
        return "scaler"
    if isinstance(module, DelayEmbeddingTransform):
        return "delay"
    if isinstance(module, LiftTransform):
        return "lift"
    raise TypeError(f"Legacy type-name export is not implemented for {type(module).__name__}")
