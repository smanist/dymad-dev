"""Explicit transform-construction boundary for typed-native and external paths."""

from __future__ import annotations

from typing import Any

import torch

from dymad.core.external_transforms import (
    CallableExternalTransform,
    DiffMapTransform,
    DiffMapVBTransform,
    IsomapTransform,
)
from dymad.core.torch_transforms import (
    AddOneTransform,
    AutoencoderTransform,
    ComposeTransform,
    DelayEmbeddingTransform,
    IdentityTransform,
    LiftTransform,
    ScalerTransform,
    SVDTransform,
)
from dymad.core.transform_module import ExternalTransformModule, TransformModule

_TYPE_ALIASES = {
    "diffmap": "dm",
    "diffmapvb": "vbdm",
    "isomap": "isomap",
}


def build_transform_module(config, state_dict: dict[str, Any] | None = None) -> TransformModule:
    """Build a typed transform module from config/state inputs."""
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
        for stage_cfg, child_state in zip(stages, child_states, strict=False)
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

    if isinstance(module, SVDTransform):
        return {
            "order": module.order,
            "ifcen": module.ifcen,
            "inp": module.input_dim,
            "out": module.output_dim,
            "P": module.projection.detach().cpu().numpy(),
            "off": module.offset.detach().cpu().numpy(),
        }

    if isinstance(module, CallableExternalTransform):
        return {
            "inp": module.input_dim,
            "out": module.output_dim,
            "C": module._pseudo_inverse_matrix,
            "fobs": module.fobs,
            "finv": module.finv,
            "fargs": dict(module.kwargs),
        }

    if isinstance(module, DiffMapVBTransform):
        return {
            "inv": module.inverse_mode,
            "Knn": module.knn,
            "Kphi": module.kphi,
            "order": module.order,
            "rcond": module.rcond,
            "inp": module.input_dim,
            "out": module.output_dim,
            "X": module._X,
            "Z": module._Z,
            "alpha": module.alpha,
            "epsilon": module.epsilon,
            "mode": module.mode,
            "DM": module._ndr.state_dict(),
            "Kb": module.kb,
        }

    if isinstance(module, DiffMapTransform):
        return {
            "inv": module.inverse_mode,
            "Knn": module.knn,
            "Kphi": module.kphi,
            "order": module.order,
            "rcond": module.rcond,
            "inp": module.input_dim,
            "out": module.output_dim,
            "X": module._X,
            "Z": module._Z,
            "alpha": module.alpha,
            "epsilon": module.epsilon,
            "mode": module.mode,
            "DM": module._ndr.state_dict(),
        }

    if isinstance(module, IsomapTransform):
        return {
            "inv": module.inverse_mode,
            "Knn": module.knn,
            "Kphi": module.kphi,
            "order": module.order,
            "rcond": module.rcond,
            "inp": module.input_dim,
            "out": module.output_dim,
            "X": module._X,
            "Z": module._Z,
        }

    if isinstance(module, AutoencoderTransform):
        raise TypeError(
            "AutoencoderTransform is runtime-only and is not exported to checkpoint state."
        )

    if isinstance(module, ExternalTransformModule):
        raise TypeError(f"Legacy export is not implemented for {type(module).__name__}")

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


def _build_stage_module(
    stage_cfg: dict[str, Any], state_dict: dict[str, Any] | None
) -> TransformModule:
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

    if stage_type == "svd":
        module = SVDTransform(order=kwargs.get("order", 1.0), ifcen=kwargs.get("ifcen", False))
        if state_dict is not None:
            _load_native_stage_state(module, state_dict)
        return module

    if stage_type == "lift":
        if _can_build_native_lift(kwargs, state_dict):
            module = LiftTransform(**kwargs)
            if state_dict is not None:
                _load_native_stage_state(module, state_dict)
            return module
        module = CallableExternalTransform(
            kwargs.get("fobs"),
            kwargs.get("finv"),
            **{k: v for k, v in kwargs.items() if k not in {"fobs", "finv"}},
        )
        if state_dict is not None:
            _load_callable_external_state(module, state_dict)
        return module

    if stage_type == "dm":
        module = DiffMapTransform(**kwargs)
        if state_dict is not None:
            _load_dm_state(module, state_dict)
        return module

    if stage_type == "vbdm":
        module = DiffMapVBTransform(**kwargs)
        if state_dict is not None:
            _load_vbdm_state(module, state_dict)
        return module

    if stage_type == "isomap":
        module = IsomapTransform(**kwargs)
        if state_dict is not None:
            _load_isomap_state(module, state_dict)
        return module

    raise ValueError(f"Unknown transform type: {stage_type}")


def _can_build_native_lift(kwargs: dict[str, Any], state_dict: dict[str, Any] | None) -> bool:
    fobs = kwargs.get("fobs")
    if callable(fobs):
        return False
    if fobs not in {"poly", "mixed"}:
        return False
    if state_dict is None:
        return True
    return state_dict.get("C") is None


def _load_native_stage_state(module: TransformModule, state_dict: dict[str, Any]) -> None:
    def _state_tensor(value):
        return torch.as_tensor(value)

    if isinstance(module, IdentityTransform):
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        return

    if isinstance(module, AddOneTransform):
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        return

    if isinstance(module, ScalerTransform):
        off = _state_tensor(state_dict["off"])
        scl = _state_tensor(state_dict["scl"])
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

    if isinstance(module, SVDTransform):
        module.order = state_dict["order"]
        module.ifcen = state_dict["ifcen"]
        module.projection = _state_tensor(state_dict["P"])
        module.offset = _state_tensor(state_dict["off"])
        module.input_dim = state_dict.get("inp")
        module.output_dim = state_dict.get("out")
        module.invertibility = "exact" if module.input_dim == module.output_dim else "approximate"
        return

    raise TypeError(f"Native state loading is not implemented for {type(module).__name__}")


def _load_callable_external_state(
    module: CallableExternalTransform,
    state_dict: dict[str, Any],
) -> None:
    module.input_dim = state_dict.get("inp")
    module.output_dim = state_dict.get("out")
    module.fobs = state_dict.get("fobs")
    module.finv = state_dict.get("finv")
    module.kwargs = dict(state_dict.get("fargs", {}))
    module._pseudo_inverse_matrix = state_dict.get("C")


def _load_isomap_state(module: IsomapTransform, state_dict: dict[str, Any]) -> None:
    module.inverse_mode = state_dict["inv"]
    module.knn = state_dict["Knn"]
    module.kphi = state_dict["Kphi"]
    module.order = state_dict["order"]
    module.rcond = state_dict["rcond"]
    module.input_dim = state_dict["inp"]
    module.output_dim = state_dict["out"]
    module.embedding_dim = state_dict["out"]
    module._X = state_dict["X"]
    module._Z = state_dict["Z"]
    module._make_ndr()
    module._Z = module._ndr.fit_transform(module._X)
    module._prepare_inverse()


def _load_dm_state(module: DiffMapTransform, state_dict: dict[str, Any]) -> None:
    module.inverse_mode = state_dict["inv"]
    module.knn = state_dict["Knn"]
    module.kphi = state_dict["Kphi"]
    module.order = state_dict["order"]
    module.rcond = state_dict["rcond"]
    module.input_dim = state_dict["inp"]
    module.output_dim = state_dict["out"]
    module.embedding_dim = state_dict["out"]
    module.alpha = state_dict["alpha"]
    module.epsilon = state_dict["epsilon"]
    module.mode = state_dict["mode"]
    module._X = state_dict["X"]
    module._Z = state_dict["Z"]
    module._make_ndr()
    module._ndr.load_state_dict(state_dict["DM"])
    module._prepare_inverse()


def _load_vbdm_state(module: DiffMapVBTransform, state_dict: dict[str, Any]) -> None:
    module.kb = state_dict["Kb"]
    _load_dm_state(module, state_dict)


def _legacy_type_name(module: TransformModule) -> str:
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
    if isinstance(module, SVDTransform):
        return "svd"
    if isinstance(module, CallableExternalTransform):
        return "lift"
    if isinstance(module, DiffMapVBTransform):
        return "vbdm"
    if isinstance(module, DiffMapTransform):
        return "dm"
    if isinstance(module, IsomapTransform):
        return "isomap"
    raise TypeError(f"Legacy type-name export is not implemented for {type(module).__name__}")
