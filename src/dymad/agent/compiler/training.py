"""Typed training compiler built on the existing execution workflow helpers."""

from __future__ import annotations

import copy
import json
from typing import Any, Never, cast

from dymad.agent.compiler.schemas import (
    CompiledTrainingRequest,
    TrainingCompileValidationError,
    TrainingRequest,
)
from dymad.agent.exec import workflow as exec_workflow
from dymad.agent.facade.operations import FacadeOperations
from dymad.agent.registry import (
    DatasetKind,
    list_profile_capabilities,
    resolve_model_capability,
)
from dymad.agent.registry.training_schema import (
    ALLOWED_DATA_OVERRIDE_KEYS,
    ALLOWED_TOP_LEVEL_OVERRIDE_KEYS,
    CV_ALLOWED_KEYS,
    RUNTIME_OWNED_MODEL_KEYS,
    RUNTIME_OWNED_OVERRIDE_PATHS,
)


def _raise_invalid(message: str, *, field_path: tuple[str, ...]) -> Never:
    raise TrainingCompileValidationError(message, field_path=field_path)


def _coerce_optional_mapping(
    value: dict[str, Any] | str | None,
    *,
    field_path: tuple[str, ...],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        _raise_invalid("overrides must be a mapping or a JSON object string", field_path=field_path)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TrainingCompileValidationError(
            "overrides must be a mapping or a JSON object string",
            field_path=field_path,
        ) from exc
    if not isinstance(decoded, dict):
        _raise_invalid(
            "overrides JSON string must decode to an object",
            field_path=field_path,
        )
    return cast(dict[str, Any], decoded)


def _is_non_empty_dotted_key(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return all(part != "" for part in value.split("."))


def _validate_override_path(path: tuple[str, ...]) -> None:
    path_str = ".".join(path)
    if len(path) == 1:
        key = path[0]
        if path_str in RUNTIME_OWNED_OVERRIDE_PATHS:
            _raise_invalid(
                f"overrides.{key} is runtime-owned and cannot be set by the caller",
                field_path=("overrides", key),
            )
        if key not in ALLOWED_TOP_LEVEL_OVERRIDE_KEYS and key != "data":
            _raise_invalid(
                f"overrides.{key} is not supported by the user-mode compiler",
                field_path=("overrides", key),
            )
    if path_str in RUNTIME_OWNED_OVERRIDE_PATHS:
        _raise_invalid(
            f"overrides.{'.'.join(path)} is runtime-owned and cannot be set by the caller",
            field_path=("overrides",) + path,
        )
    if path[:1] == ("data",) and len(path) == 2 and path[1] not in ALLOWED_DATA_OVERRIDE_KEYS:
        _raise_invalid(
            f"overrides.data.{path[1]} is not supported by the user-mode compiler",
            field_path=("overrides",) + path,
        )
    if path[:1] == ("model",) and len(path) == 2 and path[1] in RUNTIME_OWNED_MODEL_KEYS:
        _raise_invalid(
            f"overrides.model.{path[1]} is runtime-owned and cannot be set by the caller",
            field_path=("overrides",) + path,
        )


def _normalize_param_grid_value(
    value: object, *, key: str
) -> list[Any] | tuple[str, tuple[Any, ...]]:
    if isinstance(value, list):
        if not value:
            _raise_invalid(
                "overrides.cv.param_grid values must be non-empty",
                field_path=("overrides", "cv", "param_grid", key),
            )
        if (
            len(value) == 2
            and value[0] in {"linspace", "logspace"}
            and isinstance(value[1], (list, tuple))
        ):
            return cast(str, value[0]), tuple(cast(list[Any] | tuple[Any, ...], value[1]))
        return value
    if isinstance(value, tuple) and len(value) == 2:
        kind, args = value
        if kind in {"linspace", "logspace"} and isinstance(args, (list, tuple)):
            return cast(str, kind), tuple(cast(list[Any] | tuple[Any, ...], args))
    _raise_invalid(
        "overrides.cv.param_grid values must be non-empty lists or ('linspace'|'logspace', ...)",
        field_path=("overrides", "cv", "param_grid", key),
    )


def _validate_cv_config(cv_config: object) -> None:
    if not isinstance(cv_config, dict):
        _raise_invalid(
            "overrides.cv must be a mapping",
            field_path=("overrides", "cv"),
        )
    cv_mapping = cast(dict[str, Any], cv_config)

    unknown_keys = sorted(key for key in cv_mapping if key not in CV_ALLOWED_KEYS)
    if unknown_keys:
        invalid_key = unknown_keys[0]
        _raise_invalid(
            f"overrides.cv.{invalid_key} is not supported by the user-mode compiler",
            field_path=("overrides", "cv", invalid_key),
        )

    param_grid = cv_mapping.get("param_grid")
    if not isinstance(param_grid, dict) or not param_grid:
        _raise_invalid(
            "overrides.cv.param_grid must be a non-empty mapping",
            field_path=("overrides", "cv", "param_grid"),
        )
    param_grid_mapping = cast(dict[str, Any], param_grid)
    for key, value in param_grid_mapping.items():
        if not _is_non_empty_dotted_key(key):
            _raise_invalid(
                "overrides.cv.param_grid keys must be non-empty dotted config paths",
                field_path=("overrides", "cv", "param_grid"),
            )
        _validate_override_path(tuple(key.split(".")))
        param_grid_mapping[key] = _normalize_param_grid_value(value, key=key)

    metric = cv_mapping.get("metric")
    if metric is not None and not isinstance(metric, str):
        _raise_invalid(
            "overrides.cv.metric must be a string",
            field_path=("overrides", "cv", "metric"),
        )


def _validate_overrides(config: dict[str, Any] | None, *, prefix: tuple[str, ...] = ()) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        _raise_invalid(
            "overrides must be a mapping of config paths to values",
            field_path=prefix or ("overrides",),
        )
    for key, value in config.items():
        path = prefix + (key,)
        _validate_override_path(path)
        if path == ("cv",):
            _validate_cv_config(value)
            continue
        if isinstance(value, dict):
            _validate_overrides(value, prefix=path)


def _optimizer_trainer_from_phase_entry(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    trainer = entry.get("trainer")
    if not isinstance(trainer, str):
        return None
    phase_type = entry.get("type")
    if phase_type not in (None, "optimizer"):
        return None
    return trainer


def _default_optimizer_phase_by_trainer(
    profile_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    phases = profile_config.get("phases")
    if not isinstance(phases, list):
        return {}

    defaults: dict[str, dict[str, Any]] = {}
    for phase in phases:
        trainer = _optimizer_trainer_from_phase_entry(phase)
        if trainer is None or trainer in defaults:
            continue
        defaults[trainer] = copy.deepcopy(cast(dict[str, Any], phase))
    return defaults


def _normalize_phase_override_entry(
    entry: object,
    *,
    default_optimizer_phase_by_trainer: dict[str, dict[str, Any]],
) -> object:
    if not isinstance(entry, dict):
        return copy.deepcopy(entry)

    if "repeat" in entry:
        normalized_entry = copy.deepcopy(entry)
        repeat_cfg = normalized_entry.get("repeat")
        if not isinstance(repeat_cfg, dict):
            return normalized_entry
        nested_phases = repeat_cfg.get("phases")
        if isinstance(nested_phases, list):
            repeat_cfg["phases"] = [
                _normalize_phase_override_entry(
                    phase_entry,
                    default_optimizer_phase_by_trainer=default_optimizer_phase_by_trainer,
                )
                for phase_entry in nested_phases
            ]
        return normalized_entry

    trainer = _optimizer_trainer_from_phase_entry(entry)
    if trainer is None:
        return copy.deepcopy(entry)

    default_phase = default_optimizer_phase_by_trainer.get(trainer)
    if default_phase is None:
        return copy.deepcopy(entry)

    return cast(dict[str, Any], exec_workflow._deep_merge(default_phase, entry))


def _normalize_overrides_against_profile(
    overrides: dict[str, Any] | None,
    *,
    profile_config: dict[str, Any],
) -> dict[str, Any] | None:
    if overrides is None:
        return None

    normalized_overrides = copy.deepcopy(overrides)
    phases = normalized_overrides.get("phases")
    if not isinstance(phases, list):
        return normalized_overrides

    default_optimizer_phase_by_trainer = _default_optimizer_phase_by_trainer(profile_config)
    normalized_overrides["phases"] = [
        _normalize_phase_override_entry(
            phase_entry,
            default_optimizer_phase_by_trainer=default_optimizer_phase_by_trainer,
        )
        for phase_entry in phases
    ]
    return normalized_overrides


def _dataset_kind_from_record(dataset_record) -> DatasetKind:
    return cast(DatasetKind, dataset_record.kind)


def _resolve_profile_capability(profile_key: str):
    for capability in list_profile_capabilities():
        if capability.key == profile_key:
            return capability
    raise TrainingCompileValidationError(
        f"unknown reference_profile '{profile_key}'",
        field_path=("reference_profile",),
    )


def _latent_dimension_field_path(model_config: dict[str, Any]) -> tuple[str, ...]:
    for key in ("koopman_dimension", "kernel_dimension", "latent_dimension"):
        if key in model_config:
            return ("overrides", "model", key)
    return ("overrides", "model")


def _latent_dimension(model_config: dict[str, Any]) -> int | None:
    for key in ("koopman_dimension", "kernel_dimension", "latent_dimension"):
        value = model_config.get(key)
        if value is not None:
            return int(value)
    return None


def _validate_zero_layer_autoencoder_identity(
    *,
    dataset_record,
    config: dict[str, Any],
) -> None:
    model_config = cast(dict[str, Any], config.get("model", {}))
    latent_dim = _latent_dimension(model_config)
    if latent_dim is None:
        return

    encoder_layers = model_config.get("encoder_layers")
    decoder_layers = model_config.get("decoder_layers")
    if encoder_layers != 0 and decoder_layers != 0:
        return

    state_dim = exec_workflow._inspect_dataset_record(dataset_record).state_dim
    if latent_dim == state_dim:
        return

    layer_names: list[str] = []
    if encoder_layers == 0:
        layer_names.append("encoder")
    if decoder_layers == 0:
        layer_names.append("decoder")
    joined = " and ".join(layer_names)
    _raise_invalid(
        f"overrides.model.{joined}_layers=0 only yields a true identity map when the latent "
        f"dimension matches the dataset state dimension ({state_dim}); got {latent_dim}",
        field_path=_latent_dimension_field_path(model_config),
    )


def compile_training_request(
    *,
    facade: FacadeOperations,
    request: TrainingRequest,
) -> CompiledTrainingRequest:
    overrides = _coerce_optional_mapping(request.overrides, field_path=("overrides",))
    normalized_request = TrainingRequest(
        train_dataset_handle=request.train_dataset_handle,
        model_key=request.model_key,
        valid_dataset_handle=request.valid_dataset_handle,
        reference_profile=request.reference_profile,
        overrides=overrides,
        run_name=request.run_name,
        seed=request.seed,
        device=request.device,
        max_workers=request.max_workers,
    )
    try:
        model = resolve_model_capability(normalized_request.model_key)
    except ValueError as exc:
        raise TrainingCompileValidationError(
            str(exc),
            field_path=("model_key",),
        ) from exc

    train_dataset = facade.get_dataset(normalized_request.train_dataset_handle)
    train_dataset_kind = _dataset_kind_from_record(train_dataset)
    if train_dataset_kind not in model.dataset_kinds:
        supported = ", ".join(model.dataset_kinds)
        raise TrainingCompileValidationError(
            f"model '{model.key}' does not support dataset kind '{train_dataset_kind}'. "
            f"supported kinds: {supported}",
            field_path=("model_key",),
        )

    valid_dataset = None
    valid_dataset_kind: DatasetKind | None = None
    if normalized_request.valid_dataset_handle is not None:
        valid_dataset = facade.get_dataset(normalized_request.valid_dataset_handle)
        valid_dataset_kind = _dataset_kind_from_record(valid_dataset)
        if valid_dataset_kind != train_dataset_kind:
            raise TrainingCompileValidationError(
                "train_dataset_handle and valid_dataset_handle must have the same dataset kind",
                field_path=("valid_dataset_handle",),
            )

    if normalized_request.reference_profile is not None:
        profile = _resolve_profile_capability(normalized_request.reference_profile)
        if profile.dataset_kind != train_dataset_kind:
            raise TrainingCompileValidationError(
                f"profile '{profile.key}' only supports dataset kind '{profile.dataset_kind}'",
                field_path=("reference_profile",),
            )
        if model.key not in profile.model_keys:
            supported_models = ", ".join(profile.model_keys)
            raise TrainingCompileValidationError(
                f"profile '{profile.key}' is not compatible with model '{model.key}'. "
                f"compatible model keys: {supported_models}",
                field_path=("reference_profile",),
            )

    model_ref = model.default_model_ref_by_dataset_kind[train_dataset_kind]
    profile_name = exec_workflow.resolve_profile_name(
        model_ref=model_ref,
        dataset_kind=train_dataset_kind,
        reference_profile=normalized_request.reference_profile,
    )
    profile = _resolve_profile_capability(profile_name)
    overrides = _normalize_overrides_against_profile(overrides, profile_config=profile.config)
    normalized_request = TrainingRequest(
        train_dataset_handle=request.train_dataset_handle,
        model_key=request.model_key,
        valid_dataset_handle=request.valid_dataset_handle,
        reference_profile=request.reference_profile,
        overrides=overrides,
        run_name=request.run_name,
        seed=request.seed,
        device=request.device,
        max_workers=request.max_workers,
    )

    _validate_overrides(overrides)
    exec_workflow._validate_user_config(overrides or {})

    effective_run_name = exec_workflow._validate_run_name(
        normalized_request.run_name or exec_workflow._default_run_name(model_ref)
    )
    profile_name, effective_config = exec_workflow._effective_config(
        model_ref=model_ref,
        dataset_record=train_dataset,
        valid_dataset_record=valid_dataset,
        reference_profile=normalized_request.reference_profile,
        user_config=overrides,
        run_name=effective_run_name,
    )
    _validate_zero_layer_autoencoder_identity(
        dataset_record=train_dataset,
        config=effective_config,
    )
    _, trainer_kind = exec_workflow._select_trainer(effective_config)

    return CompiledTrainingRequest(
        request=normalized_request,
        model=model,
        profile=profile,
        model_ref=model_ref,
        train_dataset_kind=train_dataset_kind,
        valid_dataset_kind=valid_dataset_kind,
        effective_run_name=effective_run_name,
        effective_config=effective_config,
        trainer_kind=trainer_kind,
    )
