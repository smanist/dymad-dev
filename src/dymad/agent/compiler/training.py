"""Typed training compiler built on the existing execution workflow helpers."""

from __future__ import annotations

import json
from typing import Any, cast

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
    RUNTIME_OWNED_MODEL_KEYS,
    RUNTIME_OWNED_OVERRIDE_PATHS,
)


def _raise_invalid(message: str, *, field_path: tuple[str, ...]) -> None:
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
        path_str = ".".join(path)
        if len(path) == 1:
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
        if isinstance(value, dict):
            _validate_overrides(value, prefix=path)


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

    _validate_overrides(overrides)
    exec_workflow._validate_user_config(overrides or {})

    model_ref = model.default_model_ref_by_dataset_kind[train_dataset_kind]
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
    profile = _resolve_profile_capability(profile_name)

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
