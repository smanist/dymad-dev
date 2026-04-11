"""Typed training compiler built on the existing execution workflow helpers."""

from __future__ import annotations

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

_ALLOWED_TOP_LEVEL_OVERRIDE_KEYS = {
    "criterion",
    "dataloader",
    "model",
    "phases",
    "plotting",
    "split",
    "transform_u",
    "transform_x",
}
_RUNTIME_OWNED_TOP_LEVEL_KEYS = {
    "data_valid",
    "path",
}
_ALLOWED_DATA_OVERRIDE_KEYS = {
    "double_precision",
}
_RUNTIME_OWNED_MODEL_KEYS = {
    "name",
}


def _raise_invalid(message: str, *, field_path: tuple[str, ...]) -> None:
    raise TrainingCompileValidationError(message, field_path=field_path)


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
        if len(path) == 1:
            if key in _RUNTIME_OWNED_TOP_LEVEL_KEYS:
                _raise_invalid(
                    f"overrides.{key} is runtime-owned and cannot be set by the caller",
                    field_path=("overrides", key),
                )
            if key not in _ALLOWED_TOP_LEVEL_OVERRIDE_KEYS and key != "data":
                _raise_invalid(
                    f"overrides.{key} is not supported by the user-mode compiler",
                    field_path=("overrides", key),
                )
        if path == ("data", "path") or path == ("data_valid", "path"):
            _raise_invalid(
                f"overrides.{'.'.join(path)} is runtime-owned and cannot be set by the caller",
                field_path=("overrides",) + path,
            )
        if path[:1] == ("data",) and len(path) == 2 and path[1] not in _ALLOWED_DATA_OVERRIDE_KEYS:
            _raise_invalid(
                f"overrides.data.{path[1]} is not supported by the user-mode compiler",
                field_path=("overrides",) + path,
            )
        if path[:1] == ("model",) and len(path) == 2 and path[1] in _RUNTIME_OWNED_MODEL_KEYS:
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


def compile_training_request(
    *,
    facade: FacadeOperations,
    request: TrainingRequest,
) -> CompiledTrainingRequest:
    try:
        model = resolve_model_capability(request.model_key)
    except ValueError as exc:
        raise TrainingCompileValidationError(
            str(exc),
            field_path=("model_key",),
        ) from exc

    train_dataset = facade.get_dataset(request.train_dataset_handle)
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
    if request.valid_dataset_handle is not None:
        valid_dataset = facade.get_dataset(request.valid_dataset_handle)
        valid_dataset_kind = _dataset_kind_from_record(valid_dataset)
        if valid_dataset_kind != train_dataset_kind:
            raise TrainingCompileValidationError(
                "train_dataset_handle and valid_dataset_handle must have the same dataset kind",
                field_path=("valid_dataset_handle",),
            )

    if request.reference_profile is not None:
        profile = _resolve_profile_capability(request.reference_profile)
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

    _validate_overrides(request.overrides)
    exec_workflow._validate_user_config(request.overrides or {})

    model_ref = model.default_model_ref_by_dataset_kind[train_dataset_kind]
    effective_run_name = exec_workflow._validate_run_name(
        request.run_name or exec_workflow._default_run_name(model_ref)
    )
    profile_name, effective_config = exec_workflow._effective_config(
        model_ref=model_ref,
        dataset_record=train_dataset,
        valid_dataset_record=valid_dataset,
        reference_profile=request.reference_profile,
        user_config=request.overrides,
        run_name=effective_run_name,
    )
    _, trainer_kind = exec_workflow._select_trainer(effective_config)
    profile = _resolve_profile_capability(profile_name)

    return CompiledTrainingRequest(
        request=request,
        model=model,
        profile=profile,
        model_ref=model_ref,
        train_dataset_kind=train_dataset_kind,
        valid_dataset_kind=valid_dataset_kind,
        effective_run_name=effective_run_name,
        effective_config=effective_config,
        trainer_kind=trainer_kind,
    )
