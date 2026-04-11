"""Deterministic training-intent compilation for MCP-friendly training requests."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from dymad.agent.exec.state import (
    DatasetInspection,
    ModelFamilyDescription,
    ReferenceProfileDescription,
)
from dymad.agent.exec.training_profiles import resolve_profile_name

AddressMode = Literal["nl", "structured", "both"]
MergeBehavior = Literal["replace", "deep_merge", "section", "phase_replace"]
ConflictBehavior = Literal["reject", "prefer_structured", "prefer_explicit"]

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?"
_RESERVED_CONFIG_PREFIXES = ("path", "data.path", "data_valid.path")
_KNOWN_BASE_TOKENS = ("kmsk", "kmm", "kbf", "ldm", "sdm", "lti", "km")
_SPECIAL_SUFFIX_TO_BASE = {
    "ldmg": "ldm",
    "sdmg": "sdm",
}
_MODEL_FAMILY_SYNONYMS = {
    "latent dynamics model": "ldm",
    "sequential dynamics model": "sdm",
    "linear time invariant": "lti",
    "linear time-invariant": "lti",
    "koopman bilinear form": "kbf",
}
_ACTIVATION_ALIASES = {
    "relu": "relu",
    "prelu": "prelu",
    "tanh": "tanh",
    "silu": "silu",
    "gelu": "gelu",
    "none": "none",
}
_WEIGHT_INIT_ALIASES = {
    "xavier uniform": "xavier_uniform",
    "xavier_uniform": "xavier_uniform",
    "zeros": "zeros",
}
_TRANSFORM_TYPE_ALIASES = {
    "identity": "identity",
    "scaler": "scaler",
    "standardize": "scaler",
    "standardise": "scaler",
    "delay": "delay",
    "delay embedding": "delay",
    "svd": "svd",
    "lift": "lift",
    "polynomial lift": "lift",
    "diffusion map": "dm",
    "diffmap": "dm",
    "vb diffusion map": "vbdm",
    "diffmapvb": "vbdm",
    "isomap": "isomap",
}


@dataclass(frozen=True)
class IntentFieldSpec:
    canonical_path: str
    value_type: str
    address_mode: AddressMode
    merge_behavior: MergeBehavior
    conflict_behavior: ConflictBehavior
    examples: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentTraceStep:
    rule_id: str
    source: str | None
    target: str
    value: Any


@dataclass(frozen=True)
class IntentRejection:
    code: str
    message: str


@dataclass(frozen=True)
class TrainingIntentDatasetCandidate:
    path: str | None = None
    handle: str | None = None
    format: str | None = None
    kind: str | None = None
    inspection: DatasetInspection | None = None


@dataclass(frozen=True)
class TrainingIntentInput:
    raw_request: str
    cwd: str | None = None
    candidate_dataset_paths: tuple[str, ...] = ()
    explicit_train_dataset_handle: str | None = None
    explicit_valid_dataset_handle: str | None = None
    model_families: tuple[ModelFamilyDescription, ...] = ()
    reference_profiles: tuple[ReferenceProfileDescription, ...] = ()
    override_payload: dict[str, Any] | None = None
    candidate_datasets: tuple[TrainingIntentDatasetCandidate, ...] = ()


@dataclass(frozen=True)
class ResolvedTrainingIntent:
    selected_train_dataset_path: str | None
    selected_train_dataset_handle: str | None
    selected_valid_dataset_path: str | None
    selected_valid_dataset_handle: str | None
    train_dataset_format: str | None
    train_dataset_kind: str | None
    valid_dataset_format: str | None
    valid_dataset_kind: str | None
    model_ref: str | None
    reference_profile: str | None
    config_overrides: dict[str, Any]
    phases_override: list[dict[str, Any]] | None
    artifact_root: str | None
    run_name: str | None
    seed: int | None
    device: str | None
    max_workers: int | None
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    unresolved_fields: tuple[str, ...]
    trace: tuple[IntentTraceStep, ...]
    rejection: IntentRejection | None = None

    @property
    def is_valid(self) -> bool:
        return self.rejection is None

    def structured_config(self) -> dict[str, Any]:
        config = copy.deepcopy(self.config_overrides)
        if self.phases_override is not None:
            config["phases"] = copy.deepcopy(self.phases_override)
        return config


TRAINING_INTENT_FIELD_REGISTRY: dict[str, IntentFieldSpec] = {
    "train_dataset": IntentFieldSpec(
        canonical_path="train_dataset",
        value_type="dataset_ref",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("lti.npz", "train dataset ./data/train.npz"),
    ),
    "valid_dataset": IntentFieldSpec(
        canonical_path="valid_dataset",
        value_type="dataset_ref",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("valid dataset ./data/valid.npz",),
    ),
    "dataset.format": IntentFieldSpec(
        canonical_path="dataset.format",
        value_type="str",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("npz",),
    ),
    "dataset.kind": IntentFieldSpec(
        canonical_path="dataset.kind",
        value_type="str",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("regular", "graph"),
    ),
    "artifact_root": IntentFieldSpec(
        canonical_path="artifact_root",
        value_type="path",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("./.dymad/artifacts",),
    ),
    "run_name": IntentFieldSpec(
        canonical_path="run_name",
        value_type="str",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("lti_dlti_run",),
    ),
    "seed": IntentFieldSpec(
        canonical_path="seed",
        value_type="int",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("seed 123",),
    ),
    "device": IntentFieldSpec(
        canonical_path="device",
        value_type="str",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("cuda", "cpu"),
    ),
    "max_workers": IntentFieldSpec(
        canonical_path="max_workers",
        value_type="int",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("max workers 4",),
    ),
    "model_ref": IntentFieldSpec(
        canonical_path="model_ref",
        value_type="str",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("dymad.models.collections:DLTI",),
        aliases=("model family", "model"),
    ),
    "reference_profile": IntentFieldSpec(
        canonical_path="reference_profile",
        value_type="str",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
        examples=("lti-regular-default",),
    ),
    "config.model": IntentFieldSpec(
        canonical_path="config.model",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.transform_x": IntentFieldSpec(
        canonical_path="config.transform_x",
        value_type="mapping|list",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
    ),
    "config.transform_u": IntentFieldSpec(
        canonical_path="config.transform_u",
        value_type="mapping|list",
        address_mode="both",
        merge_behavior="replace",
        conflict_behavior="prefer_structured",
    ),
    "config.split": IntentFieldSpec(
        canonical_path="config.split",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.dataloader": IntentFieldSpec(
        canonical_path="config.dataloader",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.criterion": IntentFieldSpec(
        canonical_path="config.criterion",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.plotting": IntentFieldSpec(
        canonical_path="config.plotting",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.cv": IntentFieldSpec(
        canonical_path="config.cv",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.log": IntentFieldSpec(
        canonical_path="config.log",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
    "config.phases": IntentFieldSpec(
        canonical_path="config.phases",
        value_type="list",
        address_mode="both",
        merge_behavior="phase_replace",
        conflict_behavior="prefer_structured",
    ),
    "config.training": IntentFieldSpec(
        canonical_path="config.training",
        value_type="mapping",
        address_mode="both",
        merge_behavior="deep_merge",
        conflict_behavior="prefer_structured",
    ),
}


def resolve_training_intent(intent_input: TrainingIntentInput) -> ResolvedTrainingIntent:
    request_text = intent_input.raw_request.strip()
    normalized_text = _normalize_text(request_text)
    traces: list[IntentTraceStep] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    unresolved_fields: list[str] = []

    candidates = _build_dataset_candidates(intent_input=intent_input)
    override_state = _normalize_override_payload(intent_input.override_payload, traces=traces)
    if override_state.rejection is not None:
        return _empty_result(
            traces=traces,
            assumptions=assumptions,
            warnings=warnings,
            unresolved_fields=unresolved_fields,
            rejection=override_state.rejection,
        )

    dataset_resolution = _resolve_datasets(
        text=request_text,
        normalized_text=normalized_text,
        intent_input=intent_input,
        candidates=candidates,
        override_state=override_state,
        traces=traces,
        assumptions=assumptions,
        warnings=warnings,
    )
    if dataset_resolution.rejection is not None:
        return _empty_result(
            traces=traces,
            assumptions=assumptions,
            warnings=warnings,
            unresolved_fields=unresolved_fields,
            rejection=dataset_resolution.rejection,
        )

    model_ref = override_state.runtime.get("model_ref")
    if model_ref is None:
        model_resolution = _resolve_model_family(
            normalized_text=normalized_text,
            model_families=intent_input.model_families,
            train_dataset_kind=dataset_resolution.train.kind,
            traces=traces,
        )
        if model_resolution.rejection is not None:
            return _empty_result(
                traces=traces,
                assumptions=assumptions,
                warnings=warnings,
                unresolved_fields=unresolved_fields,
                rejection=model_resolution.rejection,
            )
        model_ref = model_resolution.model_ref
    else:
        traces.append(
            IntentTraceStep(
                rule_id="override.model_ref",
                source="structured override",
                target="model_ref",
                value=model_ref,
            )
        )

    if model_ref is None:
        unresolved_fields.append("model_ref")

    reference_profile = override_state.runtime.get("reference_profile")
    if (
        reference_profile is None
        and model_ref is not None
        and dataset_resolution.train.kind is not None
    ):
        try:
            reference_profile = resolve_profile_name(
                model_ref=model_ref,
                dataset_kind=dataset_resolution.train.kind,
                reference_profile=None,
            )
            traces.append(
                IntentTraceStep(
                    rule_id="infer.reference_profile",
                    source=dataset_resolution.train.kind,
                    target="reference_profile",
                    value=reference_profile,
                )
            )
        except Exception:
            unresolved_fields.append("reference_profile")
    elif reference_profile is not None:
        traces.append(
            IntentTraceStep(
                rule_id="override.reference_profile",
                source="structured override",
                target="reference_profile",
                value=reference_profile,
            )
        )

    config_overrides = copy.deepcopy(override_state.config_overrides)
    phase_overrides = copy.deepcopy(override_state.phases_override)

    _merge_model_overrides(
        config_overrides=config_overrides,
        phase_overrides=phase_overrides,
        model_ref=model_ref,
        normalized_text=normalized_text,
        traces=traces,
    )
    _merge_transform_overrides(
        config_overrides=config_overrides,
        normalized_text=normalized_text,
        traces=traces,
    )
    _merge_split_dataloader_plotting(
        config_overrides=config_overrides,
        normalized_text=normalized_text,
        traces=traces,
    )
    _merge_criterion_overrides(
        config_overrides=config_overrides,
        normalized_text=normalized_text,
        traces=traces,
    )
    _merge_log_overrides(
        config_overrides=config_overrides,
        normalized_text=normalized_text,
        traces=traces,
    )

    if phase_overrides is None:
        phase_overrides = _parse_phase_overrides(normalized_text=normalized_text, traces=traces)

    runtime_values = _resolve_runtime_values(
        request_text=request_text,
        normalized_text=normalized_text,
        cwd=intent_input.cwd,
        override_state=override_state,
        traces=traces,
        assumptions=assumptions,
    )

    return ResolvedTrainingIntent(
        selected_train_dataset_path=dataset_resolution.train.path,
        selected_train_dataset_handle=dataset_resolution.train.handle,
        selected_valid_dataset_path=dataset_resolution.valid.path,
        selected_valid_dataset_handle=dataset_resolution.valid.handle,
        train_dataset_format=dataset_resolution.train.format,
        train_dataset_kind=dataset_resolution.train.kind,
        valid_dataset_format=dataset_resolution.valid.format,
        valid_dataset_kind=dataset_resolution.valid.kind,
        model_ref=model_ref,
        reference_profile=reference_profile,
        config_overrides=config_overrides,
        phases_override=phase_overrides,
        artifact_root=runtime_values["artifact_root"],
        run_name=runtime_values["run_name"],
        seed=runtime_values["seed"],
        device=runtime_values["device"],
        max_workers=runtime_values["max_workers"],
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
        unresolved_fields=tuple(dict.fromkeys(unresolved_fields)),
        trace=tuple(traces),
        rejection=None if not unresolved_fields else None,
    )


@dataclass(frozen=True)
class _OverrideState:
    runtime: dict[str, Any]
    config_overrides: dict[str, Any]
    phases_override: list[dict[str, Any]] | None
    rejection: IntentRejection | None = None


@dataclass(frozen=True)
class _DatasetSelection:
    path: str | None = None
    handle: str | None = None
    format: str | None = None
    kind: str | None = None
    inspection: DatasetInspection | None = None


@dataclass(frozen=True)
class _DatasetResolution:
    train: _DatasetSelection
    valid: _DatasetSelection
    rejection: IntentRejection | None = None


@dataclass(frozen=True)
class _ModelResolution:
    model_ref: str | None = None
    rejection: IntentRejection | None = None


def _empty_result(
    *,
    traces: list[IntentTraceStep],
    assumptions: list[str],
    warnings: list[str],
    unresolved_fields: list[str],
    rejection: IntentRejection,
) -> ResolvedTrainingIntent:
    return ResolvedTrainingIntent(
        selected_train_dataset_path=None,
        selected_train_dataset_handle=None,
        selected_valid_dataset_path=None,
        selected_valid_dataset_handle=None,
        train_dataset_format=None,
        train_dataset_kind=None,
        valid_dataset_format=None,
        valid_dataset_kind=None,
        model_ref=None,
        reference_profile=None,
        config_overrides={},
        phases_override=None,
        artifact_root=None,
        run_name=None,
        seed=None,
        device=None,
        max_workers=None,
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
        unresolved_fields=tuple(unresolved_fields),
        trace=tuple(traces),
        rejection=rejection,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _build_dataset_candidates(
    intent_input: TrainingIntentInput,
) -> list[TrainingIntentDatasetCandidate]:
    candidates = list(intent_input.candidate_datasets)
    seen_paths = {item.path for item in candidates if item.path is not None}
    for raw_path in intent_input.candidate_dataset_paths:
        path = str(Path(raw_path).expanduser())
        if path in seen_paths:
            continue
        candidates.append(
            TrainingIntentDatasetCandidate(
                path=path,
                format=_infer_dataset_format(path),
            )
        )
        seen_paths.add(path)
    return candidates


def _normalize_override_payload(
    payload: dict[str, Any] | None,
    *,
    traces: list[IntentTraceStep],
) -> _OverrideState:
    if payload is None:
        return _OverrideState(runtime={}, config_overrides={}, phases_override=None)

    runtime: dict[str, Any] = {}
    config_overrides: dict[str, Any] = {}
    phases_override: list[dict[str, Any]] | None = None
    explicit_config = payload.get("config")
    if explicit_config is not None:
        if not isinstance(explicit_config, dict):
            return _OverrideState(
                runtime={},
                config_overrides={},
                phases_override=None,
                rejection=IntentRejection(
                    code="invalid_override_config",
                    message="override payload 'config' must be a mapping",
                ),
            )
        config_overrides = copy.deepcopy(explicit_config)
        traces.append(
            IntentTraceStep(
                rule_id="override.config",
                source="structured override",
                target="config",
                value=copy.deepcopy(explicit_config),
            )
        )

    if "phases" in payload:
        if not isinstance(payload["phases"], list):
            return _OverrideState(
                runtime={},
                config_overrides={},
                phases_override=None,
                rejection=IntentRejection(
                    code="invalid_phase_override",
                    message="override payload 'phases' must be a list",
                ),
            )
        phases_override = copy.deepcopy(payload["phases"])
        traces.append(
            IntentTraceStep(
                rule_id="override.phases",
                source="structured override",
                target="config.phases",
                value=copy.deepcopy(phases_override),
            )
        )

    for key, value in payload.items():
        if key in {"config", "phases"}:
            continue
        if key in {
            "artifact_root",
            "run_name",
            "seed",
            "device",
            "max_workers",
            "model_ref",
            "reference_profile",
            "train_dataset_path",
            "valid_dataset_path",
            "train_dataset_handle",
            "valid_dataset_handle",
            "dataset.kind",
            "dataset.format",
        }:
            runtime[key] = copy.deepcopy(value)
            continue
        if key in {"path", "data.path", "data_valid.path"}:
            return _OverrideState(
                runtime={},
                config_overrides={},
                phases_override=None,
                rejection=IntentRejection(
                    code="reserved_runtime_path",
                    message=f"override field '{key}' is runtime-owned and cannot be set",
                ),
            )
        if "." in key:
            if any(
                key == prefix or key.startswith(f"{prefix}.")
                for prefix in _RESERVED_CONFIG_PREFIXES
            ):
                return _OverrideState(
                    runtime={},
                    config_overrides={},
                    phases_override=None,
                    rejection=IntentRejection(
                        code="reserved_runtime_path",
                        message=f"override field '{key}' is runtime-owned and cannot be set",
                    ),
                )
            if key.startswith("phases."):
                if phases_override is None:
                    phases_override = []
                set_by_dotted_key({"phases": phases_override}, key, copy.deepcopy(value))
            else:
                set_by_dotted_key(config_overrides, key, copy.deepcopy(value))
            traces.append(
                IntentTraceStep(
                    rule_id="override.dotted_key",
                    source="structured override",
                    target=key,
                    value=copy.deepcopy(value),
                )
            )
            continue
        if key.startswith("config_") and isinstance(value, dict):
            section_name = key.removeprefix("config_")
            config_overrides[section_name] = copy.deepcopy(value)
            traces.append(
                IntentTraceStep(
                    rule_id="override.config_section",
                    source="structured override",
                    target=f"config.{section_name}",
                    value=copy.deepcopy(value),
                )
            )
            continue
        config_overrides[key] = copy.deepcopy(value)
        traces.append(
            IntentTraceStep(
                rule_id="override.top_level",
                source="structured override",
                target=f"config.{key}",
                value=copy.deepcopy(value),
            )
        )
    return _OverrideState(
        runtime=runtime,
        config_overrides=config_overrides,
        phases_override=phases_override,
    )


def _resolve_datasets(
    *,
    text: str,
    normalized_text: str,
    intent_input: TrainingIntentInput,
    candidates: list[TrainingIntentDatasetCandidate],
    override_state: _OverrideState,
    traces: list[IntentTraceStep],
    assumptions: list[str],
    warnings: list[str],
) -> _DatasetResolution:
    del warnings
    inferred_kind = (
        override_state.runtime.get("dataset.kind")
        or _infer_dataset_kind_from_text(normalized_text)
        or "regular"
    )
    train_selection = _select_dataset(
        role="train",
        text=text,
        normalized_text=normalized_text,
        candidates=candidates,
        explicit_handle=override_state.runtime.get("train_dataset_handle")
        or intent_input.explicit_train_dataset_handle,
        explicit_path=override_state.runtime.get("train_dataset_path"),
        traces=traces,
        assumptions=assumptions,
        fallback_kind=inferred_kind,
    )
    if isinstance(train_selection, IntentRejection):
        return _DatasetResolution(
            train=_DatasetSelection(),
            valid=_DatasetSelection(),
            rejection=train_selection,
        )

    valid_selection = _select_dataset(
        role="valid",
        text=text,
        normalized_text=normalized_text,
        candidates=candidates,
        explicit_handle=override_state.runtime.get("valid_dataset_handle")
        or intent_input.explicit_valid_dataset_handle,
        explicit_path=override_state.runtime.get("valid_dataset_path"),
        traces=traces,
        assumptions=assumptions,
        fallback_kind=override_state.runtime.get("dataset.kind")
        or train_selection.kind
        or inferred_kind,
    )
    if isinstance(valid_selection, IntentRejection):
        return _DatasetResolution(
            train=train_selection,
            valid=_DatasetSelection(),
            rejection=valid_selection,
        )

    if train_selection.path is None and train_selection.handle is None:
        return _DatasetResolution(
            train=train_selection,
            valid=valid_selection,
            rejection=IntentRejection(
                code="train_dataset_unresolved",
                message="could not resolve a training dataset from the request",
            ),
        )
    return _DatasetResolution(train=train_selection, valid=valid_selection)


def _select_dataset(
    *,
    role: Literal["train", "valid"],
    text: str,
    normalized_text: str,
    candidates: list[TrainingIntentDatasetCandidate],
    explicit_handle: str | None,
    explicit_path: str | None,
    traces: list[IntentTraceStep],
    assumptions: list[str],
    fallback_kind: str,
) -> _DatasetSelection | IntentRejection:
    if explicit_handle is not None:
        for candidate in candidates:
            if candidate.handle == explicit_handle:
                traces.append(
                    IntentTraceStep(
                        rule_id=f"dataset.{role}.explicit_handle",
                        source=explicit_handle,
                        target=f"{role}_dataset_handle",
                        value=explicit_handle,
                    )
                )
                return _candidate_to_selection(candidate, fallback_kind=fallback_kind)
        traces.append(
            IntentTraceStep(
                rule_id=f"dataset.{role}.explicit_handle",
                source=explicit_handle,
                target=f"{role}_dataset_handle",
                value=explicit_handle,
            )
        )
        return _DatasetSelection(handle=explicit_handle, kind=fallback_kind)

    if explicit_path is not None:
        matched = _match_dataset_path(explicit_path, candidates)
        if matched is None:
            return IntentRejection(
                code=f"{role}_dataset_not_found",
                message=f"could not resolve {role} dataset path '{explicit_path}'",
            )
        traces.append(
            IntentTraceStep(
                rule_id=f"dataset.{role}.explicit_path",
                source=explicit_path,
                target=f"{role}_dataset_path",
                value=matched.path,
            )
        )
        return _candidate_to_selection(matched, fallback_kind=fallback_kind)

    mentions = _extract_dataset_mentions(text=text, role=role)
    if mentions:
        resolved = []
        for mention in mentions:
            matched = _match_dataset_path(mention, candidates)
            if matched is not None:
                resolved.append(matched)
        if len(resolved) == 1:
            traces.append(
                IntentTraceStep(
                    rule_id=f"dataset.{role}.prompt_mention",
                    source=mentions[0],
                    target=f"{role}_dataset_path",
                    value=resolved[0].path,
                )
            )
            return _candidate_to_selection(resolved[0], fallback_kind=fallback_kind)
        if len(resolved) > 1:
            return IntentRejection(
                code=f"{role}_dataset_ambiguous",
                message=f"multiple dataset candidates matched the {role} dataset mention",
            )

    if role == "valid":
        return _DatasetSelection()

    if len(candidates) == 1:
        assumptions.append("selected the only candidate dataset as the training dataset")
        traces.append(
            IntentTraceStep(
                rule_id="dataset.train.single_candidate",
                source=normalized_text,
                target="train_dataset_path",
                value=candidates[0].path or candidates[0].handle,
            )
        )
        return _candidate_to_selection(candidates[0], fallback_kind=fallback_kind)

    if len(candidates) > 1:
        return IntentRejection(
            code="train_dataset_ambiguous",
            message="multiple dataset candidates are available; specify which file to use for training",
        )
    return _DatasetSelection()


def _extract_dataset_mentions(*, text: str, role: Literal["train", "valid"]) -> list[str]:
    mentions = []
    path_matches = re.findall(r"(?:(?:\.{1,2}/|/)?[\w./-]+\.npz)", text)
    for match in path_matches:
        lowered = text.lower()
        position = lowered.find(match.lower())
        window_start = max(0, position - 32)
        context = lowered[window_start : position + len(match) + 16]
        if role == "valid":
            if "valid" in context or "validation" in context:
                mentions.append(match)
        else:
            if "valid" not in context and "validation" not in context:
                mentions.append(match)
    return mentions


def _match_dataset_path(
    needle: str,
    candidates: list[TrainingIntentDatasetCandidate],
) -> TrainingIntentDatasetCandidate | None:
    normalized_needle = needle.strip().strip("'\"")
    for candidate in candidates:
        if candidate.path is None:
            continue
        candidate_name = Path(candidate.path).name
        if normalized_needle in {candidate.path, candidate_name}:
            return candidate
        if candidate.path.endswith(normalized_needle):
            return candidate
    return None


def _candidate_to_selection(
    candidate: TrainingIntentDatasetCandidate,
    *,
    fallback_kind: str,
) -> _DatasetSelection:
    kind = candidate.kind
    if kind is None and candidate.inspection is not None:
        kind = candidate.inspection.kind
    if kind is None:
        kind = fallback_kind
    fmt = candidate.format
    if fmt is None and candidate.path is not None:
        fmt = _infer_dataset_format(candidate.path)
    return _DatasetSelection(
        path=candidate.path,
        handle=candidate.handle,
        format=fmt,
        kind=kind,
        inspection=candidate.inspection,
    )


def _infer_dataset_format(path: str | None) -> str | None:
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    return "npz" if suffix == ".npz" else None


def _infer_dataset_kind_from_text(normalized_text: str) -> str | None:
    if (
        "graph" in normalized_text
        or "node dynamics" in normalized_text
        or "per-node" in normalized_text
    ):
        return "graph"
    if "regular" in normalized_text or "non-graph" in normalized_text:
        return "regular"
    return None


def _resolve_model_family(
    *,
    normalized_text: str,
    model_families: tuple[ModelFamilyDescription, ...],
    train_dataset_kind: str | None,
    traces: list[IntentTraceStep],
) -> _ModelResolution:
    if not model_families:
        return _ModelResolution(
            rejection=IntentRejection(
                code="no_model_families",
                message="no model families are available for intent resolution",
            )
        )

    exact_matches = _exact_model_name_matches(normalized_text, model_families)
    if len(exact_matches) > 1:
        return _ModelResolution(
            rejection=IntentRejection(
                code="model_family_conflict",
                message="multiple exact model-family names were mentioned in the request",
            )
        )
    if len(exact_matches) == 1:
        traces.append(
            IntentTraceStep(
                rule_id="model.exact_name",
                source=exact_matches[0].name,
                target="model_ref",
                value=exact_matches[0].model_ref,
            )
        )
        return _ModelResolution(model_ref=exact_matches[0].model_ref)

    family_token = _extract_family_token(normalized_text)
    time_domain = _extract_time_domain(normalized_text)
    graph_mode = _extract_graph_mode(normalized_text)
    candidates = list(model_families)
    if family_token is not None:
        candidates = [
            item for item in candidates if _family_base_token(item.model_ref) == family_token
        ]
    if time_domain is not None:
        candidates = [item for item in candidates if item.time_domain == time_domain]
    if graph_mode is not None:
        candidates = [item for item in candidates if item.graph_mode == graph_mode]

    if train_dataset_kind == "regular":
        candidates = [item for item in candidates if not item.expects_graph_data]
    elif train_dataset_kind == "graph":
        graph_candidates = [item for item in candidates if item.expects_graph_data]
        if graph_candidates:
            candidates = graph_candidates

    if len(candidates) == 1:
        traces.append(
            IntentTraceStep(
                rule_id="model.filtered_match",
                source=family_token or f"{time_domain}/{graph_mode}",
                target="model_ref",
                value=candidates[0].model_ref,
            )
        )
        return _ModelResolution(model_ref=candidates[0].model_ref)
    if not candidates:
        return _ModelResolution(
            rejection=IntentRejection(
                code="model_family_unresolved",
                message="the request did not resolve to a known compatible model family",
            )
        )
    return _ModelResolution(
        rejection=IntentRejection(
            code="model_family_ambiguous",
            message="the request matches multiple model families; specify the family or exact model name",
        )
    )


def _exact_model_name_matches(
    normalized_text: str,
    model_families: tuple[ModelFamilyDescription, ...],
) -> list[ModelFamilyDescription]:
    matches = []
    for family in model_families:
        public_name = family.model_ref.split(":", 1)[1].lower()
        if public_name in _KNOWN_BASE_TOKENS:
            continue
        if re.search(rf"\b{re.escape(public_name.lower())}\b", normalized_text):
            matches.append(family)
    return matches


def _extract_family_token(normalized_text: str) -> str | None:
    for phrase, token in _MODEL_FAMILY_SYNONYMS.items():
        if phrase in normalized_text:
            return token
    for token in _KNOWN_BASE_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", normalized_text):
            return token
    return None


def _extract_time_domain(normalized_text: str) -> str | None:
    has_discrete = any(
        phrase in normalized_text for phrase in ("discrete-time", "discrete time", "discrete")
    )
    has_continuous = any(
        phrase in normalized_text for phrase in ("continuous-time", "continuous time", "continuous")
    )
    if has_discrete and has_continuous:
        return None
    if has_discrete:
        return "discrete"
    if has_continuous:
        return "continuous"
    return None


def _extract_graph_mode(normalized_text: str) -> str | None:
    if "regular" in normalized_text or "non-graph" in normalized_text:
        return "none"
    if (
        "node dynamics" in normalized_text
        or "node-level" in normalized_text
        or "per-node" in normalized_text
    ):
        return "node"
    if "graph" in normalized_text:
        return "graph"
    return None


def _family_base_token(model_ref: str) -> str:
    public_name = model_ref.split(":", 1)[1].lower()
    for suffix, base in _SPECIAL_SUFFIX_TO_BASE.items():
        if public_name.endswith(suffix):
            return base
    for token in _KNOWN_BASE_TOKENS:
        if public_name.endswith(token):
            return token
    return public_name


def _merge_model_overrides(
    *,
    config_overrides: dict[str, Any],
    phase_overrides: list[dict[str, Any]] | None,
    model_ref: str | None,
    normalized_text: str,
    traces: list[IntentTraceStep],
) -> None:
    del phase_overrides
    model_cfg = config_overrides.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        return
    if (
        "trivial encoder and decoder" in normalized_text
        or "identity encoder and decoder" in normalized_text
    ):
        model_cfg.setdefault("encoder_layers", 0)
        model_cfg.setdefault("decoder_layers", 0)
        traces.append(
            IntentTraceStep(
                rule_id="model.trivial_autoencoder",
                source="trivial encoder and decoder",
                target="config.model",
                value={"encoder_layers": 0, "decoder_layers": 0},
            )
        )

    dimension_patterns = [
        (r"\b(\d+)\s+koopman (?:states?|dims?|dimensions?)\b", "koopman_dimension"),
        (r"\bkoopman (?:dimension|dim)\s+(\d+)\b", "koopman_dimension"),
        (r"\b(\d+)\s+latent (?:states?|dims?|dimensions?)\b", "latent_dimension"),
        (r"\blatent (?:dimension|dim)\s+(\d+)\b", "latent_dimension"),
        (r"\bhidden (?:dimension|size)\s+(\d+)\b", "hidden_dimension"),
        (r"\b(\d+)\s+hidden (?:units|features)\b", "hidden_dimension"),
        (r"\b(\d+)\s+encoder layers?\b", "encoder_layers"),
        (r"\b(\d+)\s+decoder layers?\b", "decoder_layers"),
        (r"\b(\d+)\s+processor layers?\b", "processor_layers"),
    ]
    for pattern, key in dimension_patterns:
        match = re.search(pattern, normalized_text)
        if match is None or key in model_cfg:
            continue
        model_cfg[key] = int(match.group(1))
        traces.append(
            IntentTraceStep(
                rule_id=f"model.{key}",
                source=match.group(0),
                target=f"config.model.{key}",
                value=int(match.group(1)),
            )
        )

    state_match = re.search(r"\b(\d+)\s+states?\b", normalized_text)
    if state_match is not None:
        state_dim = int(state_match.group(1))
        dimension_key = "latent_dimension"
        if model_ref is not None and _family_base_token(model_ref) in {
            "kbf",
            "lti",
            "km",
            "kmm",
            "kmsk",
        }:
            dimension_key = "koopman_dimension"
        model_cfg.setdefault(dimension_key, state_dim)
        traces.append(
            IntentTraceStep(
                rule_id="model.state_dimension",
                source=state_match.group(0),
                target=f"config.model.{dimension_key}",
                value=state_dim,
            )
        )

    for alias, canonical in _ACTIVATION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)} activation\b", normalized_text):
            model_cfg.setdefault("activation", canonical)
            traces.append(
                IntentTraceStep(
                    rule_id="model.activation",
                    source=alias,
                    target="config.model.activation",
                    value=canonical,
                )
            )
            break

    for alias, canonical in _WEIGHT_INIT_ALIASES.items():
        if alias in normalized_text and "init" in normalized_text:
            model_cfg.setdefault("weight_init", canonical)
            traces.append(
                IntentTraceStep(
                    rule_id="model.weight_init",
                    source=alias,
                    target="config.model.weight_init",
                    value=canonical,
                )
            )
            break

    if "without constant term" in normalized_text or "no constant term" in normalized_text:
        model_cfg.setdefault("const_term", False)
        traces.append(
            IntentTraceStep(
                rule_id="model.const_term.false",
                source="without constant term",
                target="config.model.const_term",
                value=False,
            )
        )
    elif "with constant term" in normalized_text or "include constant term" in normalized_text:
        model_cfg.setdefault("const_term", True)
        traces.append(
            IntentTraceStep(
                rule_id="model.const_term.true",
                source="with constant term",
                target="config.model.const_term",
                value=True,
            )
        )

    order_match = re.search(r"\b(linear|quadratic|cubic)\s+input order\b", normalized_text)
    if order_match is not None:
        model_cfg.setdefault("input_order", order_match.group(1))
        traces.append(
            IntentTraceStep(
                rule_id="model.input_order",
                source=order_match.group(0),
                target="config.model.input_order",
                value=order_match.group(1),
            )
        )

    gcl_match = re.search(r"\b(sage|gcn|gat)\s+(?:graph conv|gcl)\b", normalized_text)
    if gcl_match is not None:
        model_cfg.setdefault("gcl", gcl_match.group(1))
        traces.append(
            IntentTraceStep(
                rule_id="model.gcl",
                source=gcl_match.group(0),
                target="config.model.gcl",
                value=gcl_match.group(1),
            )
        )

    ae_match = re.search(r"\bautoencoder type\s+([a-z0-9_]+)\b", normalized_text)
    if ae_match is not None:
        model_cfg.setdefault("autoencoder_type", ae_match.group(1))
        traces.append(
            IntentTraceStep(
                rule_id="model.autoencoder_type",
                source=ae_match.group(0),
                target="config.model.autoencoder_type",
                value=ae_match.group(1),
            )
        )
    proc_match = re.search(r"\bprocessor type\s+([a-z0-9_]+)\b", normalized_text)
    if proc_match is not None:
        model_cfg.setdefault("processor_type", proc_match.group(1))
        traces.append(
            IntentTraceStep(
                rule_id="model.processor_type",
                source=proc_match.group(0),
                target="config.model.processor_type",
                value=proc_match.group(1),
            )
        )
    if not model_cfg:
        config_overrides.pop("model", None)


def _merge_transform_overrides(
    *,
    config_overrides: dict[str, Any],
    normalized_text: str,
    traces: list[IntentTraceStep],
) -> None:
    transform_x = _parse_transform_spec(normalized_text=normalized_text, role="x")
    transform_u = _parse_transform_spec(normalized_text=normalized_text, role="u")
    if (
        "trivial encoder and decoder" in normalized_text
        or "identity encoder and decoder" in normalized_text
    ):
        transform_x = transform_x or {"type": "identity"}
        transform_u = transform_u or {"type": "identity"}

    if transform_x is not None and "transform_x" not in config_overrides:
        config_overrides["transform_x"] = transform_x
        traces.append(
            IntentTraceStep(
                rule_id="transform.x",
                source="state transform prose",
                target="config.transform_x",
                value=copy.deepcopy(transform_x),
            )
        )
    if transform_u is not None and "transform_u" not in config_overrides:
        config_overrides["transform_u"] = transform_u
        traces.append(
            IntentTraceStep(
                rule_id="transform.u",
                source="control transform prose",
                target="config.transform_u",
                value=copy.deepcopy(transform_u),
            )
        )


def _parse_transform_spec(
    *, normalized_text: str, role: Literal["x", "u"]
) -> dict[str, Any] | list[dict[str, Any]] | None:
    section_markers = {
        "x": ("state transform", "transform_x", "state transforms", "states with"),
        "u": ("control transform", "transform_u", "control transforms", "controls with"),
    }
    generic_identity = (
        "identity transforms" in normalized_text
        or (
            role == "x"
            and ("identity state" in normalized_text or "identity transform_x" in normalized_text)
        )
        or (
            role == "u"
            and ("identity control" in normalized_text or "identity transform_u" in normalized_text)
        )
    )
    stages: list[tuple[int, dict[str, Any]]] = []
    if generic_identity:
        stages.append((normalized_text.find("identity"), {"type": "identity"}))
    if role == "x" and (
        "standardize states" in normalized_text or "state scaler" in normalized_text
    ):
        stages.append(
            (
                _first_index(normalized_text, "standardize states", "state scaler"),
                {"type": "scaler", "mode": "std"},
            )
        )
    if role == "u" and (
        "standardize controls" in normalized_text or "control scaler" in normalized_text
    ):
        stages.append(
            (
                _first_index(normalized_text, "standardize controls", "control scaler"),
                {"type": "scaler", "mode": "std"},
            )
        )
    if any(marker in normalized_text for marker in section_markers[role]):
        for alias, canonical in _TRANSFORM_TYPE_ALIASES.items():
            if alias not in normalized_text:
                continue
            if canonical == "identity" and generic_identity:
                continue
            index = normalized_text.find(alias)
            stage: dict[str, Any] = {"type": canonical}
            if canonical == "scaler":
                mode_match = re.search(
                    rf"{re.escape(alias)}(?: mode)?\s+(std|01|-11)", normalized_text
                )
                if mode_match is not None:
                    stage["mode"] = mode_match.group(1)
                elif "standardize" in alias or "standardise" in alias:
                    stage["mode"] = "std"
            elif canonical == "delay":
                delay_match = re.search(r"\bdelay(?: embedding)?\s+(\d+)\b", normalized_text)
                if delay_match is not None:
                    stage["delay"] = int(delay_match.group(1))
            elif canonical == "svd":
                order_match = re.search(r"\bsvd(?: order| rank)?\s+(\d+)\b", normalized_text)
                if order_match is not None:
                    stage["order"] = int(order_match.group(1))
            elif canonical == "lift":
                if "poly" in normalized_text or "polynomial" in normalized_text:
                    stage["fobs"] = "poly"
                elif "mixed" in normalized_text:
                    stage["fobs"] = "mixed"
            stages.append((index, stage))
    if not stages:
        return None
    deduped: list[dict[str, Any]] = []
    seen = set()
    for _, stage in sorted(stages, key=lambda item: item[0]):
        signature = tuple(sorted(stage.items()))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(stage)
    if len(deduped) == 1:
        return deduped[0]
    return deduped


def _first_index(text: str, *needles: str) -> int:
    indices = [text.find(needle) for needle in needles if needle in text]
    return min(indices) if indices else 0


def _merge_split_dataloader_plotting(
    *,
    config_overrides: dict[str, Any],
    normalized_text: str,
    traces: list[IntentTraceStep],
) -> None:
    train_frac_match = re.search(rf"\btrain fraction\s+({_FLOAT_PATTERN})\b", normalized_text)
    if train_frac_match is None:
        train_frac_match = re.search(
            rf"\b({_FLOAT_PATTERN})\s*(?:train split|train fraction)\b", normalized_text
        )
    if train_frac_match is not None and "split" not in config_overrides:
        config_overrides["split"] = {"train_frac": float(train_frac_match.group(1))}
        traces.append(
            IntentTraceStep(
                rule_id="split.train_frac",
                source=train_frac_match.group(0),
                target="config.split.train_frac",
                value=float(train_frac_match.group(1)),
            )
        )

    batch_match = re.search(r"\bbatch size\s+(\d+)\b", normalized_text)
    if batch_match is not None:
        dataloader = config_overrides.setdefault("dataloader", {})
        if isinstance(dataloader, dict):
            dataloader.setdefault("batch_size", int(batch_match.group(1)))
            traces.append(
                IntentTraceStep(
                    rule_id="dataloader.batch_size",
                    source=batch_match.group(0),
                    target="config.dataloader.batch_size",
                    value=int(batch_match.group(1)),
                )
            )

    if "without shuffling" in normalized_text or "no shuffle" in normalized_text:
        dataloader = config_overrides.setdefault("dataloader", {})
        if isinstance(dataloader, dict):
            dataloader.setdefault("shuffle", False)
            traces.append(
                IntentTraceStep(
                    rule_id="dataloader.shuffle.false",
                    source="no shuffle",
                    target="config.dataloader.shuffle",
                    value=False,
                )
            )
    elif "shuffle" in normalized_text and "dataloader" in normalized_text:
        dataloader = config_overrides.setdefault("dataloader", {})
        if isinstance(dataloader, dict):
            dataloader.setdefault("shuffle", True)
            traces.append(
                IntentTraceStep(
                    rule_id="dataloader.shuffle.true",
                    source="shuffle",
                    target="config.dataloader.shuffle",
                    value=True,
                )
            )

    plotting = config_overrides.get("plotting")
    if plotting is None:
        plotting = {}
        config_overrides["plotting"] = plotting
    if isinstance(plotting, dict):
        if "disable prediction plot" in normalized_text or "no prediction plot" in normalized_text:
            plotting.setdefault("prediction", False)
            traces.append(
                IntentTraceStep(
                    rule_id="plotting.prediction.false",
                    source="disable prediction plot",
                    target="config.plotting.prediction",
                    value=False,
                )
            )
        elif "prediction plot" in normalized_text or "plot predictions" in normalized_text:
            plotting.setdefault("prediction", True)
            traces.append(
                IntentTraceStep(
                    rule_id="plotting.prediction.true",
                    source="plot predictions",
                    target="config.plotting.prediction",
                    value=True,
                )
            )

        xidx_match = re.search(r"\bxidx\s+([\d,\s]+)\b", normalized_text)
        if xidx_match is not None:
            plotting.setdefault("xidx", _parse_index_list(xidx_match.group(1)))
            traces.append(
                IntentTraceStep(
                    rule_id="plotting.xidx",
                    source=xidx_match.group(0),
                    target="config.plotting.xidx",
                    value=_parse_index_list(xidx_match.group(1)),
                )
            )
        uidx_match = re.search(r"\buidx\s+([\d,\s]+)\b", normalized_text)
        if uidx_match is not None:
            plotting.setdefault("uidx", _parse_index_list(uidx_match.group(1)))
            traces.append(
                IntentTraceStep(
                    rule_id="plotting.uidx",
                    source=uidx_match.group(0),
                    target="config.plotting.uidx",
                    value=_parse_index_list(uidx_match.group(1)),
                )
            )
    if not plotting:
        config_overrides.pop("plotting", None)


def _parse_index_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _merge_criterion_overrides(
    *,
    config_overrides: dict[str, Any],
    normalized_text: str,
    traces: list[IntentTraceStep],
) -> None:
    criterion = config_overrides.get("criterion")
    if criterion is None:
        criterion = {}
        config_overrides["criterion"] = criterion
    if not isinstance(criterion, dict):
        return
    dynamics_match = re.search(rf"\bdynamics weight\s+({_FLOAT_PATTERN})\b", normalized_text)
    if dynamics_match is not None:
        criterion.setdefault("dynamics", {"weight": float(dynamics_match.group(1))})
        traces.append(
            IntentTraceStep(
                rule_id="criterion.dynamics.weight",
                source=dynamics_match.group(0),
                target="config.criterion.dynamics.weight",
                value=float(dynamics_match.group(1)),
            )
        )
    recon_match = re.search(
        rf"\brecon(?:struction)? weight\s+({_FLOAT_PATTERN})\b", normalized_text
    )
    if recon_match is not None:
        criterion.setdefault("recon", {"weight": float(recon_match.group(1))})
        traces.append(
            IntentTraceStep(
                rule_id="criterion.recon.weight",
                source=recon_match.group(0),
                target="config.criterion.recon.weight",
                value=float(recon_match.group(1)),
            )
        )
    if not criterion:
        config_overrides.pop("criterion", None)


def _merge_log_overrides(
    *,
    config_overrides: dict[str, Any],
    normalized_text: str,
    traces: list[IntentTraceStep],
) -> None:
    log_cfg = config_overrides.get("log")
    if log_cfg is None:
        log_cfg = {}
        config_overrides["log"] = log_cfg
    if not isinstance(log_cfg, dict):
        return
    if "debug logging" in normalized_text:
        log_cfg.setdefault("level", "debug")
        traces.append(
            IntentTraceStep(
                rule_id="log.level.debug",
                source="debug logging",
                target="config.log.level",
                value="debug",
            )
        )
    elif "info logging" in normalized_text:
        log_cfg.setdefault("level", "info")
        traces.append(
            IntentTraceStep(
                rule_id="log.level.info",
                source="info logging",
                target="config.log.level",
                value="info",
            )
        )
    if "stdout logging" in normalized_text:
        log_cfg.setdefault("stdout", True)
        traces.append(
            IntentTraceStep(
                rule_id="log.stdout.true",
                source="stdout logging",
                target="config.log.stdout",
                value=True,
            )
        )
    if not log_cfg:
        config_overrides.pop("log", None)


def _parse_phase_overrides(
    *,
    normalized_text: str,
    traces: list[IntentTraceStep],
) -> list[dict[str, Any]] | None:
    phase_specs: list[tuple[int, dict[str, Any]]] = []
    phase_patterns = [
        ("Linear", re.compile(r"\blinear (?:fit|warm ?start|phase|optimizer)\b")),
        ("Weak", re.compile(r"\bweak(?: form)?\b")),
        ("NODE", re.compile(r"\bnode\b|\bneural ode\b")),
        ("linear_solve", re.compile(r"\blinear solve\b")),
    ]
    for name, pattern in phase_patterns:
        for match in pattern.finditer(normalized_text):
            if name == "linear_solve":
                phase_specs.append(
                    (
                        match.start(),
                        {
                            "type": "linear_solve",
                            "name": "LinearSolve",
                            "method": "full",
                        },
                    )
                )
            else:
                phase_specs.append(
                    (
                        match.start(),
                        {
                            "type": "optimizer",
                            "name": name,
                            "trainer": name,
                        },
                    )
                )
            break
    if not phase_specs:
        return None
    phase_specs.sort(key=lambda item: item[0])
    phases = [copy.deepcopy(spec) for _, spec in phase_specs]
    phase_defaults = _parse_global_phase_defaults(normalized_text)
    if phase_defaults:
        for phase in phases:
            if phase.get("type") != "optimizer":
                continue
            for key, value in phase_defaults.items():
                phase.setdefault(key, value)
    traces.append(
        IntentTraceStep(
            rule_id="phases.from_prose",
            source="training phase prose",
            target="config.phases",
            value=copy.deepcopy(phases),
        )
    )
    return phases


def _parse_global_phase_defaults(normalized_text: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    lr_match = re.search(rf"\blearning rate\s+({_FLOAT_PATTERN})\b", normalized_text)
    if lr_match is not None:
        defaults["learning_rate"] = float(lr_match.group(1))
    decay_match = re.search(rf"\bdecay rate\s+({_FLOAT_PATTERN})\b", normalized_text)
    if decay_match is not None:
        defaults["decay_rate"] = float(decay_match.group(1))
    save_match = re.search(r"\bsave (?:every|interval)\s+(\d+)\b", normalized_text)
    if save_match is not None:
        defaults["save_interval"] = int(save_match.group(1))
    epochs_match = re.search(r"\bfor\s+(\d+)\s+epochs?\b", normalized_text)
    if epochs_match is not None:
        defaults["n_epochs"] = int(epochs_match.group(1))
    return defaults


def _resolve_runtime_values(
    *,
    request_text: str,
    normalized_text: str,
    cwd: str | None,
    override_state: _OverrideState,
    traces: list[IntentTraceStep],
    assumptions: list[str],
) -> dict[str, Any]:
    runtime = {
        "artifact_root": override_state.runtime.get("artifact_root"),
        "run_name": override_state.runtime.get("run_name"),
        "seed": override_state.runtime.get("seed"),
        "device": override_state.runtime.get("device"),
        "max_workers": override_state.runtime.get("max_workers"),
    }
    if runtime["artifact_root"] is None:
        artifact_match = re.search(
            r"(?:artifact root|artifacts?)\s+([./\w-]+)", request_text, flags=re.I
        )
        if artifact_match is not None:
            runtime["artifact_root"] = artifact_match.group(1)
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.artifact_root.prose",
                    source=artifact_match.group(0),
                    target="artifact_root",
                    value=artifact_match.group(1),
                )
            )
        else:
            base = Path(cwd) if cwd is not None else Path(".")
            runtime["artifact_root"] = str(base / ".dymad" / "artifacts")
            assumptions.append(
                "defaulted artifact_root to ./.dymad/artifacts under the working directory"
            )
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.artifact_root.default",
                    source=cwd,
                    target="artifact_root",
                    value=runtime["artifact_root"],
                )
            )

    if runtime["run_name"] is None:
        run_name_match = re.search(r"\brun name\s+([a-zA-Z0-9_.-]+)\b", request_text, flags=re.I)
        if run_name_match is not None:
            runtime["run_name"] = run_name_match.group(1)
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.run_name.prose",
                    source=run_name_match.group(0),
                    target="run_name",
                    value=run_name_match.group(1),
                )
            )

    if runtime["seed"] is None:
        seed_match = re.search(r"\bseed\s+(\d+)\b", normalized_text)
        if seed_match is not None:
            runtime["seed"] = int(seed_match.group(1))
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.seed.prose",
                    source=seed_match.group(0),
                    target="seed",
                    value=int(seed_match.group(1)),
                )
            )

    if runtime["device"] is None:
        if re.search(r"\bcuda\b|\bgpu\b", normalized_text):
            runtime["device"] = "cuda"
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.device.cuda",
                    source="cuda/gpu",
                    target="device",
                    value="cuda",
                )
            )
        elif re.search(r"\bcpu\b", normalized_text):
            runtime["device"] = "cpu"
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.device.cpu",
                    source="cpu",
                    target="device",
                    value="cpu",
                )
            )

    if runtime["max_workers"] is None:
        workers_match = re.search(r"\bmax workers\s+(\d+)\b", normalized_text)
        if workers_match is not None:
            runtime["max_workers"] = int(workers_match.group(1))
            traces.append(
                IntentTraceStep(
                    rule_id="runtime.max_workers.prose",
                    source=workers_match.group(0),
                    target="max_workers",
                    value=int(workers_match.group(1)),
                )
            )

    return runtime


def set_by_dotted_key(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    curr: dict[str, Any] | list[Any] = d
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        next_is_index = next_part.isdigit()
        if isinstance(curr, list):
            list_curr = cast(list[Any], curr)
            list_index = int(part)
            while len(list_curr) <= list_index:
                list_curr.append([] if next_is_index else {})
            curr = list_curr[list_index]
            continue
        if part not in curr or not isinstance(curr[part], (dict, list)):
            curr[part] = [] if next_is_index else {}
        next_value = curr[part]
        assert isinstance(next_value, (dict, list))
        curr = next_value

    last = parts[-1]
    if isinstance(curr, list):
        list_index = int(last)
        while len(curr) <= list_index:
            curr.append(None)
        curr[list_index] = value
        return
    curr[last] = value
