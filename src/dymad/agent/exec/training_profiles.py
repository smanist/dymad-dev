"""Explicit reference profiles for MCP-driven training runs."""

from __future__ import annotations

from typing import Any, cast

from dymad.agent import registry as _registry

PROFILE_REGISTRY: dict[str, dict[str, Any]] = _registry.profile_registry_payload()
PROFILE_ALIASES: dict[tuple[str, str], str] = cast(
    dict[tuple[str, str], str],
    _registry.profile_alias_mapping(),
)


def available_profiles() -> list[str]:
    return _registry.available_profiles()


def resolve_profile_name(
    *,
    model_ref: str,
    dataset_kind: str,
    reference_profile: str | None,
) -> str:
    return _registry.resolve_profile_name(
        model_ref=model_ref,
        dataset_kind=cast(_registry.DatasetKind, dataset_kind),
        reference_profile=reference_profile,
    )


def profile_config(profile_name: str) -> dict[str, Any]:
    return _registry.profile_config(profile_name)
