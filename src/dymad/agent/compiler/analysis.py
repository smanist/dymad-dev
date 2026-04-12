"""Typed analysis compiler scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dymad.agent.registry import AnalysisCapability, resolve_analysis_capability


class AnalysisCompileValidationError(ValueError):
    """Raised when an analysis request cannot be compiled safely."""

    def __init__(self, message: str, *, field_path: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.field_path = field_path


@dataclass(frozen=True)
class AnalysisRequest:
    workflow_key: str
    checkpoint_handle: str | None = None
    dataset_handles: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledAnalysisRequest:
    request: AnalysisRequest
    capability: AnalysisCapability
    checkpoint_handle: str | None
    dataset_handles: dict[str, str]
    parameters: dict[str, Any]
    warnings: tuple[dict[str, Any], ...] = ()


def compile_analysis_request(*, request: AnalysisRequest) -> CompiledAnalysisRequest:
    try:
        capability = resolve_analysis_capability(request.workflow_key)
    except ValueError as exc:
        raise AnalysisCompileValidationError(str(exc), field_path=("workflow_key",)) from exc

    if capability.requires_checkpoint and request.checkpoint_handle is None:
        raise AnalysisCompileValidationError(
            f"workflow '{capability.key}' requires checkpoint_handle",
            field_path=("checkpoint_handle",),
        )

    missing_dataset_keys = sorted(
        key for key in capability.dataset_input_keys if key not in request.dataset_handles
    )
    if missing_dataset_keys:
        raise AnalysisCompileValidationError(
            f"workflow '{capability.key}' requires dataset handles: {', '.join(missing_dataset_keys)}",
            field_path=("dataset_handles",),
        )

    return CompiledAnalysisRequest(
        request=request,
        capability=capability,
        checkpoint_handle=request.checkpoint_handle,
        dataset_handles=dict(request.dataset_handles),
        parameters=dict(request.parameters),
    )
