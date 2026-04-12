"""Analysis capability registry."""

from __future__ import annotations

from functools import lru_cache

from dymad.agent.registry.types import AnalysisCapability


@lru_cache(maxsize=1)
def _analysis_capabilities() -> tuple[AnalysisCapability, ...]:
    return (
        AnalysisCapability(
            key="spectral_koopman",
            name="Spectral Koopman Analysis",
            summary="Checkpoint-backed spectral Koopman analysis via the adapter compatibility layer.",
            support_level="supported",
            implementation="library",
            requires_checkpoint=True,
            dataset_input_keys=(),
            parameter_schema={
                "dt": {"type": "number", "default": 1.0},
                "forder": {"type": "string", "default": "full"},
                "reps": {"type": "number", "default": 1e-10},
                "etol": {"type": "number", "default": 1e-13},
                "remove_one": {"type": "boolean", "default": True},
            },
        ),
        AnalysisCapability(
            key="vortex_transform_modes",
            name="Vortex Transform/Mode Analysis",
            summary="Transform-backed forward/backward mode analysis for the vortex workflow.",
            support_level="supported",
            implementation="library",
            requires_checkpoint=False,
            dataset_input_keys=("train_dataset_handle", "test_dataset_handle"),
            parameter_schema={
                "config_path": {"type": "string"},
                "index": {"type": "integer", "default": 5},
                "nx": {"type": "integer", "default": 199},
                "ny": {"type": "integer", "default": 449},
            },
        ),
    )


def list_analysis_capabilities() -> tuple[AnalysisCapability, ...]:
    return _analysis_capabilities()


def resolve_analysis_capability(key: str) -> AnalysisCapability:
    normalized = key.strip().lower()
    if not normalized:
        raise ValueError("analysis capability key cannot be empty")
    for capability in _analysis_capabilities():
        if capability.key == normalized:
            return capability
    supported = ", ".join(capability.key for capability in _analysis_capabilities())
    raise ValueError(f"unknown analysis capability '{key}'. supported capabilities: {supported}")
