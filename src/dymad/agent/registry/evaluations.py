"""Evaluation capability registry."""

from __future__ import annotations

from functools import lru_cache

from dymad.agent.registry.types import DatasetKind, EvaluationCapability

SUPPORTED_EVALUATION_METRICS: tuple[str, ...] = ("rollout_rmse",)


@lru_cache(maxsize=1)
def _evaluation_capabilities() -> tuple[EvaluationCapability, ...]:
    return (
        EvaluationCapability(
            key="checkpoint_rollout",
            name="Checkpoint Rollout Evaluation",
            summary="Evaluate a persisted checkpoint against a registered dataset by rollout error.",
            dataset_kinds=("regular", "graph"),
            supported_metrics=SUPPORTED_EVALUATION_METRICS,
            parameter_schema={
                "metric": {
                    "type": "string",
                    "enum": list(SUPPORTED_EVALUATION_METRICS),
                    "default": "rollout_rmse",
                },
                "plot_selection": {
                    "type": "string",
                    "enum": ["best", "worst", "median"],
                    "default": "median",
                },
                "max_plots": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 0,
                },
                "predict_kwargs": {
                    "type": "object",
                    "default": {},
                },
            },
            notes=(
                "Graph datasets skip trajectory plotting in v1.",
                "Regular datasets write up to max_plots trajectory comparisons when plotting succeeds.",
            ),
        ),
    )


def list_evaluation_capabilities(
    *, dataset_kind: DatasetKind | None = None
) -> tuple[EvaluationCapability, ...]:
    capabilities = _evaluation_capabilities()
    if dataset_kind is None:
        return capabilities
    return tuple(cap for cap in capabilities if dataset_kind in cap.dataset_kinds)
