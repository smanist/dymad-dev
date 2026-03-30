"""Checkpoint compatibility adapter routed through facade/store/exec boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable

from dymad.exec.context import ExecutionContext, build_default_context
from dymad.exec.state import PredictionWorkflowPlan


@dataclass(frozen=True)
class BoundaryLoadTrace:
    plan: PredictionWorkflowPlan
    model_ref: str


def load_model_compat(
    model_class: type[Any],
    checkpoint_path: str | PathLike[str],
    *,
    context: ExecutionContext | None = None,
    horizon: int = 1,
    has_control: bool = False,
    has_graph: bool = False,
    return_trace: bool = False,
) -> tuple[Any, Callable[..., Any]] | tuple[Any, Callable[..., Any], BoundaryLoadTrace]:
    """Route compatibility model loading through the new boundary skeleton."""
    active_context = context or build_default_context()
    model_ref = f"{model_class.__module__}:{model_class.__name__}"
    plan = active_context.executor.plan_checkpoint_prediction(
        model_ref=model_ref,
        checkpoint_path=str(checkpoint_path),
        horizon=horizon,
        has_control=has_control,
        has_graph=has_graph,
    )
    model, predict_fn = active_context.executor.materialize_checkpoint_prediction(
        plan=plan,
        model_class=model_class,
    )
    if return_trace:
        return model, predict_fn, BoundaryLoadTrace(
            plan=plan,
            model_ref=model_ref,
        )
    return model, predict_fn
