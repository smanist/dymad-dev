"""Minimal exec workflow over facade operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from dymad.exec.state import PredictionWorkflowPlan
from dymad.facade.operations import FacadeOperations


@dataclass
class CompatibilityExecutor:
    """Plans a typed-handle flow for checkpoint-compatible prediction."""

    facade: FacadeOperations

    def plan_checkpoint_prediction(
        self,
        *,
        model_ref: str,
        checkpoint_path: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> PredictionWorkflowPlan:
        checkpoint = self.facade.register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
        )
        request = self.facade.prepare_prediction_request(
            checkpoint_handle=checkpoint.handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )
        return PredictionWorkflowPlan(
            checkpoint_handle=checkpoint.handle,
            prediction_handle=request.handle,
            entrypoint="dymad.io.checkpoint.load_model",
            notes=(
                "This skeleton intentionally records boundary state only.",
                "Numerical model behavior remains in legacy io/models modules.",
            ),
        )

    def materialize_checkpoint_prediction(
        self,
        *,
        plan: PredictionWorkflowPlan,
        model_class: type[Any],
    ) -> tuple[Any, Callable[..., Any]]:
        request = self.facade.get_prediction_request(plan.prediction_handle)
        if request.checkpoint_handle != plan.checkpoint_handle:
            raise ValueError("plan checkpoint/prediction handles are inconsistent")
        checkpoint = self.facade.get_checkpoint(request.checkpoint_handle)

        from dymad.io.checkpoint import load_model as legacy_load_model

        return legacy_load_model(model_class, checkpoint.checkpoint_path)
