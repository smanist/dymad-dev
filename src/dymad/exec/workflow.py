"""Minimal exec workflow over facade operations."""

from __future__ import annotations

from dataclasses import dataclass

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
