"""Minimal exec workflow over facade operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dymad.exec.state import PredictionWorkflowPlan, SpectralWorkflowPlan
from dymad.facade.operations import FacadeOperations

if TYPE_CHECKING:
    from dymad.sako.adapter import SpectralAnalysisAdapter, SpectralEigensystem, SpectralRuntime
    from dymad.sako.snapshot import SpectralSnapshot


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
        raise NotImplementedError(
            "Checkpoint materialization is no longer routed through CompatibilityExecutor. "
            "Use dymad.io.load_model for now; executor-native materialization is pending."
        )

    def plan_spectral_analysis(
        self,
        *,
        model_ref: str,
        checkpoint_path: str,
        snapshot: SpectralSnapshot,
    ) -> SpectralWorkflowPlan:
        checkpoint = self.facade.register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
        )
        snapshot_summary = self.facade.register_spectral_snapshot(
            checkpoint_handle=checkpoint.handle,
            snapshot=snapshot,
        )
        return SpectralWorkflowPlan(
            checkpoint_handle=checkpoint.handle,
            spectral_snapshot_handle=snapshot_summary.handle,
            entrypoint="dymad.sako.SpectralAnalysis",
            notes=(
                "Spectral snapshot is persisted and resolved through facade/store handles.",
                "Numerical kernels still execute through the adapter compatibility layer.",
            ),
        )

    def materialize_spectral_adapter(
        self,
        *,
        plan: SpectralWorkflowPlan,
        eigensystem: SpectralEigensystem,
        runtime: SpectralRuntime | None = None,
        reps: float = 1e-10,
        etol: float = 1e-13,
    ) -> SpectralAnalysisAdapter:
        from dymad.sako.adapter import SpectralAnalysisAdapter

        snapshot_record = self.facade.get_spectral_snapshot(plan.spectral_snapshot_handle)
        if snapshot_record.checkpoint_handle != plan.checkpoint_handle:
            raise ValueError("plan checkpoint/spectral handles are inconsistent")
        return SpectralAnalysisAdapter(
            snapshot=snapshot_record.snapshot,
            eigensystem=eigensystem,
            runtime=runtime,
            reps=reps,
            etol=etol,
        )
