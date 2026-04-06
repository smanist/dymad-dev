from __future__ import annotations

import copy
from typing import Any

import torch

from dymad.training.execution_services import ExecutionServices
from dymad.training.phase_pipeline import PhasePipeline
from dymad.training.phase_runtime import (
    ArtifactRegistry,
    PhaseContext,
    PhaseResult,
    TrainerState,
    TrainingCheckpointError,
    build_initial_trainer_state,
)


class TrainerRun:
    """Owns one concrete training run identity, artifacts, and typed phase pipeline."""

    def __init__(
        self,
        config: dict[str, Any],
        model_class: type,
        device: torch.device,
        dtype: torch.dtype,
        run_name: str,
        checkpoint_prefix: str,
        results_prefix: str,
        execution_services: ExecutionServices | None = None,
    ):
        self.run_name = run_name
        self.execution_services = execution_services or ExecutionServices.from_config(
            config,
            default_device=device,
        )
        self.execution_services = self.execution_services.with_paths(
            checkpoint_prefix=checkpoint_prefix,
            results_prefix=results_prefix,
        )
        self.execution_services.ensure_artifact_dirs()
        self.checkpoint_prefix = self.execution_services.checkpoint_prefix
        self.results_prefix = self.execution_services.results_prefix

        self.config = self.execution_services.apply_to_config(copy.deepcopy(config))
        self.config.setdefault("model", {})
        self.config["model"]["name"] = run_name
        self.model_class = model_class
        self.device = self.execution_services.device
        self.dtype = dtype
        self.pipeline = PhasePipeline(
            config=self.config,
            model_class=self.model_class,
            device=self.device,
            dtype=self.dtype,
            execution_services=self.execution_services,
        )
        self.config = self.pipeline.config

    @property
    def run_checkpoint_path(self) -> str:
        return self.execution_services.checkpoint_file(f"{self.run_name}_run_checkpoint.pt")

    def save_run_checkpoint(self, trainer_state: TrainerState, artifacts: ArtifactRegistry) -> str:
        payload = {
            "schema": "dymad.training.run_checkpoint.v1",
            "trainer_state": trainer_state.checkpoint_payload(),
            "artifacts": artifacts.checkpoint_payload(),
        }
        torch.save(payload, self.run_checkpoint_path)
        return self.run_checkpoint_path

    def load_run_checkpoint(self, path: str | None = None) -> tuple[TrainerState, ArtifactRegistry]:
        checkpoint_path = self.run_checkpoint_path if path is None else path
        payload = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        if payload.get("schema") != "dymad.training.run_checkpoint.v1":
            raise TrainingCheckpointError(
                f"Unsupported training checkpoint schema in '{checkpoint_path}'. "
                "Legacy optimizer checkpoints are not resumable after the Phase 4 migration."
            )
        trainer_state = TrainerState.from_checkpoint_payload(
            payload["trainer_state"],
            execution_services=self.execution_services,
        )
        trainer_state.config = self.execution_services.apply_to_config(trainer_state.config)
        trainer_state.device = self.device
        artifacts = ArtifactRegistry.from_checkpoint_payload(payload.get("artifacts"))
        return trainer_state, artifacts

    def _maybe_resume(self) -> tuple[TrainerState | None, ArtifactRegistry | None]:
        if not self.pipeline.phase_specs:
            return None, None
        first_spec = self.pipeline.phase_specs[0]
        load_checkpoint = getattr(first_spec, "config", {}).get("load_checkpoint", False)
        if not load_checkpoint:
            return None, None
        path = load_checkpoint if isinstance(load_checkpoint, str) else None
        return self.load_run_checkpoint(path=path)

    def run(
        self,
        *,
        initial_context: PhaseContext,
        initial_state: TrainerState | None = None,
        artifacts: ArtifactRegistry | None = None,
    ) -> list[PhaseResult]:
        resumed_state, resumed_artifacts = self._maybe_resume()
        if resumed_state is not None:
            active_state = resumed_state
            active_artifacts = resumed_artifacts or ArtifactRegistry()
        else:
            active_state = initial_state or build_initial_trainer_state(
                self.config,
                execution_services=self.execution_services,
            )
            active_artifacts = ArtifactRegistry() if artifacts is None else artifacts

        return self.pipeline.run(
            initial_context=initial_context,
            initial_state=active_state,
            artifacts=active_artifacts,
            run_name=self.run_name,
            checkpoint_callback=self.save_run_checkpoint,
        )
