from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch

from dymad.training.execution_services import ExecutionServices
from dymad.training.phase_runtime import ArtifactRegistry, PhaseContext, PhaseResult, TrainerState
from dymad.training.phases import PhaseSpec, build_phase, normalize_phase_specs


class PhasePipeline:
    """Runs typed training phases in sequence."""

    def __init__(
        self,
        config: dict[str, Any],
        model_class: type,
        device: torch.device,
        dtype: torch.dtype,
        execution_services: ExecutionServices | None = None,
    ):
        self.config = copy.deepcopy(config)
        self.model_class = model_class
        self.execution_services = execution_services or ExecutionServices.from_config(
            self.config,
            default_device=device,
        )
        self.config = self.execution_services.apply_to_config(self.config)
        self.device = self.execution_services.device
        self.dtype = dtype
        self.phase_specs: list[PhaseSpec] = normalize_phase_specs(self.config)
        self.phases = self.phase_specs
        if not self.phase_specs:
            raise ValueError("Experiment config must define at least one phase.")

    def run(
        self,
        *,
        initial_context: PhaseContext,
        initial_state: TrainerState,
        artifacts: ArtifactRegistry | None = None,
        run_name: str,
        checkpoint_callback: Callable[[TrainerState, ArtifactRegistry], None] | None = None,
    ) -> list[PhaseResult]:
        artifacts = ArtifactRegistry() if artifacts is None else artifacts
        results: list[PhaseResult] = []
        active_state = initial_state
        active_context = initial_context

        self.execution_services.ensure_artifact_dirs()
        logger = self.execution_services.configure_logger(
            "dymad",
            prefix=self.execution_services.logger_prefix(run_name),
        )

        for phase_index in range(active_state.phase_cursor, len(self.phase_specs)):
            spec = self.phase_specs[phase_index]
            logger.info("=== Starting phase '%s' (%s) ===", spec.name, spec.kind)
            phase = build_phase(
                spec,
                config=self.config,
                model_class=self.model_class,
                dtype=self.dtype,
                execution_services=self.execution_services,
            )
            result = phase.execute(
                trainer_state=active_state,
                phase_context=active_context,
                artifacts=artifacts,
                run_name=run_name,
                logger=logger,
            )
            active_state = result.trainer_state
            active_context = result.phase_context
            artifacts = result.artifacts
            active_state.phase_cursor = phase_index + 1
            results.append(result)
            if checkpoint_callback is not None:
                checkpoint_callback(active_state, artifacts)

        logger.info("=== All phases completed ===")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        return results
