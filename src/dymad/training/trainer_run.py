from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Type

import torch

from dymad.training.execution_services import ExecutionServices
from dymad.training.helper import RunState
from dymad.training.phase_pipeline import PhasePipeline, PhaseResult


class TrainerRun:
    """Owns one concrete training run identity, artifacts, and phase pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_class: Type,
        device: torch.device,
        dtype: torch.dtype,
        run_name: str,
        checkpoint_prefix: str,
        results_prefix: str,
        execution_services: Optional[ExecutionServices] = None,
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
        self.checkpoint_prefix = self.execution_services.checkpoint_prefix
        self.results_prefix = self.execution_services.results_prefix
        self.execution_services.ensure_artifact_dirs()

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

    def run(self, initial_state: RunState) -> List[PhaseResult]:
        """Execute this run's phase pipeline and return ordered phase results."""

        return self.pipeline.run(initial_state=initial_state)
