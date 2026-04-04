from __future__ import annotations

import copy
from typing import Any, Dict, List, Type

import torch

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
    ):
        self.run_name = run_name
        self.checkpoint_prefix = checkpoint_prefix
        self.results_prefix = results_prefix

        self.config = copy.deepcopy(config)
        self.config.setdefault("model", {})
        self.config["model"]["name"] = run_name
        self.config.setdefault("path", {})
        self.config["path"]["checkpoint_prefix"] = checkpoint_prefix
        self.config["path"]["results_prefix"] = results_prefix

        self.model_class = model_class
        self.device = device
        self.dtype = dtype
        self.pipeline = PhasePipeline(
            config=self.config,
            model_class=self.model_class,
            device=self.device,
            dtype=self.dtype,
        )
        self.config = self.pipeline.config

    def run(self, initial_state: RunState) -> List[PhaseResult]:
        """Execute this run's phase pipeline and return ordered phase results."""

        return self.pipeline.run(initial_state=initial_state)
