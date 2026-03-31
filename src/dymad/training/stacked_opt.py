import copy
import logging
import os
import torch
from typing import Dict, Any, Type

from dymad.training.phase_runtime import (
    compose_run_state,
    run_state_to_phase_context,
    run_state_to_trainer_state,
)
from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase
from dymad.training.opt_linear import OptLinear
from dymad.training.opt_node import OptNODE
from dymad.training.opt_weak_form import OptWeakForm
from dymad.utils import config_logger

OPT_REGISTRY: Dict[str, Type[OptBase]] = {
    "NODE": OptNODE,
    "Weak": OptWeakForm,
    "Linear": OptLinear,
}

class PhaseResult:
    def __init__(self, name: str, run_state: RunState, hist):
        self.name = name
        self.run_state = run_state
        self.hist = hist

class StackedOpt:
    """
    Stack multiple optimization phases (e.g., WF -> NODE -> LR)
    on a (potentially precomputed) RunState.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_class: Type,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = copy.deepcopy(config)
        self.model_class = model_class
        self.device = device
        self.dtype = dtype

        self.phases = copy.deepcopy(self.config.get("phases", []))
        if not self.phases:
            raise ValueError("Experiment config must contain a non-empty 'phases' list.")

    def run(self, initial_state: RunState) -> Dict[str, PhaseResult]:
        """
        Run all prescribed phases in order.

        initial_state:
          - None          => first trainer will build data/model itself (legacy/simple usage).
          - data-only     => first trainer reuses data, builds model.
          - full RunState => continuation: reuse data + model + optimizer, etc.
        """
        results = []
        phase_context = run_state_to_phase_context(initial_state)
        trainer_state = run_state_to_trainer_state(initial_state)

        log_config = self.config.get("log", {})
        ifstdout = log_config.get("stdout", False)
        self.logger = logging.getLogger('dymad')
        path = trainer_state.config['path']['results_prefix']
        os.makedirs(path, exist_ok=True)
        path += '/' + path.split('/')[-1]
        config_logger(
            self.logger,
            mode=log_config.get("level", "info"),
            prefix='' if ifstdout else path)

        for i, phase_cfg in enumerate(self.phases):
            phase_name  = phase_cfg.get("name", f"phase_{i}")
            trainer_key = phase_cfg["trainer"]
            trainer_cls = OPT_REGISTRY[trainer_key]

            self.logger.info(f"=== Starting phase '{phase_name}' with trainer '{trainer_key}' ===")

            # Temporary compatibility adapter while trainer classes still consume
            # legacy RunState directly.
            current_state = compose_run_state(trainer_state, phase_context)

            # Instantiate trainer; it will attach to provided RunState (data-only or full).
            trainer = trainer_cls(
                config=self.config,
                config_phase=phase_cfg,
                model_class=self.model_class,
                run_state=current_state,
                device=self.device,
                dtype=self.dtype,
            )

            # Run this phase
            epoch = trainer.train()

            # Export state for the next phase
            phase_state = trainer.export_run_state(epoch)
            trainer_state = run_state_to_trainer_state(phase_state)
            phase_context = run_state_to_phase_context(phase_state)
            current_state = compose_run_state(trainer_state, phase_context)
            results.append(PhaseResult(
                name=phase_name,
                run_state=current_state,
                hist=trainer.hist,
            ))

        self.logger.info("=== All phases completed ===")
        self.logger.removeHandler(self.logger.handlers[0])

        return results
