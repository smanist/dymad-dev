import copy
import logging
import os
import torch
from typing import Dict, Any, Type

from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase
from dymad.training.opt_infer_latents import OptInferLatents
from dymad.training.opt_linear import OptLinear
from dymad.training.opt_node import OptNODE
from dymad.training.opt_update_theta import OptUpdateTheta
from dymad.training.opt_weak_form import OptWeakForm
from dymad.utils.misc import config_logger

OPT_REGISTRY: Dict[str, Type[OptBase]] = {
    "InferLatents": OptInferLatents,
    "NODE": OptNODE,
    "UpdateTheta": OptUpdateTheta,
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
          - Required: a RunState instance.
          - Must contain a config dict with path.results_prefix.
          - data-only RunState is supported (first phase builds model/optimizer).
          - full RunState is supported (continuation across phases).
        """
        if initial_state is None:
            raise ValueError(
                "StackedOpt.run requires initial_state to be a RunState; got None."
            )
        if not isinstance(initial_state, RunState):
            raise ValueError(
                f"StackedOpt.run requires initial_state to be RunState, got {type(initial_state).__name__}."
            )
        if not isinstance(initial_state.config, dict):
            raise ValueError(
                "StackedOpt.run requires initial_state.config to be a dict containing path.results_prefix."
            )
        path_cfg = initial_state.config.get("path", None)
        if not isinstance(path_cfg, dict):
            raise ValueError(
                "StackedOpt.run requires initial_state.config['path'] to be a dict containing results_prefix."
            )
        results_prefix = path_cfg.get("results_prefix", None)
        if not results_prefix:
            raise ValueError(
                "StackedOpt.run requires initial_state.config['path']['results_prefix']."
            )

        results = []
        current_state = initial_state

        log_config = self.config.get("log", {})
        ifstdout = log_config.get("stdout", False)
        self.logger = logging.getLogger('dymad')
        path = results_prefix
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
            current_state = trainer.export_run_state(epoch)
            results.append(PhaseResult(
                name=phase_name,
                run_state=current_state,
                hist=trainer.hist,
            ))

        self.logger.info("=== All phases completed ===")
        self.logger.removeHandler(self.logger.handlers[0])

        return results
