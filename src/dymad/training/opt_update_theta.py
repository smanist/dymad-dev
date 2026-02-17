import copy
from dataclasses import replace
import logging
import torch
from typing import Any, Dict, Type

from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase

logger = logging.getLogger(__name__)


class OptUpdateTheta(OptBase):
    """
    Stub phase for theta update plumbing.

    This phase is a no-op unless `enable_latent` is true and latent payload exists.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_phase: Dict[str, Any],
        model_class: Type[torch.nn.Module],
        run_state: RunState,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = copy.deepcopy(config)
        self.config_phase = copy.deepcopy(config_phase)
        self.model_class = model_class
        self.run_state = run_state
        self.device = device
        self.dtype = dtype
        self.hist = copy.deepcopy(run_state.hist) if run_state.hist is not None else []

    def train(self) -> int:
        if not self.config_phase.get("enable_latent", False):
            logger.info("UpdateTheta disabled (enable_latent=False). Skipping.")
            return self.run_state.epoch

        if self.run_state.latent is None:
            logger.info("UpdateTheta skipped: no latent payload in RunState.")
            return self.run_state.epoch

        self.run_state.latent["theta_update"] = {
            "method": "stub",
            "objective": 0.0,
            "note": "no-op theta update",
        }
        logger.info("UpdateTheta wrote latent.theta_update placeholder.")

        if self.config_phase.get("do_optimizer_step", False):
            model = self.run_state.model
            optimizer = self.run_state.optimizer
            if model is None or optimizer is None:
                logger.info(
                    "UpdateTheta requested do_optimizer_step, but model or optimizer is missing. Skipping step."
                )
                return self.run_state.epoch

            optimizer.zero_grad(set_to_none=True)
            loss = None
            for param in model.parameters():
                term = 0.0 * torch.sum(param * param)
                loss = term if loss is None else loss + term

            if loss is None:
                logger.info("UpdateTheta optimizer step skipped: model has no parameters.")
            else:
                loss.backward()
                optimizer.step()
                logger.info("UpdateTheta performed one no-op optimizer step.")

        return self.run_state.epoch

    def export_run_state(self, epoch: int) -> RunState:
        return replace(
            self.run_state,
            epoch=epoch,
            hist=copy.deepcopy(self.hist),
            latent=copy.deepcopy(self.run_state.latent),
        )
