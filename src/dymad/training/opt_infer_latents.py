import copy
from dataclasses import replace
import logging
import torch
from typing import Any, Dict, Type

from dymad.training.helper import RunState
from dymad.training.latent_state import MapLatent
from dymad.training.opt_base import OptBase

logger = logging.getLogger(__name__)


class OptInferLatents(OptBase):
    """
    Stub phase for latent inference plumbing.

    This phase is a no-op unless `enable_latent` is set to true in the phase config.
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
            logger.info("InferLatents disabled (enable_latent=False). Skipping.")
            return self.run_state.epoch

        z_map = torch.empty(0, dtype=self.dtype, device=self.device)
        self.run_state.latent = MapLatent(
            z_map=z_map,
            diag={
                "method": "stub",
                "diagnostics": {
                    "note": "placeholder latent inference",
                },
            },
        )
        logger.info(
            "InferLatents wrote latent payload kind=%s with diag keys: %s",
            self.run_state.latent.kind,
            sorted(self.run_state.latent.diagnostic_info().keys()),
        )
        return self.run_state.epoch

    def export_run_state(self, epoch: int) -> RunState:
        return replace(
            self.run_state,
            epoch=epoch,
            hist=copy.deepcopy(self.hist),
            latent=copy.deepcopy(self.run_state.latent),
        )
