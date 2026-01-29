import logging
import torch
from typing import Any, Dict, Type

from dymad.io import DynData
from dymad.numerics import generate_weak_weights
from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase

logger = logging.getLogger(__name__)

class OptWeakForm(OptBase):
    """
    Optimization using weak form loss.
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
        super().__init__(config, config_phase, model_class, run_state, device, dtype)

        # Weak form parameters
        self.N      = self.config_phase["weak_form_params"]["N"]
        self.dN     = self.config_phase["weak_form_params"]["dN"]
        self.ordpol = self.config_phase["weak_form_params"]["ordpol"]
        self.ordint = self.config_phase["weak_form_params"]["ordint"]
        self._gen_params()

        # Additional logging
        logger.info(f"Weak-form weights generated")

    def _gen_params(self):
        """Generate weak form parameters."""
        dtype = next(self.model.parameters()).dtype
        C, D = generate_weak_weights(
            dt                   = self.train_md["dt_and_n_steps"][0][0],
            n_integration_points = self.N,
            poly_order           = self.ordpol,
            int_rule_order       = self.ordint,
        )

        # store as ordpol x N
        self.C = torch.tensor(C.T, dtype=dtype, device=self.device)
        self.D = torch.tensor(D.T, dtype=dtype, device=self.device)

    def _process_batch(self, batch: DynData) -> torch.Tensor:
        B = batch.to(self.device)
        z = self.model.encoder(B)
        z_dot = self.model.dynamics(z, B)
        x_hat = self.model.decoder(z, B)

        z_windows = z.unfold(1, self.N, self.dN)
        z_dot_windows = z_dot.unfold(1, self.N, self.dN)

        true_weak = z_windows @ self.C
        pred_weak = z_dot_windows @ self.D
        weak_loss = self.criteria[0](pred_weak, true_weak)
        loss_list = [weak_loss]

        # Other criteria
        _list = self._additional_criteria_evaluation(x_hat, None, B)
        loss_list.extend(_list)

        return loss_list
