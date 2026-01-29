import logging
import torch
from typing import Any, Dict, Type

from dymad.io import DynData
from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase

logger = logging.getLogger(__name__)

class OptLinear(OptBase):
    """
    Optimization using Linear approach.
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

        if self.config_phase['n_epochs'] > 1 or self.config_phase['save_interval'] > 1:
            logger.info("Linear training is typically one epoch, setting n_epochs=1 and save_interval=1.")
            self.config_phase['n_epochs'] = 1
            self.config_phase['save_interval'] = 1

        self._ls_update_times = 0
        self._start_w_ls = False

        # Additional logging
        logger.info(f"LinearTrainer: method {self._ls.method}, params {self._ls.params}")

    def _process_batch(self, batch: DynData) -> torch.Tensor:
        """
        Process a batch and return predictions and ground truth states.

        Only used in `evaluation` in this Trainer.
        """
        B = batch.to(self.device)
        linear_loss = self._ls.eval_batch(self.model, B, self.criteria[0])
        loss_list = [linear_loss]

        # Other criteria
        # x_hat and predictions are computed inside criteria evaluation if needed
        x_hat, preds = None, None
        _list = self._additional_criteria_evaluation(x_hat, preds, B)
        loss_list.extend(_list)

        return loss_list

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        logger.info("Least squares update in OptLinear.")
        avg_epoch_loss, _ = self._ls.update(self.model, self.train_loader)
        loss_items = [avg_epoch_loss] + [0.0] * (len(self.criteria_weights) - 1)
        return torch.tensor(avg_epoch_loss, dtype=torch.float64), loss_items
