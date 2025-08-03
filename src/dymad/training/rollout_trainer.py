import logging
import torch
from typing import Dict, Type, Union

from dymad.data import DynData, DynGeoData
from dymad.training.node_trainer import NODETrainer
from dymad.utils.scheduler import make_scheduler

logger = logging.getLogger(__name__)

def _determine_chop_step(window: int, step: Union[int, float]) -> int:
    """
    Determine the chop step based on the window size and step value.
    """
    if isinstance(step, int):
        return step
    elif isinstance(step, float):
        stp = int(window * step)
        return min(max(stp, 1), window)
    else:
        raise ValueError(f"Invalid step type: {type(step)}. Expected int or float.")

class RollOutTrainer(NODETrainer):
    """
    Trainer for discrete-time models.

    Implemented as a simplified version of NODETrainer that does not use ODE solvers.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize NODE trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)

        self.chop_mode = self.config['training'].get('chop_mode', 'initial')
        assert self.chop_mode in ['initial', 'unfold'], f"Invalid chop_mode: {self.chop_mode}"

        self.chop_step = self.config['training'].get('chop_step', 1.0)
        assert self.chop_step > 0, f"Chop step must be positive. Got: {self.chop_step}"

        sweep_lengths = self.config['training'].get('sweep_lengths', [len(self.t)])
        epoch_step = self.config['training'].get('sweep_epoch_step', self.config['training']['n_epochs'])
        sweep_tols = self.config['training'].get('sweep_tols', None)
        sweep_mode = self.config['training'].get('sweep_mode', 'skip')
        self.schedulers.append(make_scheduler(
            scheduler_type="sweep", sweep_lengths=sweep_lengths, sweep_tols=sweep_tols, \
            epoch_step=epoch_step, mode=sweep_mode))

        # Additional logging
        if self.chop_mode == 'initial':
            logger.info(f"Chop mode: {self.chop_mode}, initial steps only)")
        else:
            logger.info(f"Chop mode: {self.chop_mode}, window stride: {self.chop_step}")
        logger.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: Union[DynData, DynGeoData]) -> torch.Tensor:
        """Process a batch and return predictions and ground truth states."""
        num_steps = self.schedulers[1].get_length()

        if self.chop_mode == 'initial':
            B = batch.truncate(num_steps)
        else:
            B = batch.unfold(num_steps, _determine_chop_step(num_steps, self.chop_step))

        B = B.to(self.device)
        init_states = B.x[:, 0, :]  # (batch_size, n_total_state_features)

        # Use the actual time points from trajectory manager
        ts = self.t[:num_steps].to(self.device)
        # Use native batch prediction
        predictions = self.model.predict(init_states, B, ts)
        # predictions shape: (time_steps, batch_size, n_total_state_features)
        # We need: (batch_size, time_steps, n_total_state_features)
        predictions = predictions.permute(1, 0, 2)
        # Dynamics loss
        dynamics_loss = self.criterion(predictions, B.x)

        if self.recon_weight > 0:
            # Add reconstruction loss
            _, _, x_hat = self.model(B)
            recon_loss = self.criterion(B.x, x_hat)
            return self.dynamics_weight * dynamics_loss + self.recon_weight * recon_loss
        else:
            return dynamics_loss
