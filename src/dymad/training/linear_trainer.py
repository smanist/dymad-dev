import logging
# import numpy as np
import torch
from typing import Dict, Type, Union

from dymad.data import DynData, DynGeoData
from dymad.training.ls_update import LSUpdater
from dymad.training.trainer_base import TrainerBase

logger = logging.getLogger(__name__)

class LinearTrainer(TrainerBase):
    """
    Trainer using Linear approach.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize Linear trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        if self.config['training']['n_epochs'] > 1 or self.config['training']['save_interval'] > 1:
            logger.info("Linear training is typically one epoch, setting n_epochs=1 and save_interval=1.")
            self.config['training']['n_epochs'] = 1
            self.config['training']['save_interval'] = 1

        self._ls = LSUpdater(
            method=self.config['training'].get('method', 'full'),
            model=self.model,
            dt=self.metadata["dt_and_n_steps"][0][0],
            params=self.config['training'].get('params', None)
        )

        # Training weights
        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)

        # Additional logging
        logger.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: Union[DynData, DynGeoData]) -> torch.Tensor:
        """
        Process a batch and return predictions and ground truth states.

        Only used in `evaluation` in this Trainer.
        """
        B = batch.to(self.device)
        linear_loss = self._ls.eval_batch(self.model, B, self.criterion)

        if self.recon_weight > 0:
            # Add reconstruction loss
            _, _, x_hat = self.model(B)
            recon_loss = self.criterion(B.x, x_hat)
            return self.dynamics_weight * linear_loss + self.recon_weight * recon_loss
        else:
            return linear_loss

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        avg_epoch_loss = self._ls.update(self.model, self.train_loader)
        return avg_epoch_loss
