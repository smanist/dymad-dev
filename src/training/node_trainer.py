import logging
import torch
from typing import Tuple, Type

from .trainer_base import TrainerBase
from ...src.models.ldm import LDM

logger = logging.getLogger(__name__)

class NODETrainer(TrainerBase):
    """
    Trainer for Neural ODE models using direct ODE integration loss.
    Uses the unified LDM model.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = LDM):
        """Initialize NODE trainer with configuration."""
        super().__init__(config_path, model_class)

        # ODE solver settings from config
        self.ode_method = self.config['training'].get('ode_method', 'dopri5')
        self.rtol = self.config['training'].get('rtol', 1e-7)
        self.atol = self.config['training'].get('atol', 1e-9)

        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)

        # Additional logging
        logging.info(f"ODE method: {self.ode_method}, rtol: {self.rtol}, atol: {self.atol}")
        logging.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch and return predictions and ground truth states."""
        batch = batch.to(self.device)
        states = batch[:, :, :self.metadata['n_total_state_features']]
        controls = batch[:, :, -self.metadata['n_control_features']:]
        init_states = states[:, 0, :]  # (batch_size, n_total_state_features)

        # Use the actual time points from trajectory manager
        ts = self.t.to(self.device)
        # Use native batch prediction
        predictions = self.model.predict(init_states, controls, ts, method=self.ode_method)
        # predictions shape: (time_steps, batch_size, n_total_state_features)
        # We need: (batch_size, time_steps, n_total_state_features)
        predictions = predictions.permute(1, 0, 2)
        # Dynamics loss
        dynamics_loss = self.criterion(predictions, states)

        if self.recon_weight > 0:
            # Add reconstruction loss
            _, _, x_hat = self.model(states, controls)
            recon_loss = self.criterion(states, x_hat)
            return self.dynamics_weight * dynamics_loss + self.recon_weight * recon_loss
        else:
            return dynamics_loss

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._process_batch(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model on the provided dataloader."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                loss = self._process_batch(batch)
                total_loss += loss.item()

        return total_loss / len(dataloader)
