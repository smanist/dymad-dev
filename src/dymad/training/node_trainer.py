import logging
import torch
from typing import Tuple, Type

from .trainer_base import TrainerBase
from ...src.models.ldm import LDM

logger = logging.getLogger(__name__)

class SweepScheduler:
    """
    Scheduler to manage sweep lengths during training.
    Cycles through predefined sweep lengths.
    """

    def __init__(self, sweep_lengths: list, epoch_step: int = 10):
        self.sweep_lengths = sweep_lengths
        self.epoch_step    = epoch_step
        self.current_epoch = 0
        self.current_index = 0

        logging.info(f"Sweep lengths: {self.sweep_lengths}, Epoch step: {self.epoch_step}")

    def step(self) -> None:
        """Advance to the next sweep length."""
        self.current_epoch += 1
        index = self.current_epoch // self.epoch_step
        old_index = self.current_index
        self.current_index = min(index, len(self.sweep_lengths)-1)

        if old_index != self.current_index:
            logging.info(f"Switching to sweep length {self.sweep_lengths[self.current_index]} at epoch {self.current_epoch}")

    def get_length(self) -> int:
        return self.sweep_lengths[self.current_index]
    
    def state_dict(self) -> dict:
        """Return the state dictionary for saving."""
        return {
            'sweep_lengths': self.sweep_lengths,
            'epoch_step':    self.epoch_step,
            'current_epoch': self.current_epoch,
            'current_index': self.current_index
        }

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

        sweep_lengths = self.config['training'].get('sweep_lengths', [len(self.t)])
        epoch_step = self.config['training'].get('sweep_epoch_step', self.config['training']['n_epochs'])
        self.schedulers.append(SweepScheduler(sweep_lengths, epoch_step))

        # Additional logging
        logging.info(f"ODE method: {self.ode_method}, rtol: {self.rtol}, atol: {self.atol}")
        logging.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch and return predictions and ground truth states."""
        num_steps = self.schedulers[1].get_length()

        batch = batch.to(self.device)
        states = batch[:, :num_steps, :self.metadata['n_total_state_features']]
        controls = batch[:, :num_steps, -self.metadata['n_control_features']:]
        init_states = states[:, 0, :]  # (batch_size, n_total_state_features)

        # Use the actual time points from trajectory manager
        ts = self.t[:num_steps].to(self.device)
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

        for scheduler in self.schedulers:
            scheduler.step()
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
