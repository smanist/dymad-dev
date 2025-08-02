import logging
import torch
from typing import Dict, Type, Union

from dymad.data import DynData, DynGeoData
from dymad.training.trainer_base import TrainerBase
from dymad.utils.scheduler import make_scheduler

logger = logging.getLogger(__name__)

class NODETrainer(TrainerBase):
    """
    Trainer using Neural ODE approach.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize NODE trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        # ODE solver settings from config
        self.ode_method = self.config['training'].get('ode_method', 'dopri5')
        self.rtol = self.config['training'].get('rtol', 1e-7)
        self.atol = self.config['training'].get('atol', 1e-9)

        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)

        sweep_lengths = self.config['training'].get('sweep_lengths', [len(self.t)])
        epoch_step = self.config['training'].get('sweep_epoch_step', self.config['training']['n_epochs'])
        sweep_tols = self.config['training'].get('sweep_tols', None)
        sweep_mode = self.config['training'].get('sweep_mode', 'skip')
        self.schedulers.append(make_scheduler(
            scheduler_type="sweep", sweep_lengths=sweep_lengths, sweep_tols=sweep_tols, \
            epoch_step=epoch_step, mode=sweep_mode))

        # Additional logging
        logger.info(f"Added scheduler: {self.schedulers[-1].diagnostic_info()}")
        logger.info(f"ODE method: {self.ode_method}, rtol: {self.rtol}, atol: {self.atol}")
        logger.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: Union[DynData, DynGeoData]) -> torch.Tensor:
        """Process a batch and return predictions and ground truth states."""
        num_steps = self.schedulers[1].get_length()

        B = batch.truncate(num_steps)  # Truncate batch to the current sweep length
        B = B.to(self.device)
        init_states = B.x[:, 0, :]  # (batch_size, n_total_state_features)

        # Use the actual time points from trajectory manager
        ts = self.t[:num_steps].to(self.device)
        # Use native batch prediction
        predictions = self.model.predict(init_states, B, ts, method=self.ode_method)
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

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr     = 5e-5

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._process_batch(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_epoch_loss = total_loss / len(self.train_loader)

        for scheduler in self.schedulers:
            flag = scheduler.step(eploss=avg_epoch_loss)
            self.convergence_tolerance_reached = self.convergence_tolerance_reached or flag

        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        return avg_epoch_loss

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model on the provided dataloader."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                loss = self._process_batch(batch)
                total_loss += loss.item()

        return total_loss / len(dataloader)
