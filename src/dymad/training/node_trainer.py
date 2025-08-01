import logging
import torch
from typing import Dict, Type

from dymad.data import DynData
from dymad.models import LDM
from dymad.training import TrainerBase

logger = logging.getLogger(__name__)

class SweepScheduler:
    """
    Scheduler to manage sweep lengths during training.
    Cycles through predefined sweep lengths.

    Args:
        sweep_lengths (list): List of sweep lengths to cycle through.
        epoch_step (int): Number of epochs after which to switch to the next sweep length.
    """


    def __init__(self, sweep_lengths: list, tolerances: list, epoch_step: int = 10):
        self.sweep_lengths = sweep_lengths
        self.epoch_step    = epoch_step
        self.tolerances    = tolerances
        self.current_epoch = 0
        self.sweep_epoch   = 0
        self.current_index = 0
        self.current_tol   = 0

        logging.info(f"Sweep lengths: {self.sweep_lengths}, Epoch step: {self.epoch_step}")

    def step(self, eploss: float = None) -> None:
        self.current_epoch += 1

        if self.tolerances is None:
            self._step_no_tolerance()
        else:
            self._step_with_tolerance(eploss)

    def _step_no_tolerance(self) -> None:
        """Handle stepping when no tolerances are provided."""
        index = self.current_epoch // self.epoch_step
        old_index = self.current_index
        self.current_index = min(index, len(self.sweep_lengths)-1)
        if old_index != self.current_index:
            logging.info(f"Switching to sweep length {self.sweep_lengths[self.current_index]} at epoch {self.current_epoch}")

    def _step_with_tolerance(self, eploss: float = None) -> None:
        self.sweep_epoch += 1
        current_tolerance = float(self.tolerances[self.current_tol])

        if self.sweep_epoch >= self.epoch_step or (eploss is not None and eploss < current_tolerance):
            self._advance_sweep(eploss, current_tolerance)

    def _advance_sweep(self, eploss: float, current_tolerance: float) -> None:
        self.sweep_epoch = 0
        self.current_index += 1
        if self.current_index >= len(self.sweep_lengths):
            self.current_index = 0
            if self.current_tol < len(self.tolerances)-1:
                self.current_tol += 1
                logging.info(f"Resetting to first sweep length after reaching end of list. Current tolerance {self.tolerances[self.current_tol]}")
            else:
                self.current_tol += 1
                logging.info("Reached Final Tolerance")
        logging.info(f"Switching to sweep length {self.sweep_lengths[self.current_index]} "
                     f"at epoch {self.current_epoch} with loss {eploss:.4e} < tolerance {current_tolerance:.4e}")

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
    Trainer using Neural ODE approach.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = LDM, config_mod: Dict = None):
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
        tolerances = self.config['training'].get('sweep_tolerances', None)
        self.convergence_tolerance = float(tolerances[-1]) if tolerances is not None else None
        self.schedulers.append(SweepScheduler(sweep_lengths, tolerances, epoch_step))

        # Additional logging
        logging.info(f"ODE method: {self.ode_method}, rtol: {self.rtol}, atol: {self.atol}")
        logging.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: DynData) -> torch.Tensor:
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
            scheduler.step((avg_epoch_loss) if type(scheduler).__name__ == "SweepScheduler" else None)

        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        if (self.schedulers[1].current_index == len(self.schedulers[1].sweep_lengths)-1
            and avg_epoch_loss < self.convergence_tolerance):
                self.convergence_tolerance_reached = True
        if self.schedulers[1].current_tol == len(self.schedulers[1].tolerances)-1:
            self.convergence_tolerance_reached = True
            
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
