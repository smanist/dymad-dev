import torch
from typing import Tuple

from .trainer_base import TrainerBase
from src.models.node import NODE

class NODETrainer(TrainerBase):
    """Trainer class for Neural ODE models."""
    
    def __init__(self, config_path: str):
        """Initialize NODE trainer with configuration."""
        super().__init__(config_path, NODE)
    
    def _process_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch and return predictions and ground truth states."""
        batch = batch.to(self.device)
        states = batch[:, :, :self.metadata['n_state_features']]
        controls = batch[:, :, -self.metadata['n_control_features']:]
        init_states = states[:, 0, :]
        # Use the actual time points from trajectory manager
        ts = self.t.to(self.device) # TODO: check trajectory of different lengths
        predictions = self.model.predict(init_states, controls[:, 0], ts)
        predictions = predictions.permute(1, 0, 2)  # (batch, time, features)
        return predictions, states
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            predictions, states = self._process_batch(batch)
            loss = self.criterion(predictions, states)
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
                predictions, states = self._process_batch(batch)
                loss = self.criterion(predictions, states)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)