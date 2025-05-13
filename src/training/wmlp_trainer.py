import torch
from typing import Tuple

from .trainer_base import TrainerBase
from src.models.wMLP import weakFormMLP
from src.losses.weak_form import weak_form_loss

class WMLPTrainer(TrainerBase):
    """Trainer class for weak form MLP models."""
    
    def __init__(self, config_path: str):
        """Initialize wMLP trainer with configuration."""
        super().__init__(config_path, weakFormMLP)
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            # Extract states and controls
            states = batch[:, :, :self.metadata['n_state_features']]
            controls = batch[:, :, -self.metadata['n_control_features']:]
            # Forward pass - specific to wMLP
            predictions = self.model(states, controls)
            # Use weak form loss
            loss = weak_form_loss(batch, predictions, self.metadata, self.criterion)
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
                batch = batch.to(self.device)
                # Extract states and controls
                states = batch[:, :, :self.metadata['n_state_features']]
                controls = batch[:, :, -self.metadata['n_control_features']:]
                # Forward pass
                predictions = self.model(states, controls)
                # Use weak form loss
                loss = weak_form_loss(batch, predictions, self.metadata, self.criterion)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)