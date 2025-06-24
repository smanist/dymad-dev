import torch
from typing import Type

from .trainer_base import TrainerBase
from ...src.models.lstm import LSTM

class LSTMTrainer(TrainerBase):
    """Trainer class for LSTM discrete-time models."""

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = LSTM):
        """Initialize LSTM trainer with configuration."""
        super().__init__(config_path, model_class)

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for feature, target in self.train_loader:
            feature = feature.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # Extract states and controls
            states = feature[:, :, :self.metadata['n_state_features']]
            controls = feature[:, :, -self.metadata['n_control_features']:]

            _, _, prediction = self.model(states, controls)

            # Compute loss against the final target state
            loss = self.criterion(prediction, target)

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
            for feature, target in dataloader:

                feature = feature.to(self.device)
                target = target.to(self.device)

                # Extract states and controls
                states = feature[:, :, :self.metadata['n_state_features']]
                controls = feature[:, :, -self.metadata['n_control_features']:]

                _, _, prediction = self.model(states, controls)

                # Compute loss against the final target state
                loss = self.criterion(prediction, target)
                total_loss += loss.item()

            return total_loss / len(dataloader)

    def get_prediction_rmse_func(self):
        """Return the LSTM-specific prediction RMSE function."""
        from ...src.losses.evaluation import prediction_rmse_lstm
        return prediction_rmse_lstm
