import torch, random
from src.models.ldm import LDM
from typing import Tuple
from .trainer_base import TrainerBase
from src.losses.evaluation import prediction_rmse
class NODETrainer(TrainerBase):
    """
    Trainer for Neural ODE models using direct ODE integration loss.
    Uses the unified LDM model.
    """
    
    def __init__(self, config_path: str):
        """Initialize NODE trainer with configuration."""
        super().__init__(config_path, LDM)
        
        # Set training mode on the model for prediction method selection
        self.model.training_mode = 'node'
        
        # ODE solver settings from config
        self.ode_method = self.config['training'].get('ode_method', 'dopri5')
        self.rtol = self.config['training'].get('rtol', 1e-7)
        self.atol = self.config['training'].get('atol', 1e-9)
    
    def _process_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch and return predictions and ground truth states."""
        batch = batch.to(self.device)
        states = batch[:, :, :self.metadata['n_state_features']]
        controls = batch[:, :, -self.metadata['n_control_features']:]
        init_states = states[:, 0, :]  # (batch_size, n_state_features)
        
        # Use the actual time points from trajectory manager
        ts = self.t.to(self.device)
        
        # Use native batch prediction 
        predictions = self.model.predict(init_states, controls, ts, method=self.ode_method)
        # predictions shape: (time_steps, batch_size, n_state_features)
        # We need: (batch_size, time_steps, n_state_features)
        predictions = predictions.permute(1, 0, 2)
        
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
    
    def evaluate_rmse(self, split: str = 'test', plot: bool = False) -> float:
        """Calculate RMSE on a random trajectory from the specified split."""
        dataset = getattr(self, f"{split}_set")
        trajectory = random.choice(dataset)
        return prediction_rmse(
            self.model, trajectory, self.t, 
            self.metadata, self.model_name, plot=plot
        )