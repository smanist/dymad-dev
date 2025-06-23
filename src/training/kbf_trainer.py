import torch
from src.models.kbf import KBF
from .trainer_base import TrainerBase
from src.losses.weak_form import weak_form_loss

class KBFTrainer(TrainerBase):
    """Trainer class for Koopman bilinear form (KBF) models."""
    
    def __init__(self, config_path: str):
        """Initialize KBF trainer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__(config_path, KBF)
        
        # Set training mode on the model for prediction method selection
        self.model.training_mode = 'weak_form'
        
        # Weak form loss weights from config
        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for trajectory in self.train_loader:
            ## NOTE: KBF type models currently do not support batch training, 
            # so we need to iterate over the trajectories (i.e. batch_size=1)
            batch = next(iter(trajectory)).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            # Extract states and controls
            states = batch[..., :self.metadata['n_total_state_features']]
            controls = batch[..., -self.metadata['n_control_features']:]
            # Forward pass
            predictions = self.model(states, controls)
            # Use weak form loss
            loss = weak_form_loss(states, predictions, self.metadata['weak_dyn_param'], self.criterion)
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
            for trajectory in dataloader:
                batch = next(iter(trajectory)).to(self.device)
                # Extract states and controls
                states = batch[..., :self.metadata['n_total_state_features']]
                controls = batch[..., -self.metadata['n_control_features']:]
                # Forward pass
                predictions = self.model(states, controls)
                # Use weak form loss
                loss = weak_form_loss(states, predictions, self.metadata['weak_dyn_param'], self.criterion)
                total_loss += loss.item()
                
        return total_loss / len(dataloader) 