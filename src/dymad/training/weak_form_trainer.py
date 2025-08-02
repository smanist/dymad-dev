import logging
import torch
from typing import Dict, Type

from dymad.training.trainer_base import TrainerBase
from dymad.losses.weak_form import generate_weak_form_params, weak_form_loss_batch

logger = logging.getLogger(__name__)

class WeakFormTrainer(TrainerBase):
    """
    Trainer using weak form loss.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize weak form trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        # Weak form loss weights from config
        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)
        dtype = next(self.model.parameters()).dtype
        self.weak_dyn_param = generate_weak_form_params(self.metadata, dtype, self.device)

        # Additional logging
        logging.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")
        logging.info(f"Weak-form weights generated")

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            # Forward pass
            predictions = self.model(batch)
            # Use weak form loss with weights
            loss = weak_form_loss_batch(
                batch.x, predictions,
                self.weak_dyn_param,
                self.criterion,
                reconstruction_weight=self.recon_weight,
                dynamics_weight=self.dynamics_weight
            )
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
                batch = batch.to(self.device)
                predictions = self.model(batch)

                # Use weak form loss with weights
                loss = weak_form_loss_batch(
                    batch.x, predictions,
                    self.weak_dyn_param,
                    self.criterion,
                    reconstruction_weight=self.recon_weight,
                    dynamics_weight=self.dynamics_weight
                )
                total_loss += loss.item()

        return total_loss / len(dataloader)
