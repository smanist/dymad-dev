import logging
import torch
from typing import Dict, Type, Union

from dymad.data import DynData, DynGeoData
from dymad.training.trainer_base import TrainerBase
from dymad.utils.weak import generate_weak_weights

logger = logging.getLogger(__name__)

class WeakFormTrainer(TrainerBase):
    """
    Trainer using weak form loss.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize weak form trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        # Weak form parameters
        self.N      = self.config["training"]["weak_form_params"]["N"]
        self.dN     = self.config["training"]["weak_form_params"]["dN"]
        self.ordpol = self.config["training"]["weak_form_params"]["ordpol"]
        self.ordint = self.config["training"]["weak_form_params"]["ordint"]
        self._gen_params()

        # Training weights
        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)

        # Additional logging
        logger.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")
        logger.info(f"Weak-form weights generated")

    def _gen_params(self):
        """Generate weak form parameters."""
        dtype = next(self.model.parameters()).dtype
        C, D = generate_weak_weights(
            dt                   = self.metadata["dt_and_n_steps"][0][0],
            n_integration_points = self.N,
            poly_order           = self.ordpol,
            int_rule_order       = self.ordint,
        )

        # store as ordpol x N
        self.C = torch.tensor(C.T, dtype=dtype, device=self.device)
        self.D = torch.tensor(D.T, dtype=dtype, device=self.device)

    def _process_batch(self, batch: Union[DynData, DynGeoData]) -> torch.Tensor:
        B = batch.to(self.device)
        z, z_dot, x_hat = self.model(B)

        z_windows = z.unfold(1, self.N, self.dN)
        z_dot_windows = z_dot.unfold(1, self.N, self.dN)

        true_weak = z_windows @ self.C
        pred_weak = z_dot_windows @ self.D
        weak_loss = self.criterion(pred_weak, true_weak)

        if self.recon_weight > 0:
            # Add reconstruction loss
            recon_loss = self.criterion(B.x, x_hat)
            return self.dynamics_weight * weak_loss + self.recon_weight * recon_loss
        else:
            return weak_loss

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
