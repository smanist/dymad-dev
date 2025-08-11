import logging
import numpy as np
import torch
from typing import Dict, Type, Union

from dymad.data import DynData, DynGeoData
from dymad.numerics.linalg import truncated_lstsq
from dymad.training.trainer_base import TrainerBase

logger = logging.getLogger(__name__)

class LinearTrainer(TrainerBase):
    """
    Trainer using Linear approach.
    """

    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize Linear trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        if self.config['training']['n_epochs'] > 1 or self.config['training']['save_interval'] > 1:
            logger.info("Linear training is typically one epoch, setting n_epochs=1 and save_interval=1.")
            self.config['training']['n_epochs'] = 1
            self.config['training']['save_interval'] = 1

        self.method = self.config['training'].get('method', 'full')
        self.params = self.config['training'].get('params', None)
        if self.method not in ['full', 'truncated']:
            raise ValueError(f"Unsupported method: {self.method}. Supported methods are 'full' and 'truncated'.")

        if model_class.CONT:
            logger.info("Using continuous-time model for linear training.")
            self._comp_linear_features = self._comp_linear_features_ct
            self._comp_linear_targets  = self._comp_linear_targets_ct
            self._comp_linear_predict  = self._comp_linear_predict_ct
        else:
            logger.info("Using discrete-time model for linear training.")
            self._comp_linear_features = self._comp_linear_features_dt
            self._comp_linear_targets  = self._comp_linear_targets_dt
            self._comp_linear_predict  = self._comp_linear_predict_dt

        # Training weights
        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)

        # Additional logging
        logger.info(f"Using method: {self.method} with params: {self.params}")
        logger.info(f"Weights: Dynamics {self.dynamics_weight}, Reconstruction {self.recon_weight}")

    def _process_batch(self, batch: Union[DynData, DynGeoData]) -> torch.Tensor:
        """
        Process a batch and return predictions and ground truth states.
        
        Only used in `evaluation` in this Trainer.
        """
        B  = batch.to(self.device)
        z, z_dot, x_hat = self.model(B)
        _b = self._comp_linear_targets(z)
        _p = self._comp_linear_predict(z_dot)
        linear_loss = self.criterion(_p, _b)

        if self.recon_weight > 0:
            # Add reconstruction loss
            recon_loss = self.criterion(B.x, x_hat)
            return self.dynamics_weight * linear_loss + self.recon_weight * recon_loss
        else:
            return linear_loss

    def _comp_linear_features_dt(self, batch: DynData | DynGeoData) -> torch.Tensor:
        """Compute linear features for discrete-time models."""
        f = self.model.linear_features(batch)
        return f[..., :-1, :]

    def _comp_linear_targets_dt(self, z: torch.Tensor) -> torch.Tensor:
        """Compute linear targets for discrete-time models."""
        return z[..., 1:, :]

    def _comp_linear_predict_dt(self, z_dot: torch.Tensor) -> torch.Tensor:
        """Compute predicted targets for discrete-time models.
        z_dot really means z_next here.
        """
        return z_dot[..., :-1, :]

    def _comp_linear_features_ct(self, batch: DynData | DynGeoData) -> torch.Tensor:
        """Compute linear features for continuous-time models."""
        f = self.model.linear_features(batch)
        return f

    def _comp_linear_targets_ct(self, z: torch.Tensor) -> torch.Tensor:
        """Compute linear targets for continuous-time models."""
        raise NotImplementedError("Continuous-time linear targets not implemented.")

    def _comp_linear_predict_ct(self, z_dot: torch.Tensor) -> torch.Tensor:
        """Compute predicted targets for continuous-time models."""
        return z_dot

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()

        with torch.no_grad():
            # Assemble the linear system
            A, b = [], []
            for batch in self.train_loader:
                z, _, _ = self.model(batch)
                _A = self._comp_linear_features(batch)
                _b = self._comp_linear_targets(z)
                A.append(_A)
                b.append(_b)
            A = torch.cat(A, dim=0).cpu().numpy()
            b = torch.cat(b, dim=0).cpu().numpy()
            A = A.reshape(-1, A.shape[-1])
            b = b.reshape(-1, b.shape[-1])

            # Solve the linear system
            if self.method == 'full':
                W = np.linalg.lstsq(A, b, rcond=None)[0]
                Wt = torch.tensor(W, dtype=batch.x.dtype, device=self.device)
                self.model.set_linear_weights(Wt)
                avg_epoch_loss = np.linalg.norm(A @ W - b) / A.shape[0]

            elif self.method == 'truncated':
                _V, _U = truncated_lstsq(A, b, tsvd=8)
                self.model.set_linear_weights(
                    U=torch.tensor(_U, dtype=batch.x.dtype, device=self.device),
                    V=torch.tensor(_V, dtype=batch.x.dtype, device=self.device))
                avg_epoch_loss = np.linalg.norm((A @ _V) @ _U.T - b) / A.shape[0]

        return avg_epoch_loss
