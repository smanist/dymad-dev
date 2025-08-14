import inspect
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union

from dymad.data import DynData, DynGeoData
from dymad.numerics.linalg import truncated_lstsq

logger = logging.getLogger(__name__)

def _dt_target(z: torch.Tensor) -> torch.Tensor:
    """Compute discrete-time targets."""
    return z[..., 1:, :]

def _comp_linear_features_dt(model, batch: DynData | DynGeoData, **kwargs) -> torch.Tensor:
    """Compute linear features for discrete-time models."""
    A, z = model.linear_features(batch)
    _A = A[..., :-1, :]
    _z = _dt_target(z)
    return _A.reshape(-1, _A.shape[-1]), _z.reshape(-1, _z.shape[-1])

def _comp_linear_eval_dt(model, batch: DynData | DynGeoData, **kwargs) -> torch.Tensor:
    """Compute predicted targets for discrete-time models.
    z_dot really means z_next here.
    """
    z_dot, z = model.linear_eval(batch)
    return z_dot[..., :-1, :], _dt_target(z)

def _ct_target(z: torch.Tensor, dt) -> torch.Tensor:
    """Compute linear targets for continuous-time models."""
    dz = np.gradient(z.cpu().numpy(), dt, axis=-2, edge_order=2)
    return torch.tensor(dz, dtype=z.dtype, device=z.device)

def _comp_linear_features_ct(model, batch: DynData | DynGeoData, **kwargs) -> torch.Tensor:
    """Compute linear features for continuous-time models."""
    A, z = model.linear_features(batch)
    _z = _ct_target(z, kwargs['dt'])
    return A.reshape(-1, A.shape[-1]), _z.reshape(-1, _z.shape[-1])

def _comp_linear_eval_ct(model, batch: DynData | DynGeoData, **kwargs) -> torch.Tensor:
    """Compute predicted targets for continuous-time models."""
    z_dot, z = model.linear_eval(batch)
    return z_dot, _ct_target(z, kwargs['dt'])

def check_linear_impl(model) -> bool:
    """
    Check if the model implements linear features and eval methods.

    Technically we should check linear_eval and set_linear_weights as well.
    """
    has_linear_features = hasattr(model, 'linear_features')
    if not has_linear_features:
        return False

    source = inspect.getsource(model.linear_features)
    if "raise NotImplementedError" in source:
        return False

    return True

class LSUpdater:
    """
    Update linear weights by least squares.
    """

    def __init__(self, method, model, dt=None, params=None):
        self.method = method
        self.params = params
        self.dt     = dt

        if not check_linear_impl(model):
            raise ValueError("Model does not implement linear_features and linear_eval methods required for LS updates.")

        if self.method not in ['full', 'truncated']:
            raise ValueError(f"Unsupported method: {self.method}. Supported methods are 'full' and 'truncated'.")

        if model.CONT:
            self._comp_linear_features = _comp_linear_features_ct
            self._comp_linear_eval     = _comp_linear_eval_ct
            logger.info(f"Using continuous-time model for linear updates, dt={self.dt}.")
        else:
            self._comp_linear_features = _comp_linear_features_dt
            self._comp_linear_eval     = _comp_linear_eval_dt
            logger.info("Using discrete-time model for linear updates.")

        # Additional logging
        logger.info(f"Using method: {self.method} with params: {self.params}")

    def eval_batch(self, model, batch: Union[DynData, DynGeoData], criterion) -> torch.Tensor:
        """
        Process a batch and return predictions and ground truth states.

        Only used in `evaluation` in this Trainer.
        """
        _p, _b = self._comp_linear_eval(model, batch, dt=self.dt)
        linear_loss = criterion(_p, _b)
        return linear_loss

    def update(self, model, dataloader: DataLoader) -> float:
        """Train the model for one epoch."""
        model.train()

        dtype, device = next(model.parameters()).dtype, next(model.parameters()).device

        with torch.no_grad():
            # Assemble the linear system
            A, b = [], []
            for batch in dataloader:
                _A, _b = self._comp_linear_features(model, batch, dt=self.dt)
                A.append(_A)
                b.append(_b)
            A = torch.cat(A, dim=0).cpu().numpy()
            b = torch.cat(b, dim=0).cpu().numpy()

            # Solve the linear system
            if self.method == 'full':
                W = np.linalg.lstsq(A, b, rcond=None)[0]
                Wt = torch.tensor(W, dtype=dtype, device=device)
                model.set_linear_weights(Wt.T)
                avg_epoch_loss = np.linalg.norm(A @ W - b) / A.shape[0]

            elif self.method == 'truncated':
                _V, _U = truncated_lstsq(A, b, tsvd=self.params)
                model.set_linear_weights(
                    U=torch.tensor(_U, dtype=dtype, device=device),
                    V=torch.tensor(_V, dtype=dtype, device=device))
                avg_epoch_loss = np.linalg.norm((A @ _V) @ _U.T - b) / A.shape[0]

        return avg_epoch_loss
