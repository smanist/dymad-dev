import inspect
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union

from dymad.data import DynData, DynGeoData
from dymad.numerics.linalg import real_lowrank_from_eigpairs, scaled_eig, truncated_lstsq
from dymad.sako import filter_spectrum, SAKO

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
            raise ValueError(f"{model} does not implement linear_features and linear_eval methods required for LS updates.")

        if self.method not in ['full', 'truncated', 'sako']:
            raise ValueError(f"Unsupported method: {self.method}. Supported methods are 'full', 'truncated', and 'sako'.")

        if model.CONT:
            if self.method == 'sako':
                logger.warning("SAKO is designed for discrete-time systems.")
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
                params = model.set_linear_weights(Wt.T)
                avg_epoch_loss = np.linalg.norm(A @ W - b) / A.shape[0]

            elif self.method == 'truncated':
                _V, _U = truncated_lstsq(A, b, tsvd=self.params)
                params = model.set_linear_weights(
                    U=torch.tensor(_U, dtype=dtype, device=device),
                    V=torch.tensor(_V, dtype=dtype, device=device))
                avg_epoch_loss = np.linalg.norm((A @ _V) @ _U.T - b) / A.shape[0]

            elif self.method == 'sako':
                W = np.linalg.lstsq(A, b, rcond=None)[0]
                _w, _vl, _vr = scaled_eig(W)
                sako = SAKO(A, b, reps=1e-10)
                eigs, _, res = filter_spectrum(sako, (_w, _vl, _vr), order=self.params)
                logger.info(f"SAKO filtered {len(_w)-len(eigs)} out of {len(_w)} eigenvalues. Max residual: {max(res[0]):3.1e}")

                _B, _R, _S = real_lowrank_from_eigpairs(*eigs)
                # S @ B @ R.T = _vr @ _w @ _vl^H = W by linalg
                # W = V @ U^T for FlexLinear
                # So 
                _V = _S @ _B
                _U = _R

                params = model.set_linear_weights(
                    U=torch.tensor(_U, dtype=dtype, device=device),
                    V=torch.tensor(_V, dtype=dtype, device=device))
                avg_epoch_loss = np.linalg.norm((A @ _V) @ _U.T - b) / A.shape[0]

        return avg_epoch_loss, params
