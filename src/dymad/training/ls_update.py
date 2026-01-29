import inspect
import logging
import numpy as np
import scipy.linalg as spl
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from dymad.io import DynData
from dymad.numerics.linalg import logm_low_rank, real_lowrank_from_eigpairs, truncated_lstsq
from dymad.sako import filter_spectrum, SAKO

logger = logging.getLogger(__name__)

# ----------------
# Elementary functions
# ----------------
def _dt_target(z: torch.Tensor) -> torch.Tensor:
    """Compute discrete-time targets."""
    return z[..., 1:, :]

def _comp_linear_features_dt(model, batch: DynData, **kwargs) -> torch.Tensor:
    """Compute linear features for discrete-time models."""
    A, z = model.linear_features(batch)
    _A = A[..., :-1, :]
    _z = _dt_target(z)
    return _A.reshape(-1, _A.shape[-1]), _z.reshape(-1, _z.shape[-1])

def _comp_linear_eval_dt(model, batch: DynData, **kwargs) -> torch.Tensor:
    """Compute predicted targets for discrete-time models.
    z_dot really means z_next here.
    """
    z_dot, z = model.linear_eval(batch)
    return z_dot[..., :-1, :], _dt_target(z)

def _ct_target(z: torch.Tensor, dt, order=2) -> torch.Tensor:
    """Compute linear targets for continuous-time models."""
    if order == 1:
        dz = (z[..., 1:, :] - z[..., :-1, :]) / dt
        dz = torch.concatenate((dz, dz[..., -1:, :]), dim=-2)
        return dz
    elif order == 2:
        dz = np.gradient(z.cpu().numpy(), dt, axis=-2, edge_order=2)
        return torch.tensor(dz, dtype=z.dtype, device=z.device)
    else:
        raise ValueError(f"Unsupported FD order: {order}. Only 1 and 2 are supported.")

def _comp_linear_features_ct(model, batch: DynData, **kwargs) -> torch.Tensor:
    """Compute linear features for continuous-time models."""
    A, z = model.linear_features(batch)
    _z = _ct_target(z, kwargs['dt'], kwargs.get('order', 2))
    return A.reshape(-1, A.shape[-1]), _z.reshape(-1, _z.shape[-1])

def _comp_linear_eval_ct(model, batch: DynData, **kwargs) -> torch.Tensor:
    """Compute predicted targets for continuous-time models."""
    z_dot, z = model.linear_eval(batch)
    return z_dot, _ct_target(z, kwargs['dt'], kwargs.get('order', 2))

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

def check_linear_solve(model) -> bool:
    """
    Check if the model implements linear_solve method.
    """
    has_linear_solve = hasattr(model, 'linear_solve')
    if not has_linear_solve:
        return False

    source = inspect.getsource(model.linear_solve)
    if "raise NotImplementedError" in source:
        return False

    return True

# ----------------
# Helper functions
# ----------------
def get_batch_dt(dataloader, model, dt, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    A, b = [], []
    for batch in dataloader:
        _A, _b = _comp_linear_features_dt(model, batch, dt=dt, **kwargs)
        A.append(_A)
        b.append(_b)
    A = torch.cat(A, dim=0).cpu().numpy()
    b = torch.cat(b, dim=0).cpu().numpy()
    return A, b

def get_batch_ct(dataloader, model, dt, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    A, b = [], []
    for batch in dataloader:
        _A, _b = _comp_linear_features_ct(model, batch, dt=dt, **kwargs)
        A.append(_A)
        b.append(_b)
    A = torch.cat(A, dim=0).cpu().numpy()
    b = torch.cat(b, dim=0).cpu().numpy()
    return A, b

# ----------------
# LS solvers
# ----------------
def _ls_full(A: np.ndarray, b: np.ndarray, params=None) -> np.ndarray:
    """Full least squares solver."""
    W = np.linalg.lstsq(A, b, rcond=None)[0]
    return W

def _ls_truncated(A: np.ndarray, b: np.ndarray, params=None) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated least squares solver."""
    tsvd = params
    V, U = truncated_lstsq(A, b, tsvd=tsvd)
    return V, U

def _ls_sako(A: np.ndarray, b: np.ndarray, params=None) -> Tuple[np.ndarray, np.ndarray]:
    """Using the SAKO object."""
    sako = SAKO(A, b, reps=1e-10, etol=1e-13)
    _w, _vl, _vr = sako.solve_eig()
    if isinstance(params, list):
        order = params[0]
        remove_one = params[1] if len(params) > 1 else True
    else:
        order = params
        remove_one = True
    eigs, _, res = filter_spectrum(sako, (_w, _vl, _vr), order=order, remove_one=remove_one)
    logger.info(f"SAKO filtered {len(_w)-len(eigs)} out of {len(_w)} eigenvalues. Max residual: {max(res[0]):3.1e}")

    _B, _R, _S = real_lowrank_from_eigpairs(*eigs)
    # S @ B @ R.T = _vr @ _w @ _vl^H = W by linalg
    # W = V @ U^T for FlexLinear
    # So 
    _V = _S @ _B
    _U = _R

    return _V, _U

# ----------------
# Combinations
# ----------------
def _dt_full(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    W = _ls_full(A, b, params)
    return (A, b), (W,)

def _dt_truncated(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    V, U = _ls_truncated(A, b, params)
    return (A, b), (V, U)

def _dt_sako(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    V, U = _ls_sako(A, b, params)
    return (A, b), (V, U)

def _ct_full_der(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_ct(dataloader, model, dt, **kwargs)
    W = _ls_full(A, b, params)
    return (A, b), (W,)

def _ct_truncated_der(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_ct(dataloader, model, dt, **kwargs)
    V, U = _ls_truncated(A, b, params)
    return (A, b), (V, U)

def _ct_full_log(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    W = _ls_full(A, b, params)
    W = spl.logm(W) / dt
    return (A, b), (W,)

def _ct_truncated_log(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    V, U = _ls_truncated(A, b, params)
    V, U = logm_low_rank(V, U, dt=dt)
    return (A, b), (V, U)

def _ct_sako_log(dataloader, model, dt, params=None, **kwargs):
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    V, U = _ls_sako(A, b, params)
    V, U = logm_low_rank(V, U, dt=dt)
    return (A, b), (V, U)

# The two below behave differently from the above,
# as they use the model's own linear_solve method
def _dt_raw(dataloader, model, dt, params=None, **kwargs):
    _p = {} if params is None else params
    A, b = get_batch_dt(dataloader, model, dt, **kwargs)
    W, r = model.linear_solve(A, b, **_p)
    return (W,), r

def _ct_raw(dataloader, model, dt, params=None, **kwargs):
    _p = {} if params is None else params
    A, b = get_batch_ct(dataloader, model, dt, **kwargs)
    W, r = model.linear_solve(A, b, **_p)
    return (W,), r

#: Mapping of linear solver methods
SOL_MAP = {
    'dt_full'      : _dt_full,
    'dt_truncated' : _dt_truncated,
    'dt_sako'      : _dt_sako,
    'ct_full'      : _ct_full_der,
    'ct_truncated' : _ct_truncated_der,
    'ct_full_log'  : _ct_full_log,
    'ct_truncated_log' : _ct_truncated_log,
    'ct_sako_log'  : _ct_sako_log,
    'dt_raw'       : _dt_raw,
    'ct_raw'       : _ct_raw,
}

class LSUpdater:
    """
    Update linear weights by least squares.
    """

    def __init__(self, method, model, dt=None, params=None, **kwargs):
        self.params = params
        self.dt     = dt
        self.kwargs = kwargs

        if not check_linear_impl(model):
            raise ValueError(f"{model} does not implement linear_features and linear_eval methods required for LS updates.")

        prefix = 'ct_' if model.CONT else 'dt_'
        if check_linear_solve(model):
            self.method = prefix + 'raw'
            self.solver = SOL_MAP[self.method]
            logger.info(f"{model} has linear_solve, default to {self.method}.")
        else:
            self.method = prefix + method
            if self.method not in SOL_MAP:
                raise ValueError(f"Unsupported method: {self.method}. Supported methods are {list(SOL_MAP.keys())}.")
            self.solver = SOL_MAP[self.method]

        if model.CONT:
            self._comp_linear_eval = _comp_linear_eval_ct
            logger.info(f"Using continuous-time model for linear updates, dt={self.dt}.")
        else:
            self._comp_linear_eval = _comp_linear_eval_dt
            logger.info("Using discrete-time model for linear updates.")

        # Additional logging
        logger.info(f"Using method: {self.method} with params: {self.params}")

    def eval_batch(self, model, batch: DynData, criterion) -> torch.Tensor:
        """
        Process a batch and return predictions and ground truth states.

        Only used in `evaluation` in this Trainer.
        """
        _p, _b = self._comp_linear_eval(model, batch, dt=self.dt, **self.kwargs)
        linear_loss = criterion(_p, _b)
        return linear_loss

    def update(self, model, dataloader: DataLoader) -> float:
        """Train the model for one epoch."""
        model.train()

        dtype, device = next(model.parameters()).dtype, next(model.parameters()).device

        if self.method.endswith('raw'):
            # Use model's own linear_solve method
            with torch.no_grad():
                params, residual = self.solver(dataloader, model, self.dt, self.params, **self.kwargs)
                avg_epoch_loss = residual.mean().item()
            return avg_epoch_loss, params

        with torch.no_grad():
            (A, b), weights = self.solver(dataloader, model, self.dt, self.params, **self.kwargs)

            if len(weights) == 1:
                W = weights[0]
                Wt = torch.tensor(W, dtype=dtype, device=device)
                params = model.set_linear_weights(Wt.T)
                avg_epoch_loss = np.linalg.norm(A @ W - b) / A.shape[0]
            else:
                _V, _U = weights
                avg_epoch_loss = np.linalg.norm((A @ _V) @ _U.T - b) / A.shape[0]
                params = model.set_linear_weights(
                    U=torch.tensor(_U, dtype=dtype, device=device),
                    V=torch.tensor(_V, dtype=dtype, device=device))

        return avg_epoch_loss, params
