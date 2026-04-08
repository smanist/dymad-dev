from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import torch

from dymad.agent.exec.context import build_default_context
from dymad.io import DataInterface
from dymad.models.collections import DKBF, KBF
from dymad.numerics import (
    check_orthogonality,
    disc2cont,
    eig_low_rank,
    scaled_eig,
    truncate_sequence,
)
from dymad.sako.adapter import SpectralEigensystem
from dymad.sako.plotting import SpectralPlottingAdapter
from dymad.sako.snapshot import SpectralSnapshot, build_spectral_snapshot

logger = logging.getLogger(__name__)


def filter_spectrum(sako, eigs, order="full", remove_one=True):
    """
    Apply SAKO to the identified eigenpairs to compute the corresponding residuals
    """
    wd, vl, vr = eigs
    res = sako.estimate_residual(wd, vr)

    # Full set
    _msk = np.argsort(res)
    res_full = res[_msk]
    wd_full = wd[_msk]
    vl_full = vl[:, _msk]
    vr_full = vr[:, _msk]

    # Truncated set
    idx = truncate_sequence(res_full, order)
    jdx = []  # Ensure all conjugates appear simultaneously
    for _i in idx:
        if _i not in jdx:
            jdx.append(_i)
            _w = wd_full[_i]
            _j = np.argmin(np.abs(wd_full - _w.conj()))
            if _j not in idx:
                logger.info(f"Adding missing conjugate {_j}: {wd_full[_j]:5.4e}")
            if _j not in jdx:
                jdx.append(_j)
    res = res_full[jdx]
    wd = wd_full[jdx]
    vl = vl_full[:, jdx]
    vr = vr_full[:, jdx]

    if remove_one:
        _d = np.abs(wd - 1.0)
        if np.min(_d) < 1e-10:
            _i = np.argmin(_d)
            logger.info(
                f"Removing eigenvalue {wd[_i]:5.4e} close to 1.0 with residual {res[_i]:3.1e}"
            )
            _m = np.arange(len(wd)) != _i
            wd = wd[_m]
            vl = vl[:, _m]
            vr = vr[:, _m]
            res = res[_m]

    return (wd, vl, vr), (wd_full, vl_full, vr_full), (res, res_full)


def encode_runtime_batch(model: torch.nn.Module, batch) -> np.ndarray:
    """Encode one trainer batch via typed-runtime payloads."""
    runtime = batch.runtime if hasattr(batch, "runtime") else batch
    encoder = cast(Any, model.encoder)
    return encoder(runtime).cpu().detach().numpy()


class SAInterface(DataInterface):
    """
    Interface for spectral analysis of KBF and DKBF models.
    """

    def __init__(
        self,
        model_class: type[torch.nn.Module],
        checkpoint_path: str,
        device: torch.device | None = None,
    ):
        assert model_class in [KBF, DKBF], (
            "Spectral Analysis is currently only implemented for KBF and DKBF."
        )
        self._checkpoint_path = str(checkpoint_path)

        super().__init__(model_class=model_class, checkpoint_path=checkpoint_path, device=device)

        self._setup_sa_terms()

        logger.info("SAInterface Initialized:")
        logger.info(self.model)
        logger.info(self.model.diagnostic_info())
        logger.info(f"Using device: {self.device}")

    def _setup_sa_terms(self):
        P0, P1 = [], []
        for batch in self.train_loader:
            _P = encode_runtime_batch(self.model, batch)
            _P0, _P1 = _P[..., :-1, :], _P[..., 1:, :]
            _P0 = _P0.reshape(-1, _P0.shape[-1])
            _P1 = _P1.reshape(-1, _P1.shape[-1])
            P0.append(_P0)
            P1.append(_P1)
        self._P0 = np.concatenate(P0, axis=0)
        self._P1 = np.concatenate(P1, axis=0)

        self._Ninp = self._trans_x._inp_dim
        if self._Ninp is None:
            raise ValueError("Spectral analysis requires a known transform input dimension.")
        self._Nout = self.model.koopman_dimension
        self._snapshot = build_spectral_snapshot(
            model_class=type(self.model).__name__,
            checkpoint_path=self._checkpoint_path,
            encoded_p0=self._P0,
            encoded_p1=self._P1,
            weights=self.get_weights(),
            input_dim=self._Ninp,
            obs_dim=self._Nout,
            metadata={
                "processor_mode": self.model.processor_net.mode,
            },
        )

    def get_weights(self) -> tuple[np.ndarray, ...]:
        """
        Get the linear weights of the dynamics model.
        """
        if self.model.processor_net.mode == "full":
            return (self.model.processor_net.weight.data.cpu().numpy(),)
        else:
            U = self.model.processor_net.U.data.cpu().numpy()
            V = self.model.processor_net.V.data.cpu().numpy()
            return (U, V)

    @property
    def snapshot(self) -> SpectralSnapshot:
        """Typed spectral snapshot extracted from checkpoint-backed model state."""
        return self._snapshot


class SpectralAnalysis:
    """
    The base class for Spectral Analysis based on Koopman operator theory.

    The formulation is based on the following convention:
    Psi_0 A = Psi_1
    where A is the finite-dimensional approximation of Koopman operator,
    Psi's are data matrices with each row containing one time step.

    Args:
        dt: Time step size.
    """

    def __init__(
        self,
        model_class: type[torch.nn.Module],
        checkpoint_path: str,
        forder="full",
        dt: float = 1.0,
        reps: float = 1e-10,
        remove_one=True,
        etol: float = 1e-13,
    ):
        self._dt = dt
        self._reps = reps
        self._etol = etol
        self._exec_context = build_default_context()
        self._spectral_plan = None
        self._reset()

        self._ctx = SAInterface(model_class, checkpoint_path)

        self._solve_eigs()
        logger.info(f"Orthonormality violation: {check_orthogonality(self._vl, self._vr)[1]:4.3e}")
        self._proc_eigs()
        self._refresh_adapter()
        self._plotting = SpectralPlottingAdapter(self)

        self.filter_spectrum(forder, remove_one=remove_one)

    def predict(self, x0, tseries, return_obs=False):
        """
        Make time-domain prediction.

        Args:
            x0: Initial states
            tseries: Time series at which to evaluate the solutions.
            return_obs: If return observables over time as well
        """
        _ts = tseries - tseries[0]
        _p0 = np.atleast_2d(self._ctx.encode(x0))  # (n_batch, n_dim)
        # Project initial conditions
        _b = self._proj.dot(_p0.T)  # (n_modes, n_batch)
        _ls = np.exp(self._wc.reshape(-1, 1) * _ts)  # (n_modes, n_steps)
        # Time evolution for each batch
        # vr (n_dim, n_modes)
        _pt = np.einsum("ij,jk,jl->kli", self._vr, _b, _ls)  # (n_batch, n_steps, n_dim)
        # Decode each trajectory
        _xt = self._ctx.decode(_pt.real).squeeze()

        if return_obs:
            return _xt, _pt.squeeze()
        return _xt

    def estimate_measure(self, fobs, order, eps, thetas=101):
        """
        Estimate the measure of the observable along the unit circle.
        """
        return self._adapter.estimate_measure(fobs, order, eps, thetas)

    def eval_eigfun(self, X, idx, rng=None):
        """
        Evaluate the eigenfunctions at given locations, possibly in embedded space
        """
        _P = self._ctx.encode(X, rng)
        return _P.dot(self._vl[:, idx])

    def eval_eigfun_par(self, par, idx, func, rng=None):
        """
        Evaluate the eigenfunctions at given parametrization
        """
        _P = self._ctx.encode(func(par), rng)
        return _P.dot(self._vl[:, idx])

    def eval_eigfunc_jac(self, ref=None, rng=None, **kwargs) -> np.ndarray:
        return self._adapter.eval_eigfunc_jac(ref=ref, rng=rng, **kwargs)

    def eval_eigmode_jac(self, ref=None, rng=None, **kwargs) -> np.ndarray:
        return self._adapter.eval_eigmode_jac(ref=ref, rng=rng, **kwargs)

    def set_conj_map(self, J):
        """
        Compute the conjugacy map assuming an equilibrium point at x=0 with Jacobian J.
        Consider eigendecomposition: J = W * L * V^H
        locally a principal eigenfunction is approximately phi_i(x) = v_i^H x
        """
        _wl, _vl, _vr = scaled_eig(J)
        _N = len(J)
        assert len(_wl) <= len(self._wc)  # Insufficient Koopman dimensions
        _idx = []
        _sgn = []
        _eps = 1e-6
        logger.info("Computing conjugacy map:")
        for _j, _w in enumerate(_wl):
            # Identify the principal eigenfunction
            _d = np.abs(self._wc - _w)
            _i = np.argmin(_d)
            logger.info(
                f"EV: Jacobian {_w:5.4e}, Koopman {self._wc[_i]:5.4e}, diff {np.abs(_d[_i] / self._wc[_i]):5.4e}"
            )
            _idx.append(_i)
            # Check the sign by evaluating along w_i, and v_i^H w_i = +/- 1
            _f1 = self.eval_eigfun(_eps * _vl[:, _j].reshape(1, -1), _i)
            _f0 = self.eval_eigfun(np.zeros((1, _N)), _i)  # Supposed to be 0
            _vw = (_f1 - _f0) / _eps
            _sgn.append(np.sign(_vw.real))
        _sgn = np.array(_sgn).reshape(-1)
        logger.info(f"Flipping: {_sgn}")
        _T = _vl * _sgn
        # The mappings
        self.mapto_cnj = lambda X, I=_idx, W=_T: self.eval_eigfun(X, I).dot(W.T)
        self.mapto_nrm = lambda X, I=_idx, S=_sgn: self.eval_eigfun(X, I) * S

    def filter_spectrum(self, order="full", remove_one=True):
        """
        Apply SAKO to the identified eigenpairs to compute the corresponding residuals
        """
        eigs, eigs_full, res = filter_spectrum(
            self._sako, (self._wd_full, self._vl_full, self._vr_full), order, remove_one=remove_one
        )

        self._wd, self._vl, self._vr = eigs
        self._wd_full, self._vl_full, self._vr_full = eigs_full
        self._res, self._res_full = res

        self._Nrank = len(self._wd)

        # Redo the eigenvalue processing
        self._proc_eigs()
        self._refresh_adapter()

    def estimate_ps(self, grid=None, return_vec=False, mode="cont", method="standard"):
        """
        Estimate pseudospectrum over a grid.

        In `disc` mode, the grid is assumed to be on discrete-time complex plane;
        the estimator should perform discrete-time resolvent analysis, using time step size of data
        In `cont` mode, the grid is assumed to be on continuous-time complex plane;
        the estimator should perform continuous-time resolvent analysis

        Args:
            grid: Mode disc: points on discrete-time plane (Re, Im)
                Mode cont: points on continuous-time plane (zeta, omega)
            return_vec: If return I/O modes
            mode: 'cont' or 'disc'
        """
        logger.info(f"Estimating PS: Mode:{mode} Method:{method}")
        return self._adapter.estimate_ps(
            grid=grid,
            return_vec=return_vec,
            mode=mode,
            method=method,
        )

    def resolvent_analysis(self, z, return_vec, mode, method):
        """
        Perform resolvent analysis of the DMD operator.

        Args:
            method: 'standard' - The projected approach where I/O modes are all in DMD mode space,
                    which is true for a low-rank DMD operator.
        """
        return self._adapter.resolvent_analysis(z, return_vec, mode, method)

    def _reset(self):
        # Dimensions
        self._Nrank = None
        # Raw eigensystem quantities
        self._wd_full = np.array([])  # Eigenvalues (discrete)
        self._wc_full = np.array([])  # Eigenvalues (continuous)
        self._vl_full = np.array([])  # Left eigenvectors
        self._vr_full = np.array([])  # Right eigenvectors
        # Retained eigensystem quantities
        self._wd = np.array([])  # Eigenvalues (discrete)
        self._vl = np.array([])  # Left eigenvectors
        self._vr = np.array([])  # Right eigenvectors
        # Residuals - not all DMD classes compute this
        self._res_full = np.array([])  # All residuals
        self._res = np.array([])  # Retained residuals
        # Derived quantities
        self._wc = np.array([])  # Eigenvalues (continuous)
        self._proj = np.array(
            []
        )  # Projector onto vl; should be vr, but this is for numerical robustness
        self.mapto_cnj = (
            None  # Conjugate mapping for systems with equilibrium point, to original Jacobian
        )
        self.mapto_nrm = (
            None  # Conjugate mapping for systems with equilibrium point, to orthogonal space
        )

    def _solve_eigs(self):
        weights = self._ctx.get_weights()
        _w: np.ndarray

        if len(weights) == 2:
            _Vr, _B = weights  # A = Vr @ B^T
            _w, self._vl, self._vr = eig_low_rank(_Vr, _B)

        elif len(weights) == 1:
            _W = weights[0]
            _w, self._vl, self._vr = scaled_eig(_W)

        if self._ctx.model.CONT:
            self._wd = np.exp(_w * self._dt)
        else:
            self._wd = _w

        # For data member consistency
        self._wd_full = self._wd
        self._vl_full = self._vl
        self._vr_full = self._vr

        self._Nrank = len(self._wd)

    def _proc_eigs(self):
        """
        Computes several data members for subsequent processing.
        """
        self._wc_full = disc2cont(self._wd_full, self._dt)
        self._wc = disc2cont(self._wd, self._dt)
        # self._proj = np.linalg.solve(self._vr.conj().T.dot(self._vr), self._vr.conj().T)
        self._proj = self._vl.conj().T  # Mathemetically correct, but numerically inaccurate.

    def _refresh_adapter(self):
        eigensystem = SpectralEigensystem(
            discrete_eigs=self._wd,
            left_eigvecs=self._vl,
            right_eigvecs=self._vr,
            projector=self._proj,
            dt=self._dt,
        )
        if self._spectral_plan is None:
            model = self._ctx.model
            model_ref = f"{type(model).__module__}:{type(model).__name__}"
            self._spectral_plan = self._exec_context.executor.plan_spectral_analysis(
                model_ref=model_ref,
                checkpoint_path=self._ctx.snapshot.checkpoint_path,
                snapshot=self._ctx.snapshot,
            )

        self._adapter = self._exec_context.executor.materialize_spectral_adapter(
            plan=self._spectral_plan,
            eigensystem=eigensystem,
            runtime=self._ctx,
            reps=self._reps,
            etol=self._etol,
        )
        self._sako = self._adapter.sako
        self._rals = self._adapter.rals

    def plot_eigs(self, fig=None, plot_full="bo", plot_filt="r^", mode="disc"):
        """Plot the eigenvalues in the complex plane."""
        return self._plotting.plot_eigs(
            fig=fig, plot_full=plot_full, plot_filt=plot_filt, mode=mode
        )

    def plot_pred(
        self,
        x0s,
        ts,
        ref=None,
        ifobs=False,
        idx="all",
        ncols=1,
        figsize=(6, 8),
        title=None,
        fig=None,
    ):
        """Plot the predicted trajectories in data space or latent space."""
        return self._plotting.plot_pred(
            x0s=x0s,
            ts=ts,
            ref=ref,
            ifobs=ifobs,
            idx=idx,
            ncols=ncols,
            figsize=figsize,
            title=title,
            fig=fig,
        )

    def plot_eigfun_2d(
        self, rngs, Ns, idx, mode="angle", space="full", ncols=2, figsize=(6, 10), fig=None
    ):
        """Plot the 2D eigenfunctions as contours."""
        return self._plotting.plot_eigfun_2d(
            rngs=rngs,
            Ns=Ns,
            idx=idx,
            mode=mode,
            space=space,
            ncols=ncols,
            figsize=figsize,
            fig=fig,
        )

    def plot_eigjac_contour(
        self,
        ref=None,
        rng=None,
        eig="func",
        lam="ct",
        comp="ri",
        idx="all",
        shape=(),
        contour_args=None,
        **kwargs,
    ):
        """Plot contour maps for Jacobian eigfunction/eigmode components."""
        if contour_args is None:
            contour_args = {}
        return self._plotting.plot_eigjac_contour(
            ref=ref,
            rng=rng,
            eig=eig,
            lam=lam,
            comp=comp,
            idx=idx,
            shape=shape,
            contour_args=contour_args,
            **kwargs,
        )

    def plot_vec_line(self, idx, which="func", modes=None, ncols=1, figsize=(6, 10)):
        """Plot slices of eigenfunctions/eigenmodes as vectors."""
        if modes is None:
            modes = ["angle"]
        return self._plotting.plot_vec_line(
            idx=idx,
            which=which,
            modes=modes,
            ncols=ncols,
            figsize=figsize,
        )
