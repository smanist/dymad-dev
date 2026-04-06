"""Typed adapter seam for checkpoint-backed spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from dymad.numerics import complex_grid, disc2cont
from dymad.sako.rals import RALowRank, estimate_pseudospectrum
from dymad.sako.sako import SAKO
from dymad.sako.snapshot import SpectralSnapshot


class SpectralRuntime(Protocol):
    """Runtime hooks needed by measure and Jacobian compatibility methods."""

    def apply_obs(self, fobs: Any) -> np.ndarray: ...

    def get_forward_modes(self, ref: np.ndarray, rng: Any = None, **kwargs: Any) -> np.ndarray: ...

    def get_backward_modes(self, ref: np.ndarray, rng: Any = None, **kwargs: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class SpectralEigensystem:
    """Eigensystem terms consumed by the spectral adapter kernels."""

    discrete_eigs: np.ndarray
    left_eigvecs: np.ndarray
    right_eigvecs: np.ndarray
    projector: np.ndarray
    dt: float = 1.0


class SpectralAnalysisAdapter:
    """Adapter that delegates spectral operations to SAKO and RALowRank kernels."""

    def __init__(
        self,
        *,
        snapshot: SpectralSnapshot,
        eigensystem: SpectralEigensystem,
        runtime: SpectralRuntime | None = None,
        reps: float = 1e-10,
        etol: float = 1e-13,
    ):
        self.snapshot = snapshot
        self._runtime = runtime
        self._wd = eigensystem.discrete_eigs
        self._vl = eigensystem.left_eigvecs
        self._vr = eigensystem.right_eigvecs
        self._proj = eigensystem.projector
        self._dt = eigensystem.dt
        self._wc = disc2cont(self._wd, self._dt)

        self._sako = SAKO(
            self.snapshot.encoded_p0,
            self.snapshot.encoded_p1,
            None,
            reps=reps,
            etol=etol,
        )
        self._rals = RALowRank(
            self._vr,
            np.diag(self._wc.conj()),
            self._vl,
            dt=self._dt,
        )

    @property
    def sako(self) -> SAKO:
        return self._sako

    @property
    def rals(self) -> RALowRank:
        return self._rals

    def estimate_ps(
        self, grid=None, return_vec: bool = False, mode: str = "cont", method: str = "standard"
    ):
        """Estimate pseudospectrum over a grid via the current adapter kernels."""
        _grid = complex_grid(grid)
        result = estimate_pseudospectrum(
            _grid,
            self.resolvent_analysis,
            return_vec=return_vec,
            **{"mode": mode, "method": method},
        )
        return _grid, result

    def estimate_measure(self, fobs, order, eps, thetas: int = 101):
        """Estimate measure using runtime observable mapping plus SAKO kernel."""
        if self._runtime is None:
            gobs = np.asarray(fobs).reshape(-1)
        else:
            gobs = np.asarray(self._runtime.apply_obs(fobs)).reshape(-1)
        return self._sako.estimate_measure(gobs, order, eps, thetas)

    def resolvent_analysis(self, z, return_vec, mode, method):
        """Resolve one pseudospectrum point through SAKO or standard low-rank RA."""
        _method = method.lower()
        _ifcont = mode.lower() == "cont"

        if _method == "sako":
            if _ifcont:
                _z = np.exp(z * self._dt)
            else:
                _z = z

            if return_vec:
                _e, _v = self._sako._ps_point(_z, True)
                _b = self._proj.dot(_v)
                _ls = self._wd.conj().reshape(-1, 1)
                _u = (self._vr * _b).dot(_ls).reshape(-1)
                _u -= _z * _v
            else:
                _e = self._sako._ps_point(_z, False)

            if _ifcont:
                _e *= self._dt
            if return_vec:
                return _e, _v, _u
            return _e

        if _method == "standard":
            return self._rals(z, return_vec, mode)

        raise ValueError(f"Method {_method} unknown for resolvent analysis")

    def eval_eigfunc_jac(self, ref=None, rng=None, **kwargs) -> np.ndarray:
        """Evaluate Jacobian of eigenfunctions via runtime forward-mode hooks."""
        runtime = self._require_runtime()
        if ref is None:
            ref = np.zeros((1, self.snapshot.input_dim))
        mode = runtime.get_forward_modes(ref, rng, **kwargs)
        return self._vl.T.dot(mode)

    def eval_eigmode_jac(self, ref=None, rng=None, **kwargs) -> np.ndarray:
        """Evaluate Jacobian of eigenmodes via runtime backward-mode hooks."""
        runtime = self._require_runtime()
        if ref is None:
            ref = np.zeros((1, self.snapshot.obs_dim))
        mode = runtime.get_backward_modes(ref, rng, **kwargs)
        return self._vr.T.dot(mode)

    def _require_runtime(self) -> SpectralRuntime:
        if self._runtime is None:
            raise RuntimeError("Spectral runtime hooks are required for Jacobian evaluation.")
        return self._runtime
