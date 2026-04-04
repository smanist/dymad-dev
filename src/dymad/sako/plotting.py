"""Optional plotting adapter for spectral-analysis compatibility surfaces."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dymad.numerics import complex_map, mode_split
from dymad.utils import plot_contour

LMAP = {
    "r": "Real",
    "i": "Imag",
    "a": "Amp.",
    "p": "Phase",
}


def per_state_err(prd: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute per-state prediction error between trajectories."""
    norm_diff = np.linalg.norm(prd - ref, axis=1)
    norm_ref = np.sqrt(prd.shape[1]) * (np.max(ref, axis=1) - np.min(ref, axis=1))
    return np.mean(norm_diff / norm_ref, axis=0)


class SpectralPlottingAdapter:
    """Holds plotting helpers so numerical analysis seams stay presentation-free."""

    def __init__(self, analysis: Any):
        self._analysis = analysis

    def plot_eigs(self, fig=None, plot_full="bo", plot_filt="r^", mode="disc"):
        """Plot the eigenvalues in the complex plane."""
        analysis = self._analysis
        if fig is None:
            f, ax = plt.subplots()
        else:
            f, ax = fig
        if mode.lower() == "disc":
            _t = np.linspace(0, 2 * np.pi, 101)
            ax.plot(np.sin(_t), np.cos(_t), "k--")
            _l1, _l2 = None, None
            if plot_full:
                _l1, = ax.plot(analysis._wd_full.real, analysis._wd_full.imag, plot_full, markerfacecolor="none")
            if plot_filt:
                _l2, = ax.plot(analysis._wd.real, analysis._wd.imag, plot_filt)
            ax.set_aspect("equal")
        elif mode.lower() == "cont":
            _l1, _l2 = None, None
            if plot_full:
                _l1, = ax.plot(analysis._wc_full.real, analysis._wc_full.imag, plot_full, markerfacecolor="none")
            if plot_filt:
                _l2, = ax.plot(analysis._wc.real, analysis._wc.imag, plot_filt)
            ax.set_aspect("equal")
        else:
            raise ValueError(f"Unknwon mode {mode} for plotting spectrum")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        _ls = []
        if _l1 is not None:
            _ls.append(_l1)
        if _l2 is not None:
            _ls.append(_l2)
        return f, ax, _ls

    def plot_pred(self, x0s, ts, ref=None, ifobs=False, idx="all", ncols=1, figsize=(6, 8), title=None, fig=None):
        """Plot predicted trajectories in data space or latent space."""
        analysis = self._analysis
        if idx == "all":
            if ifobs:
                _idx = np.arange(analysis._ctx._Nout, dtype=int)
            else:
                _idx = np.arange(analysis._ctx._Ninp, dtype=int)
        elif isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)
        _Nst = len(_idx)

        if ifobs:
            _prds = analysis.predict(x0s, ts, return_obs=True)[1].real
            _ylbl = "Obs"
        else:
            _prds = analysis.predict(x0s, ts, return_obs=False).real
            _ylbl = "State"
        if _prds.ndim == 2:
            _prds = np.array([_prds])
        _Nx0 = len(_prds)

        if ref is None:
            _refs, _errs = None, None
        else:
            if ifobs:
                _refs = analysis._ctx.encode(ref).real
            else:
                _refs = np.array(ref)
            if _refs.ndim == 2:
                _refs = np.array([_refs])
            _errs = per_state_err(_prds, _refs)

        _nr = _Nst // ncols + _Nst % ncols
        if fig is None:
            f, _ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        else:
            f, _ax = fig
        ax = _ax.flatten()
        for _k, _j in enumerate(_idx):
            for _i in range(_Nx0):
                l1, = ax[_k].plot(ts, _prds[_i][:, _j], "b-")
                if _refs is not None:
                    l2, = ax[_k].plot(ts, _refs[_i][:, _j], "r--")
            ax[_k].set_ylabel(f"{_ylbl} {_j}")
        if _refs is not None:
            for _k, _j in enumerate(_idx):
                ax[_k].set_title(f"{title}, Error {_errs[_j] * 100:3.2f}%")
            ax[0].legend([l1, l2], ["Prediction", "Reference"])
        ax[-1].set_xlabel("time, s")

        return f, ax

    def plot_eigfun_2d(self, rngs, Ns, idx, mode="angle", space="full", ncols=2, figsize=(6, 10), fig=None):
        """Plot 2D eigenfunctions as contours."""
        analysis = self._analysis
        x1s = np.linspace(rngs[0][0], rngs[0][1], Ns[0])
        x2s = np.linspace(rngs[1][0], rngs[1][1], Ns[1])
        X1, X2 = np.meshgrid(x1s, x2s)

        if isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)

        _tmp = np.vstack([X1.reshape(-1), X2.reshape(-1)]).T
        if space == "full":
            _fun = analysis.eval_eigfun(_tmp, _idx)
        elif callable(space):
            _fun = analysis.eval_eigfun_par(_tmp, _idx, func=space)
        else:
            _fun = analysis.eval_eigfun(_tmp, _idx, rng=space)

        _func = complex_map[mode]
        _Np = len(_idx)
        _nr = _Np // ncols + _Np % ncols
        if fig is None:
            f, ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        else:
            f, ax = fig
        _ax = ax.flatten()
        for _i, _j in enumerate(_idx):
            _F = _fun[:, _i].reshape(Ns[1], Ns[0])
            _ax[_i].contourf(X1, X2, _func(_F), levels=20)
            _ax[_i].set_title(f"{_j}: {np.angle(analysis._wc[_j]):3.2e} / {analysis._res[_j].real:3.2e}")

        return f, ax

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
        """Plot contour maps of Jacobian eigfunction/eigmode components."""
        analysis = self._analysis
        contour_args = {} if contour_args is None else dict(contour_args)
        assert len(shape) == 2, "Shape should be of length 2 for contour plotting"

        if eig == "func":
            _modes = analysis.eval_eigfunc_jac(ref, rng, **kwargs)
        elif eig == "mode":
            _ref = None
            if ref is not None:
                _ref = analysis._ctx.encode(ref)
            _modes = analysis.eval_eigmode_jac(_ref, rng, **kwargs)
        else:
            raise ValueError(f"Unknown eig {eig} for Jacobian plotting")

        _w = analysis._wc if lam == "ct" else analysis._wd
        ls, ms = mode_split(_w, _modes, comp=comp)
        if idx == "all":
            idx = np.arange(len(ls), dtype=int)
        assert isinstance(idx, (list, np.ndarray)), "idx should be a list or array of integers"
        ls, ms = ls[idx], ms[idx]

        if comp == "p":
            vdx = [0]
        else:
            vdx = []
            for _i, _c in enumerate(comp):
                if _c != "p":
                    vdx.append(_i)
        tmp = ms[:, vdx]
        vmin, vmax = np.min(tmp), np.max(tmp)

        labels = []
        for _i, _idx in enumerate(idx):
            tmp = []
            for _j, _c in enumerate(comp):
                if _j == 0:
                    _prf = f"Mode {_idx}: "
                else:
                    _prf = ""
                lbl = f"{_prf}{LMAP[_c]} / {ls[_i][_j]:3.2e}"
                tmp.append(lbl)
            labels += tmp

        contour_args.update(label=labels, vmin=vmin, vmax=vmax)
        f, ax = plot_contour(ms.reshape(len(labels), *shape), **contour_args)

        return (f, ax), (ls, ms)

    def plot_vec_line(self, idx, which="func", modes=None, ncols=1, figsize=(6, 10)):
        """Plot slices of eigenfunctions/eigenmodes as vectors."""
        analysis = self._analysis
        modes = ["angle"] if modes is None else modes

        if isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)

        if which == "func":
            _vec = analysis._vl
        elif which == "mode":
            _vec = analysis._vr
        else:
            raise ValueError(f"Unknown quantity to plot: {which}")

        _fs = [complex_map[_m] for _m in modes]
        _Np = len(_idx)
        _nr = _Np // ncols + _Np % ncols
        f, ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        _ax = ax.flatten()
        for _i in _idx:
            for _f in _fs:
                _ax[_i].plot(_f(_vec[_i]))
            _ax[_i].set_title(f"{_i}: {np.angle(analysis._wc[_i]):3.2e} / {analysis._res[_i].real:3.2e}")

        return f, ax
