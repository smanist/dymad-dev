import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple, Type

from dymad.io import DataInterface, DynData
from dymad.models import KBF, DKBF
from dymad.numerics import check_orthogonality, complex_grid, complex_map, disc2cont, eig_low_rank, mode_split, scaled_eig, truncate_sequence
from dymad.sako.rals import estimate_pseudospectrum, RALowRank
from dymad.sako.sako import SAKO
from dymad.utils import plot_contour

logger = logging.getLogger(__name__)

LMAP = {
    'r': 'Real',
    'i': 'Imag',
    'a': 'Amp.',
    'p': 'Phase'
}

def filter_spectrum(sako, eigs, order='full', remove_one=True):
    """
    Apply SAKO to the identified eigenpairs to compute the corresponding residuals
    """
    wd, vl, vr = eigs
    res = sako.estimate_residual(wd, vr)

    # Full set
    _msk = np.argsort(res)
    res_full = res[_msk]
    wd_full = wd[_msk]
    vl_full = vl[:,_msk]
    vr_full = vr[:,_msk]

    # Truncated set
    idx = truncate_sequence(res_full, order)
    jdx = []  # Ensure all conjugates appear simultaneously
    for _i in idx:
        if _i not in jdx:
            jdx.append(_i)
            _w = wd_full[_i]
            _j = np.argmin(np.abs(wd_full-_w.conj()))
            if _j not in idx:
                logger.info(f"Adding missing conjugate {_j}: {wd_full[_j]:5.4e}")
            if _j not in jdx:
                jdx.append(_j)
    res = res_full[jdx]
    wd = wd_full[jdx]
    vl = vl_full[:,jdx]
    vr = vr_full[:,jdx]

    if remove_one:
        _d = np.abs(wd-1.0)
        if np.min(_d) < 1e-10:
            _i = np.argmin(_d)
            logger.info(f"Removing eigenvalue {wd[_i]:5.4e} close to 1.0 with residual {res[_i]:3.1e}")
            _m = np.arange(len(wd)) != _i
            wd = wd[_m]
            vl = vl[:,_m]
            vr = vr[:,_m]
            res = res[_m]

    return (wd, vl, vr), (wd_full, vl_full, vr_full), (res, res_full)

def per_state_err(prd, ref):
    """
    Compute the per-state error between predicted and reference trajectories.
    """
    # (n_batch, n_steps, n_states)
    norm_diff = np.linalg.norm(prd - ref, axis=1)   # (n_batch, n_states)
    norm_ref  = np.sqrt(prd.shape[1]) * (np.max(ref, axis=1) - np.min(ref, axis=1))
    e0 = np.mean(norm_diff / norm_ref, axis=0)
    return e0

class SAInterface(DataInterface):
    """
    Interface for spectral analysis of KBF and DKBF models.
    """
    def __init__(self, model_class: Type[torch.nn.Module], checkpoint_path: str, device: Optional[torch.device] = None):
        assert model_class in [KBF, DKBF], "Spectral Analysis is currently only implemented for KBF and DKBF."

        super().__init__(model_class=model_class, checkpoint_path=checkpoint_path, device=device)

        self._setup_sa_terms()

        logger.info("SAInterface Initialized:")
        logger.info(self.model)
        logger.info(self.model.diagnostic_info())
        logger.info(f"Using device: {self.device}")

    def _setup_sa_terms(self):
        P0, P1 = [], []
        for batch in self.train_loader:
            _P = self.model.encoder(DynData(x=batch.x)).cpu().detach().numpy()
            _P0, _P1 = _P[..., :-1, :], _P[..., 1:, :]
            _P0 = _P0.reshape(-1, _P0.shape[-1])
            _P1 = _P1.reshape(-1, _P1.shape[-1])
            P0.append(_P0)
            P1.append(_P1)
        self._P0 = np.concatenate(P0, axis=0)
        self._P1 = np.concatenate(P1, axis=0)

        self._Ninp = self._trans_x._inp_dim
        self._Nout = self.model.koopman_dimension

    def get_weights(self) -> Tuple[np.ndarray]:
        """
        Get the linear weights of the dynamics model.
        """
        if self.model.processor_net.mode == "full":
            return (self.model.processor_net.weight.data.cpu().numpy(), )
        else:
            U = self.model.processor_net.U.data.cpu().numpy()
            V = self.model.processor_net.V.data.cpu().numpy()
            return (U, V)

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
    def __init__(self,
                 model_class: Type[torch.nn.Module], checkpoint_path: str,
                 forder='full', dt: float = 1.0, reps: float = 1e-10, remove_one=True, etol: float = 1e-13):
        self._dt = dt
        self._reset()

        self._ctx = SAInterface(model_class, checkpoint_path)

        self._solve_eigs()
        logger.info(f"Orthonormality violation: {check_orthogonality(self._vl, self._vr)[1]:4.3e}")
        self._proc_eigs()
        self._sako = SAKO(self._ctx._P0, self._ctx._P1, None, reps=reps, etol=etol)
        self._rals = RALowRank(self._vr, np.diag(self._wc.conj()), self._vl, dt=self._dt)

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
        _p0 = np.atleast_2d(self._ctx.encode(x0))    # (n_batch, n_dim)
        # Project initial conditions
        _b = self._proj.dot(_p0.T)                   # (n_modes, n_batch)
        _ls = np.exp(self._wc.reshape(-1, 1) * _ts)  # (n_modes, n_steps)
        # Time evolution for each batch
        # vr (n_dim, n_modes)
        _pt = np.einsum("ij,jk,jl->kli", self._vr, _b, _ls)  # (n_batch, n_steps, n_dim)
        # Decode each trajectory
        _xt = self._ctx.decode(_pt.real).squeeze()

        if return_obs:
            return _xt, _pt.squeeze()
        return _xt

    def estimate_measure(self, fobs, order, eps, thetas = 101):
        """
        Estimate the measure of the observable along the unit circle.
        """
        gobs = self._ctx.apply_obs(fobs).reshape(-1)
        return self._sako.estimate_measure(gobs, order, eps, thetas)

    def eval_eigfun(self, X, idx, rng=None):
        """
        Evaluate the eigenfunctions at given locations, possibly in embedded space
        """
        _P = self._ctx.encode(X, rng)
        return _P.dot(self._vl[:,idx])

    def eval_eigfun_par(self, par, idx, func, rng=None):
        """
        Evaluate the eigenfunctions at given parametrization
        """
        _P = self._ctx.encode(func(par), rng)
        return _P.dot(self._vl[:,idx])

    def eval_eigfunc_jac(self, ref=None, rng=None, **kwargs) -> np.ndarray:
        if ref is None:
            ref = np.zeros((1, self._ctx._Ninp))
        _mode = self._ctx.get_forward_modes(ref, rng, **kwargs)
        return self._vl.T.dot(_mode)

    def eval_eigmode_jac(self, ref=None, rng=None, **kwargs) -> np.ndarray:
        if ref is None:
            ref = np.zeros((1, self._ctx._Nout))
        _mode = self._ctx.get_backward_modes(ref, rng, **kwargs)
        return self._vr.T.dot(_mode)

    def set_conj_map(self, J):
        """
        Compute the conjugacy map assuming an equilibrium point at x=0 with Jacobian J.
        Consider eigendecomposition: J = W * L * V^H
        locally a principal eigenfunction is approximately phi_i(x) = v_i^H x
        """
        _wl, _vl, _vr = scaled_eig(J)
        _N = len(J)
        assert len(_wl) <= len(self._wc)   # Insufficient Koopman dimensions
        _idx = []
        _sgn = []
        _eps = 1e-6
        logger.info("Computing conjugacy map:")
        for _j, _w in enumerate(_wl):
            # Identify the principal eigenfunction
            _d = np.abs(self._wc-_w)
            _i = np.argmin(_d)
            logger.info("EV: Jacobian {0:5.4e}, Koopman {1:5.4e}, diff {2:5.4e}".format(
                _w, self._wc[_i], np.abs(_d[_i]/self._wc[_i])
            ))
            _idx.append(_i)
            # Check the sign by evaluating along w_i, and v_i^H w_i = +/- 1
            _f1 = self.eval_eigfun(_eps*_vl[:,_j].reshape(1,-1), _i)
            _f0 = self.eval_eigfun(np.zeros((1,_N)), _i)   # Supposed to be 0
            _vw =  (_f1-_f0) / _eps
            _sgn.append(np.sign(_vw.real))
        _sgn = np.array(_sgn).reshape(-1)
        logger.info(f"Flipping: {_sgn}")
        _T = _vl * _sgn
        # The mappings
        self.mapto_cnj = lambda X, I=_idx, W=_T: self.eval_eigfun(X, I).dot(W.T)
        self.mapto_nrm = lambda X, I=_idx, S=_sgn: self.eval_eigfun(X, I) * S

    def filter_spectrum(self, order='full', remove_one=True):
        """
        Apply SAKO to the identified eigenpairs to compute the corresponding residuals
        """
        eigs, eigs_full, res = filter_spectrum(
            self._sako, (self._wd_full, self._vl_full, self._vr_full), order,
            remove_one=remove_one)

        self._wd, self._vl, self._vr = eigs
        self._wd_full, self._vl_full, self._vr_full = eigs_full
        self._res, self._res_full = res

        self._Nrank = len(self._wd)

        # Redo the eigenvalue processing
        self._proc_eigs()

    def estimate_ps(self, grid=None, return_vec=False, mode='cont', method='standard'):
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
        _g = complex_grid(grid)
        res = estimate_pseudospectrum(_g, self.resolvent_analysis, return_vec=return_vec, \
            **{'mode':mode, 'method':method})
        return _g, res

    def resolvent_analysis(self, z, return_vec, mode, method):
        """
        Perform resolvent analysis of the DMD operator.

        Args:
            method: 'standard' - The projected approach where I/O modes are all in DMD mode space,
                    which is true for a low-rank DMD operator.
        """
        _method = method.lower()
        _ifcont = mode.lower() == 'cont'

        if _method == 'sako':
            if _ifcont:
                # In continuous mode, the inquiry point will be on continuous complex plane
                # But the SAKO formulation is always for discrete time.
                _z = np.exp(z*self._dt)
            else:
                _z = z
            if return_vec:
                _e, _v = self._sako._ps_point(_z, True)
                # _v is the output mode, then recover the input mode by
                # u=(K-zI)v
                _b  = self._proj.dot(_v)
                _ls = self._wd.conj().reshape(-1,1)
                _u = (self._vr*_b).dot(_ls).reshape(-1)
                _u -= _z*_v
            else:
                _e = self._sako._ps_point(_z, False)
            if _ifcont:
                # The gain is in discrete time, and we convert it back
                _e *= self._dt
            if return_vec:
                return _e, _v, _u
            return _e

        elif _method == 'standard':
            return self._rals(z, return_vec, mode)

        else:
            raise ValueError(f"Method {_method} unknown for resolvent analysis in {self._type}")

    def _reset(self):
        # Dimensions
        self._Nrank = None
        # Raw eigensystem quantities
        self._wd_full = np.array([])    # Eigenvalues (discrete)
        self._wc_full = np.array([])    # Eigenvalues (continuous)
        self._vl_full = np.array([])    # Left eigenvectors
        self._vr_full = np.array([])    # Right eigenvectors
        # Retained eigensystem quantities
        self._wd = np.array([])    # Eigenvalues (discrete)
        self._vl = np.array([])    # Left eigenvectors
        self._vr = np.array([])    # Right eigenvectors
        # Residuals - not all DMD classes compute this
        self._res_full = np.array([])   # All residuals
        self._res      = np.array([])   # Retained residuals
        # Derived quantities
        self._wc   = np.array([])  # Eigenvalues (continuous)
        self._proj = np.array([])  # Projector onto vl; should be vr, but this is for numerical robustness
        self.mapto_cnj = None      # Conjugate mapping for systems with equilibrium point, to original Jacobian
        self.mapto_nrm = None      # Conjugate mapping for systems with equilibrium point, to orthogonal space

    def _solve_eigs(self):
        weights = self._ctx.get_weights()

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
        self._proj = self._vl.conj().T   # Mathemetically correct, but numerically inaccurate.

    def plot_eigs(self, fig=None, plot_full='bo', plot_filt='r^', mode='disc'):
        """
        Plot the eigenvalues in the complex plane.
        """
        if fig is None:
            f, ax = plt.subplots()
        else:
            f, ax = fig
        if mode.lower() == 'disc':
            _t = np.linspace(0, 2*np.pi, 101)
            ax.plot(np.sin(_t), np.cos(_t), 'k--')
            _l1, _l2 = None, None
            if plot_full:
                _l1, = ax.plot(self._wd_full.real, self._wd_full.imag, plot_full, markerfacecolor='none')
            if plot_filt:
                _l2, = ax.plot(self._wd.real, self._wd.imag, plot_filt)
            ax.set_aspect('equal')
        elif mode.lower() == 'cont':
            _l1, _l2 = None, None
            if plot_full:
                _l1, = ax.plot(self._wc_full.real, self._wc_full.imag, plot_full, markerfacecolor='none')
            if plot_filt:
                _l2, = ax.plot(self._wc.real, self._wc.imag, plot_filt)
            ax.set_aspect('equal')
        else:
            raise ValueError(f"Unknwon mode {mode} for plotting spectrum")
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')
        _ls = []
        if _l1 is not None:
            _ls.append(_l1)
        if _l2 is not None:
            _ls.append(_l2)
        return f, ax, _ls

    def plot_pred(self, x0s, ts, ref=None, ifobs=False, idx='all', ncols=1, figsize=(6,8), title=None, fig=None):
        """
        Plot the predicted trajectories, in either data space or latent space.
        """
        if idx == 'all':
            if ifobs:
                _idx = np.arange(self._ctx._Nout, dtype=int)
            else:
                _idx = np.arange(self._ctx._Ninp, dtype=int)
        elif isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)
        _Nst = len(_idx)

        if ifobs:
            _prds = self.predict(x0s, ts, return_obs=True)[1].real
            _ylbl = 'Obs'
        else:
            _prds = self.predict(x0s, ts, return_obs=False).real
            _ylbl = 'State'
        if _prds.ndim == 2:
            _prds = np.array([_prds])
        _Nx0 = len(_prds)

        if ref is None:
            _refs, _errs = None, None
        else:
            if ifobs:
                _refs = self._ctx.encode(ref).real
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
                l1, = ax[_k].plot(ts, _prds[_i][:,_j], 'b-')
                if _refs is not None:
                    l2, = ax[_k].plot(ts, _refs[_i][:,_j], 'r--')
            ax[_k].set_ylabel(f'{_ylbl} {_j}')
        if _refs is not None:
            for _k, _j in enumerate(_idx):
                ax[_k].set_title(f'{title}, Error {_errs[_j]*100:3.2f}%')
            ax[0].legend([l1, l2], ['Prediction', 'Reference'])
        ax[-1].set_xlabel('time, s')

        return f, ax

    def plot_eigfun_2d(self, rngs, Ns, idx, mode='angle', space='full', ncols=2, figsize=(6,10), fig=None):
        """
        Plot the 2D eigenfunctions as contours.
        """
        # Regular grid
        x1s = np.linspace(rngs[0][0], rngs[0][1], Ns[0])
        x2s = np.linspace(rngs[1][0], rngs[1][1], Ns[1])
        X1, X2 = np.meshgrid(x1s, x2s)

        # Indexing
        if isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)

        # Eigenfunction
        _tmp = np.vstack([X1.reshape(-1), X2.reshape(-1)]).T
        if space == 'full':
            _fun = self.eval_eigfun(_tmp, _idx)
        elif callable(space):
            _fun = self.eval_eigfun_par(_tmp, _idx, func=space)
        else:
            # For higher dimensional states, use embedded space
            # `space` should specify the sequence of encoder to use
            _fun = self.eval_eigfun(_tmp, _idx, rng=space)

        # Plotting
        _func = complex_map[mode]
        _Np = len(_idx)
        _nr = _Np // ncols + _Np % ncols
        if fig is None:
            f, ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        else:
            f, ax = fig
        _ax = ax.flatten()
        for _i, _j in enumerate(_idx):
            _F = _fun[:,_i].reshape(Ns[1], Ns[0])
            _ax[_i].contourf(X1, X2, _func(_F), levels=20)
            _ax[_i].set_title(f'{_j}: {np.angle(self._wc[_j]):3.2e} / {self._res[_j].real:3.2e}')

        return f, ax

    def plot_eigjac_contour(self,
                            ref=None, rng=None, eig='func',
                            lam='ct', comp='ri', idx='all', shape=(),
                            contour_args={}, **kwargs):
        assert len(shape) == 2, "Shape should be of length 2 for contour plotting"

        if eig == 'func':
            _modes = self.eval_eigfunc_jac(ref, rng, **kwargs)
        elif eig == 'mode':
            _ref = None
            if ref is not None:
                _ref = self._ctx.encode(ref)
            _modes = self.eval_eigmode_jac(_ref, rng, **kwargs)
        else:
            raise ValueError(f"Unknown eig {eig} for Jacobian plotting")

        _w = self._wc if lam == 'ct' else self._wd
        ls, ms = mode_split(_w, _modes, comp=comp)
        if idx == 'all':
            idx = np.arange(len(ls), dtype=int)
        assert isinstance(idx, (list, np.ndarray)), "idx should be a list or array of integers"
        ls, ms = ls[idx], ms[idx]

        if comp == 'p':
            vdx = [0]
        else:
            vdx = []
            for _i, _c in enumerate(comp):
                if _c != 'p':
                    vdx.append(_i)
        tmp = ms[:,vdx]
        vmin, vmax = np.min(tmp), np.max(tmp)

        labels = []
        for _i, _idx in enumerate(idx):
            tmp = []
            for _j, _c in enumerate(comp):
                if _j == 0:
                    _prf = f'Mode {_idx}: '
                else:
                    _prf = ''
                lbl = f"{_prf}{LMAP[_c]} / {ls[_i][_j]:3.2e}"
                tmp.append(lbl)
            labels += tmp

        contour_args.update(label=labels, vmin=vmin, vmax=vmax)
        f, ax = plot_contour(ms.reshape(len(labels), *shape), **contour_args)

        return (f, ax), (ls, ms)

    def plot_vec_line(self, idx, which='func', modes=['angle'], ncols=1, figsize=(6,10)):
        """
        Plot slices of eigenfunctions as vectors.
        """
        # Indexing
        if isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)

        # Vectors
        if which == 'func':
            _vec = self._vl
        elif which == 'mode':
            _vec = self._vr
        else:
            raise ValueError(f"Unknown quantity to plot: {which}")

        # Plotting
        _fs = [complex_map[_m] for _m in modes]
        _Np = len(_idx)
        _nr = _Np // ncols + _Np % ncols
        f, ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        _ax = ax.flatten()
        for _i in _idx:
            for _f in _fs:
                _ax[_i].plot(_f(_vec[_i]))
            _ax[_i].set_title(f'{_i}: {np.angle(self._wc[_i]):3.2e} / {self._res[_i].real:3.2e}')

        return f, ax
