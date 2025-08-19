import logging
import matplotlib.pyplot as plt
import numpy as np

from dymad.numerics import check_orthogonality, complex_grid, complex_map, disc2cont, scaled_eig, truncate_sequence
from dymad.sako.interface import SAInterface
from dymad.sako.rals import estimate_pseudospectrum, RALowRank
from dymad.sako.sako import SAKO

logger = logging.getLogger(__name__)

class SpectralAnalysis:
    """
    The base class for Spectral Analysis based on Koopman operator theory.

    The formulation is based on the following convention:
    Psi_0 A = Psi_1
    where A is the finite-dimensional approximation of Koopman operator,
    Psi's are data matrices with each row containing one time step.

    Args:
        order: Thresholds to trim the system.  Algorithm-dependent.
        dt: Time step size.
    """
    def __init__(self, order=0.98, dt=1.0, reps=1e-10):
        self._order = order
        self._dt = dt
        self._reset()

        self._ctx = SAInterface()

        self._solve_eigs()
        logger.info(f"Orthonormality violation: {check_orthogonality(self._vl, self._vr)[1]:4.3e}")
        self._proc_eigs()
        self._sako = SAKO(self._ctx._P0, self._ctx._P1, None, reps=reps)
        self._rals = RALowRank(self._vr, np.diag(self._wc.conj()), self._vl, dt=self._dt)

    def predict(self, x0, tseries, return_obs=False):
        """
        Make time-domain prediction.

        Args:
            x0: Initial states
            tseries: Time series at which to evaluate the solutions.
            return_obs: If return observables over time as well
        """
        _ts = tseries - tseries[0]
        _p0 = self._ctx.encode(x0).reshape(-1)
        _b  = self._proj.dot(_p0)
        _ls = np.exp(self._wc.conj().reshape(-1,1) * _ts)
        _pt = (self._vl*_b).dot(_ls).T
        _xt = self._ctx.decode(_pt)
        if return_obs:
            return _xt, _pt
        return _xt

    def mapto_obs(self, X):
        """
        Map new trajectory data to the observer space.
        """
        return self._ctx.encode(X)

    def apply_obs(self, fobs):
        """
        Apply a generic observable to the data.

        Args:
            fobs: Observable function.  Assuming 2D array input with each row as one step.
                The output should be a 1D array, whose ith entry corresponds to the ith step.
        """
        return self._ctx.apply_obs(fobs).reshape(-1)

    def eval_eigfun(self, X, idx):
        """
        Evaluate the eigenfunctions at given locations
        """
        _P = self.mapto_obs(X)
        return _P.dot(self._vr[:,idx])

    def eval_eigfun_em(self, X, idx, rng):
        """
        Evaluate the eigenfunctions at given locations in embedded space
        """
        _P = self._ctx.encode(X, rng)
        return _P.dot(self._vr[:,idx])

    def eval_eigfun_par(self, par, idx, func):
        """
        Evaluate the eigenfunctions at given parametrization
        """
        _P = self.mapto_obs(func(par))
        return _P.dot(self._vr[:,idx])

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
            _f1 = self.eval_eigfun(_eps*_vr[:,_j].reshape(1,-1), _i)
            _f0 = self.eval_eigfun(np.zeros((1,_N)), _i)   # Supposed to be 0
            _vw =  (_f1-_f0) / _eps
            _sgn.append(np.sign(_vw.real))
        _sgn = np.array(_sgn).reshape(-1)
        logger.info(f"Flipping: {_sgn}")
        _T = _vr * _sgn
        # The mappings
        self.mapto_cnj = lambda X, I=_idx, W=_T: self.eval_eigfun(X, I).dot(W.T)
        self.mapto_nrm = lambda X, I=_idx, S=_sgn: self.eval_eigfun(X, I) * S

    def filter_spectrum(self, sako, order='full'):
        """
        Apply SAKO to the identified eigenpairs to compute the corresponding residuals
        """
        self._res_full = sako.estimate_residual(self._wd_full, self._vr_full)

        # Full set
        _msk = np.argsort(self._res_full)
        self._res_full = self._res_full[_msk]
        self._wd_full = self._wd_full[_msk]
        self._vl_full = self._vl_full[:,_msk]
        self._vr_full = self._vr_full[:,_msk]

        # Truncated set
        idx = truncate_sequence(self._res_full, order)
        jdx = []  # Ensure all conjugates appear simultaneously
        for _i in idx:
            if _i not in jdx:
                jdx.append(_i)
                _w = self._wd_full[_i]
                _j = np.argmin(np.abs(self._wd_full-_w.conj()))
                if _j not in idx:
                    logger.info(f"Adding missing conjugate {_j}: {self._wd_full[_j]:5.4e}")
                if _j not in jdx:
                    jdx.append(_j)
        self._res = self._res_full[jdx]
        self._wd = self._wd_full[jdx]
        self._vl = self._vl_full[:,jdx]
        self._vr = self._vr_full[:,jdx]
        self._Nrank = len(jdx)

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
        logger.info(f"Estimating PS: {self._type} Mode:{mode} Method:{method}")
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
                _u = (self._vl*_b).dot(_ls).reshape(-1)
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
            _Vr, _B = weights
            _At = _B.dot(_Vr)
            self._wd, _vl, _vr = scaled_eig(_At)
            self._vl = _B.conj().T.dot(_vl) / self._wd.conj().reshape(1,-1)
            self._vr = _Vr.dot(_vr)
        elif len(weights) == 1:
            _W = weights[0]
            self._wd, self._vl, self._vr = scaled_eig(_W)

        # For data member consistency
        self._wd_full = self._wd
        self._vl_full = self._vl
        self._vr_full = self._vr

        # Unsure yet whether GEP is needed:

        # _M0 = self._sako._M0
        # _M1 = self._sako._M1
        # ## Generalized eigenvalue problem
        # self._wd_full, _vl_full, self._vr_full = scaled_eig(_M1, _M0)
        # self._vl_full = _M0.conj().T.dot(_vl_full)

    def _proc_eigs(self):
        """
        Computes several data members for subsequent processing.
        """
        self._wc_full = disc2cont(self._wd_full, self._dt)
        self._wc = disc2cont(self._wd, self._dt)
        # self._proj = np.linalg.solve(self._vl.conj().T.dot(self._vl), self._vl.conj().T)
        self._proj = self._vr.conj().T   # Mathemetically correct, but numerically inaccurate.

    def plot_eigs(self, fig=None, plot_full='bo', plot_filt='r^', mode='disc'):
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
        return f, ax, [_l1, _l2]

    def plot_pred_x(self, x0s, ts, ref=None, idx='all', figsize=(6,8)):
        if idx == 'all':
            _idx = np.arange(self._ctx._Ninp, dtype=int)
        elif isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)
        _Nst = len(_idx)
        _Nx0 = len(x0s)

        f, ax = plt.subplots(nrows=_Nst, sharex=True, figsize=figsize)
        e0 = 0.0
        for _i in range(_Nx0):
            _pred = self.predict(x0s[_i], ts, return_obs=False).real
            # e0 += np.linalg.norm(_pred-ref[_i]) / np.linalg.norm(ref[_i])
            e0 += np.linalg.norm(_pred-ref[_i], axis=0) / np.sqrt(len(_pred)) / (np.max(ref[_i], axis=0)-np.min(ref[_i], axis=0))
            for _j in _idx:
                l1, = ax[_j].plot(ts, _pred[:,_j], 'b-')
                l2, = ax[_j].plot(ts, ref[_i][:,_j], 'r--')
                ax[_j].set_ylabel(f'State {_j}')
        ax[0].legend([l1, l2], ['Prediction', 'Reference'])
        for _j in range(_Nst):
            ax[_j].set_title(f'{self._type}, Error {e0[_j]/_Nx0*100:3.2f}%')
        ax[-1].set_xlabel('time, s')

        return f, ax

    def plot_pred_psi(self, x0s, ts, ref=None, idx='all', ncols=1, figsize=(6,8)):
        if isinstance(idx, str) and idx == 'all':
            _idx = np.arange(self._ctx._Nout)
        elif isinstance(idx, int):
            _idx = np.arange(idx)
        else:
            _idx = np.array(idx)
        _Nst = len(_idx)
        _Nx0 = len(x0s)

        _nr = _Nst // ncols + _Nst % ncols
        f, _ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        ax = _ax.flatten()
        e0 = 0.0
        for _i in range(_Nx0):
            _pred = self.predict(x0s[_i], ts, return_obs=True)[1].real
            _ref = self.mapto_obs(ref[_i]).real
            e0 += np.linalg.norm(_pred-_ref) / np.linalg.norm(_ref)
            for _j in range(_Nst):
                _k = _idx[_j]
                l1, = ax[_j].plot(ts, _pred[:,_k], 'b-')
                l2, = ax[_j].plot(ts, _ref[:,_k], 'r--')
                ax[_j].set_ylabel(f'Observable {_k}')
        ax[0].legend([l1, l2], ['Prediction', 'Reference'])
        ax[0].set_title(f'{self._type}, Error {e0/_Nx0*100:3.2f}%')
        ax[-1].set_xlabel('time, s')

        return f, ax

    def plot_eigfun_2d(self, rngs, Ns, idx, mode='angle', space='full', ncols=2, figsize=(6,10)):
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
            _fun = self.eval_eigfun_par(_tmp, _idx, space)
        else:
            # For higher dimensional states, use embedded space
            # `space` should specify the sequence of encoder to use
            _fun = self.eval_eigfun_em(_tmp, _idx, space)

        # Plotting
        _func = complex_map[mode]
        _Np = len(_idx)
        _nr = _Np // ncols + _Np % ncols
        f, ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        _ax = ax.flatten()
        for _i in _idx:
            _F = _fun[:,_i].reshape(Ns[1], Ns[0])
            _ax[_i].contourf(X1, X2, _func(_F), levels=20)
            _ax[_i].set_title(f'{_i}: {np.angle(self._wc[_i]):3.2e} / {self._res[_i].real:3.2e}')

        return f, ax

    def plot_vec_line(self, idx, which='func', modes=['angle'], ncols=1, figsize=(6,10)):
        # Indexing
        if isinstance(idx, int):
            _idx = np.arange(idx, dtype=int)
        else:
            _idx = np.array(idx)

        # Vectors
        if which == 'func':
            _vec = self._vr
        elif which == 'mode':
            _vec = self._vl
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
