"""
@package base

The basic interfaces of DMD class and the implementation
of some algorithm-independent functionalities.

@author Dr. Daning Huang
@date 06/18/23
"""
import matplotlib.pyplot as plt
import numpy as np
from .pipeline.pipeline import Pipeline
from .rals import estimatePSpec, RALowRank
from .utils import cpGrid, cpMap, disc2cont, chkOrth, scaledEig, truncateSeq

class DMDBase:
    """
    The base class for DMD.  It processes the inputs and makes predictions
    assuming that the eigensystem is available.

    The formulation is based on the following convention:
    Psi_0 A = Psi_1
    where A is the finite-dimensional approximation of Koopman operator,
    Psi's are data matrices with each row containing one time step.
    """
    def __init__(self, order=0.98, dt=1.0, verbose=True):
        """
        Initialize the instance.

        @param trans: List of tuples defining the Transform objects.
        @param order: Thresholds to trim the system.  Algorithm-dependent.
        @param dt: Time step size.
        @param verbose: If print out processing information.
        """
        self._type = 'none'
        self._order = order
        self._dt = dt
        self._verbose = verbose
        self._reset()

    def fit(self, Xs, trans=[], **kwargs):
        """
        The wrapper function to process multiple trajectories and perform the DMD prediction.
        """
        print(f"{self._type}")
        _fil = " "*3
        print(f"{_fil} Started")
        self._reset()
        if trans is None:
            self._data = Pipeline(Xs, [('iden',{})])
        elif isinstance(trans, list):
            self._data = Pipeline(Xs, trans)
        elif isinstance(trans, Pipeline):
            self._data = trans
            self._data.update(Xs)
        else:
            raise ValueError(f"Unknown data Transform object {trans}")
        self._batch_obs()
        self._solve_ls()
        print(f"{_fil} Orthonormality violation: {chkOrth(self._vl, self._vr)[1]:4.3e}")
        self._proc_eigs()
        self._rals = RALowRank(self._vr, np.diag(self._wc.conj()), self._vl, dt=self._dt)
        self._is_fitted = True
        print(f"{_fil} Done")

    def predict(self, x0, tseries, return_obs=False):
        """
        Make time-domain prediction.

        @param x0: Initial states
        @param tseries: Time series at which to evaluate the solutions.
        @param return_obs: If return observables over time as well
        """
        if not self._is_fitted:
            raise ValueError("DMD problem not solved yet!")

        _ts = tseries - tseries[0]
        _p0 = self._data.EN(x0).reshape(-1)
        _b  = self._proj.dot(_p0)
        _ls = np.exp(self._wc.conj().reshape(-1,1) * _ts)
        _pt = (self._vl*_b).dot(_ls).T
        _xt = self._data.DE(_pt)
        if return_obs:
            return _xt, _pt
        return _xt

    def predict_wp(self, x0, tseries, return_obs=False):
        """
        Make time-domain prediction, with autoencoder every step.

        @param x0: Initial states
        @param tseries: Time series at which to evaluate the solutions.
        @param return_obs: If return observables over time as well
        """
        if not self._is_fitted:
            raise ValueError("DMD problem not solved yet!")

        _Nt = len(tseries)
        _p0 = self._data.EN(x0).reshape(-1)
        _xt, _pt = [x0], [_p0]
        for _i in range(_Nt-1):
            _dt = tseries[_i+1] - tseries[_i]
            _b  = self._proj.dot(_pt[_i])
            _ls = np.exp(self._wc.conj() * _dt)
            _p1 = (self._vl*_b).dot(_ls).reshape(1,-1)
            _x1 = self._data.DE(_p1)
            _p1 = self._data.EN(_x1).reshape(-1)
            _xt.append(_x1)
            _pt.append(_p1)
        _xt = np.vstack(_xt)
        _pt = np.vstack(_pt)
        if return_obs:
            return _xt, _pt
        return _xt

    def mapto_obs(self, X):
        """
        Map new trajectory data to the observer space.
        """
        return self._data.EN(X)

    def apply_obs(self, fobs):
        """
        Apply a generic observable to the data.
        @param fobs: Observable function.  Assuming 2D array input with each row as one step.
                     The output should be a 1D array, whose ith entry corresponds to the ith step.
        """
        return self._data.apply_obs(fobs).reshape(-1)

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
        _P = self._data.EN(X, rng)
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
        _wl, _vl, _vr = scaledEig(J)
        _N = len(J)
        assert len(_wl) <= len(self._wc)   # Insufficient Koopman dimensions
        _idx = []
        _sgn = []
        _eps = 1e-6
        print("Computing conjugacy map:")
        for _j, _w in enumerate(_wl):
            # Identify the principal eigenfunction
            _d = np.abs(self._wc-_w)
            _i = np.argmin(_d)
            print("    EV: Jacobian {0:5.4e}, Koopman {1:5.4e}, diff {2:5.4e}".format(
                _w, self._wc[_i], np.abs(_d[_i]/self._wc[_i])
            ))
            _idx.append(_i)
            # Check the sign by evaluating along w_i, and v_i^H w_i = +/- 1
            _f1 = self.eval_eigfun(_eps*_vr[:,_j].reshape(1,-1), _i)
            _f0 = self.eval_eigfun(np.zeros((1,_N)), _i)   # Supposed to be 0
            print(_f0)
            _vw =  (_f1-_f0) / _eps
            _sgn.append(np.sign(_vw.real))
        _sgn = np.array(_sgn).reshape(-1)
        print(f"    Flipping: {_sgn}")
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
        idx = truncateSeq(self._res_full, order)
        jdx = []  # Ensure all conjugates appear simultaneously
        for _i in idx:
            if _i not in jdx:
                jdx.append(_i)
                _w = self._wd_full[_i]
                _j = np.argmin(np.abs(self._wd_full-_w.conj()))
                if _j not in idx:
                    print(f"    Adding missing conjugate {_j}: {self._wd_full[_j]:5.4e}")
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

        @param grid: Mode disc: points on discrete-time plane (Re, Im)
                     Mode cont: points on continuous-time plane (zeta, omega)
        @param return_vec: If return I/O modes
        @param mode: 'cont' or 'disc'
        @param verbose: Whether to print info.
        """
        print(f"{self._type} Mode:{mode} Method:{method}")
        _g = cpGrid(grid)
        res = estimatePSpec(_g, self.resolvent_analysis, verbose=self._verbose, return_vec=return_vec, \
            **{'mode':mode, 'method':method})
        return _g, res

    def resolvent_analysis(self, z, return_vec, mode, method):
        """
        Perform resolvent analysis of the DMD operator.

        @param method: 'standard' - The projected approach where I/O modes are all in DMD mode space,
                       which is true for a low-rank DMD operator.
        """
        if method.lower() != 'standard':
            raise ValueError(f"    Method {method} unknown for resolvent analysis in {self._type}")
        return self._rals(z, return_vec, mode)

    def _reset(self):
        self._is_fitted = False
        # Dimensions
        self._Nrank = None
        # Data
        self._P0 = np.array([])    # Psi_0
        self._P1 = np.array([])    # Psi_1 = Psi_0 A
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

    def _batch_obs(self):
        """
        Convert a list of trajectories to observables, and align to the Psi_0 and Psi_1 format.
        """
        self._P0, self._P1 = self._data.get_data()

    def _solve_ls(self):
        """
        Solve the least-squares problem and produce the eigensystem of the operator A.

        This function should produce the discrete eigensystem: wd, vl, vr.
        """
        raise NotImplementedError("_solve_ls needs to be defined in derived classes.")

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
            _idx = np.arange(self._data._Ninp, dtype=int)
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
            _idx = np.arange(self._data._Nout)
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
        _func = cpMap[mode]
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
        _fs = [cpMap[_m] for _m in modes]
        _Np = len(_idx)
        _nr = _Np // ncols + _Np % ncols
        f, ax = plt.subplots(nrows=_nr, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        _ax = ax.flatten()
        for _i in _idx:
            for _f in _fs:
                _ax[_i].plot(_f(_vec[_i]))
            _ax[_i].set_title(f'{_i}: {np.angle(self._wc[_i]):3.2e} / {self._res[_i].real:3.2e}')

        return f, ax