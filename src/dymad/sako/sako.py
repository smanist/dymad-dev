"""
@package sako

The implementation of the spectral analysis algorithms for Koopman operators
from Colbrook et al.

@author Dr. Daning Huang
@date 10/27/22
"""
import numpy as np
import scipy.linalg as spl
from .utils import disc2cont, genCoef, truncatedSVD, cmpMat

class SAKO:
    """
    Spectral Analysis for Koopman Operators

    The formulation is based on the following convention:
    Psi_0 A = Psi_1
    where A is the finite-dimensional approximation of Koopman operator,
    Psi's are data matrices with each row containing one time step.
    """
    def __init__(self, P0, P1, W, dt=1.0, reps=1e-10, order=None, verbose=True):
        """
        Initialize the instance.

        @param P0: Psi_0
        @param P1: Psi_1
        @param W: Weights for integral in inner products
        @param dt: Time step size.
        @param reps: Threshold for imaginary part of residual.  Default 1e-10
        @param order: Generic parameter for order truncation.  Unused in base class.
        @param verbose: If print out processing information.
        """
        self._P0 = P0
        self._P1 = P1
        _, self._Nobs = self._P0.shape
        self._W  = np.array(W)
        self._dt = dt
        self._reps = reps
        self._order = order
        self._verbose = verbose

        # The inner product matrices
        # M_ij = Psi_i^H W Psi_j
        #
        # Directly computing M_ij by definition may result in minor violations in hermitianity of
        # the matrices, i.e.,
        #       M_ii^H = M_ii and M_ij^H = M_ji
        # We do an averaging to enforce the hermitianity
        # self._M00 = (self._P0.conj().T * _W).dot(self._P0)
        # self._M01 = (self._P0.conj().T * _W).dot(self._P1)
        # self._M10 = (self._P1.conj().T * _W).dot(self._P0)
        # self._M11 = (self._P1.conj().T * _W).dot(self._P1)
        _M00 = (self._P0.conj().T * self._W).dot(self._P0)
        self._M00 = 0.5 * (_M00 + _M00.conj().T)
        _M01 = (self._P0.conj().T * self._W).dot(self._P1)
        _M10 = (self._P1.conj().T * self._W).dot(self._P0)
        self._M01 = 0.5 * (_M01 + _M10.conj().T)
        self._M10 = self._M01.conj().T
        _M11 = (self._P1.conj().T * self._W).dot(self._P1)
        self._M11 = 0.5 * (_M11 + _M11.conj().T)

        self._M0 = self._M00
        self._M1 = self._M01
 
    def estimate_residual(self, ls, gs):
        """
        Wrapper of _residual_G and _residual_r for a batch of eigenpairs
        @param ls: Array of eigenvalues
        @param gs: Array of right eigenvectors, column wise
        """
        _N = len(ls)
        _r = np.zeros(_N,)
        for _i in range(_N):
            _r[_i] = self._residual_r(gs[:,_i], self._residual_G(ls[_i]))
        return _r

    def estimate_measure(self, fobs, order, eps, thetas=101):
        """
        Estimate the spectral measure associated with a given scalar observable using
        a kernel of rational function.

        @param fobs: Observable function evaluated at X0.  Assuming 1D array
        @param order: Order of the rational kernel
        @param eps: Smoothing parameter of the rational kernel
        @param thetas: The thetas at which to evaluate the measure; if int, a uniform grid is generated.
        """
        if isinstance(thetas, int):
            _th = np.linspace(-np.pi, np.pi, thetas)
        else:
            _th = np.array(thetas).reshape(-1)
        _N = len(_th)

        # Generalized Schur decomposition
        _S, _T, _Q, _Z = spl.qz(self._M01, self._M00, output='complex')

        # Projection of observable
        _b = (self._P0.conj().T).dot(self._W*fobs.reshape(-1))
        ## Enforcing complex128 for compatibility with early version of Scipy
        _a = spl.solve(np.array(self._M00, dtype=np.complex128), _b, assume_a='her')

        # Working arrays
        _v1 = _T.dot(_Z.conj().T.dot(_a))
        _v2 = _T.conj().T.dot(_Q.conj().T.dot(_a))
        _v3 = _S.conj().T.dot(_Q.conj().T.dot(_a))

        # Kernel coefficients
        _, _zs, _cs, _ds = genCoef(order, eps)

        # Loop over theta
        _ex = np.exp(1j * _th)
        _nu = np.zeros(_N,)
        for _i in range(_N):
            for _j in range(order):
                _l = _ex[_i] * (1+eps*_zs[_j])
                _I = spl.solve_triangular(_S - _l*_T, _v1, lower=False)
                _nu[_i] += np.real(
                    _cs[_j] * _l.conj() * _I.conj().dot(_v2) + \
                    _ds[_j] * _v3.conj().dot(_I))
        return _th, _nu / (-2*np.pi)

    def _residual_G(self, l):
        """
        The core math device in SAKO - Part I
        @param l: Eigenvalue, or eigenvalue-like
        """
        _G = self._M11 - l*self._M10 - np.conj(l)*self._M01 + np.abs(l)**2 * self._M00
        # Enforce the hermitianity of the G matrix
        return 0.5 * (_G + _G.conj().T)

    def _residual_r(self, g, G):
        """
        The core math device in SAKO - Part II
        @param g: Eigenvector, or eigenvector-like
        @param G: Output from _residual_G
        """
        _r = g.conj().T.dot(G).dot(g) / g.conj().T.dot(self._M00).dot(g)
        if np.abs(_r.imag) > self._reps:
            print(f"    SAKO Warning: Imaginary part of residual {_r.imag:5.4e} exceeds " \
                f"threshold {self._reps:3.2e}; real part {_r.real:5.4e}")
        return np.sqrt(np.abs(_r))

    def _ps_point(self, z, return_vec):
        """
        Compute the resolvent norm at given complex point for pseudospectrum.
        The matrices involved are hermitian, so a GEP is solved instead of a
        generalized SVD problem.

        In the context of resolvent analysis, and given the ResDMD formulation,
        the norm is the gain, the singular vector is the output mode, and one can
        further compute the input mode by applying the learned Koopman operator.

        However, note that the resolvent analysis is applied to a discrete-time system
        and the resolvent norms and modes approximate those of a continuous-time system
        ONLY when the time step size is small.

        If the purpose is to perform modal analysis of an inherently cont-time problem,
        one should convert the truncated disc-time Koopman operator to cont-time and then
        perform the resolvent analysis.  This should be done with DMD.

        @param z: The point at which to evaluate the resolvent norm
        @param return_vec: If return singular vector
        """
        _G = self._residual_G(z)
        if return_vec:
            _e, _v = spl.eigh(_G, self._M00, subset_by_index=[0, 0], driver='gvx')
        else:
            _e = spl.eigh(_G, self._M00, subset_by_index=[0, 0], driver='gvx', eigvals_only=True)
            _v = None
        if not _e >= 0:
            print(f"    SAKO Warning: Non-positive norm {_e[0]:4.3e} found for {z:4.3e}")
            _e = np.array([1e-16])
        if return_vec:
            return 1/np.sqrt(_e), _v.reshape(-1)
        return 1/np.sqrt(_e)
