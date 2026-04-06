import logging

import numpy as np
import scipy.linalg as spl

from dymad.numerics import generate_coef, scaled_eig

logger = logging.getLogger(__name__)


def clip_eig(M, tol=1e-13):
    """
    Clip small eigenvalues in the matrix.
    """
    w, v = spl.eigh(M)
    msk = w < tol
    if np.any(msk):
        logger.info(f"clip_eig: {np.sum(msk)} small eigenvalues found, max {np.max(-w[msk]):4.3e}")
        w[msk] = tol
    return v @ np.diag(w) @ v.conj().T


class SAKO:
    """
    Spectral Analysis for Koopman Operators

    The implementation of the spectral analysis algorithms for Koopman operators
    from Colbrook et al.

    The formulation is based on the following convention:
    Psi_0 A = Psi_1
    where A is the finite-dimensional approximation of Koopman operator,
    Psi's are data matrices with each row containing one time step.

    Args:
        P0 (np.ndarray): Psi_0
        P1 (np.ndarray): Psi_1
        W (np.ndarray): Weights for the inner product, default is identity matrix
        reps (float): Threshold for the imaginary part of the residual, default is 1e-10
        etol (float): Tolerance for clipping small eigenvalues in the inner product matrices, default is 1e-13
    """

    def __init__(
        self,
        P0: np.ndarray,
        P1: np.ndarray,
        W: np.ndarray = None,
        reps: float = 1e-10,
        etol: float = 1e-13,
    ):
        self._P0 = P0
        self._P1 = P1
        self._W = (
            np.array(W)
            if W is not None
            else np.ones(
                len(P0),
            )
        )
        self._reps = reps

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

        if etol is not None:
            self._M00 = clip_eig(self._M00, tol=etol)
            self._M11 = clip_eig(self._M11, tol=etol)

        self._M0 = self._M00
        self._M1 = self._M01

    def solve_eig(self):
        """
        Solve the generalized eigenvalue problem to obtain the eigenpairs.
        """
        _wd_full, _vl_full, _vr_full = scaled_eig(self._M01, self._M00)
        _vl_full = self._M00.conj().T.dot(_vl_full)
        return _wd_full, _vl_full, _vr_full

    def estimate_residual(self, ls: np.ndarray, gs: np.ndarray) -> np.ndarray:
        """
        Wrapper of _residual_G and _residual_r for a batch of eigenpairs

        Args:
            ls (np.ndarray): Array of eigenvalues
            gs (np.ndarray): Array of right eigenvectors, column wise
        """
        _N = len(ls)
        _r = np.zeros(
            _N,
        )
        for _i in range(_N):
            _r[_i] = self._residual_r(gs[:, _i], self._residual_G(ls[_i]))
        return _r

    def estimate_measure(
        self, fobs: np.ndarray, order: int, eps: float, thetas: int | np.ndarray = 101
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the spectral measure associated with a given scalar observable using
        a kernel of rational function.

        Args:
            fobs (np.ndarray): Observable function evaluated at X0.  Assuming 1D array
            order (int): Order of the rational kernel
            eps (float): Smoothing parameter of the rational kernel
            thetas (Union[int, np.ndarray]): The thetas at which to evaluate the measure; if int, a uniform grid is generated.
        """
        if isinstance(thetas, int):
            _th = np.linspace(-np.pi, np.pi, thetas)
        else:
            _th = np.array(thetas).reshape(-1)
        _N = len(_th)

        # Generalized Schur decomposition
        _S, _T, _Q, _Z = spl.qz(self._M01, self._M00, output="complex")

        # Projection of observable
        _b = (self._P0.conj().T).dot(self._W * fobs.reshape(-1))
        ## Enforcing complex128 for compatibility with early version of Scipy
        _a = spl.solve(np.array(self._M00, dtype=np.complex128), _b, assume_a="her")

        # Working arrays
        _v1 = _T.dot(_Z.conj().T.dot(_a))
        _v2 = _T.conj().T.dot(_Q.conj().T.dot(_a))
        _v3 = _S.conj().T.dot(_Q.conj().T.dot(_a))

        # Kernel coefficients
        _, _zs, _cs, _ds = generate_coef(order, eps)

        # Loop over theta
        _ex = np.exp(1j * _th)
        _nu = np.zeros(
            _N,
        )
        for _i in range(_N):
            for _j in range(order):
                _l = _ex[_i] * (1 + eps * _zs[_j])
                _I = spl.solve_triangular(_S - _l * _T, _v1, lower=False)
                _nu[_i] += np.real(
                    _cs[_j] * _l.conj() * _I.conj().dot(_v2) + _ds[_j] * _v3.conj().dot(_I)
                )
        return _th, _nu / (-2 * np.pi)

    def _residual_G(self, l: float | complex) -> np.ndarray:
        """
        The core math device in SAKO - Part I

        Args:
            l (Union[float, complex]): Eigenvalue, or eigenvalue-like
        """
        _G = self._M11 - l * self._M10 - np.conj(l) * self._M01 + np.abs(l) ** 2 * self._M00
        # Enforce the hermitianity of the G matrix
        return 0.5 * (_G + _G.conj().T)

    def _residual_r(self, g: np.ndarray, G: np.ndarray) -> float:
        """
        The core math device in SAKO - Part II

        Args:
            g (np.ndarray): Eigenvector, or eigenvector-like
            G (np.ndarray): Output from _residual_G
        """
        _r = g.conj().T.dot(G).dot(g) / g.conj().T.dot(self._M00).dot(g)
        if np.abs(_r.imag) > self._reps:
            logger.info(
                f"SAKO Warning: Imaginary part of residual {_r.imag:5.4e} exceeds "
                f"threshold {self._reps:3.2e}; real part {_r.real:5.4e}"
            )
        return np.sqrt(np.abs(_r))

    def _ps_point(
        self, z: float | complex, return_vec: bool = False
    ) -> float | tuple[float, np.ndarray]:
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

        Args:
            z (Union[float, complex]): The point at which to evaluate the resolvent norm
            return_vec (bool): If return singular vector
        """
        _G = self._residual_G(z)
        if return_vec:
            _e, _v = spl.eigh(_G, self._M00, subset_by_index=[0, 0], driver="gvx")
        else:
            _e = spl.eigh(_G, self._M00, subset_by_index=[0, 0], driver="gvx", eigvals_only=True)
            _v = None
        if not _e >= 0:
            logger.info(f"SAKO Warning: Non-positive norm {_e[0]:4.3e} found for {z:4.3e}")
            _e = np.array([1e-16])
        if return_vec:
            return 1 / np.sqrt(_e), _v.reshape(-1)
        return 1 / np.sqrt(_e)
