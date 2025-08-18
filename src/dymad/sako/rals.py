import logging
import numpy as np
import scipy.linalg as spl
from typing import Callable

logger = logging.getLogger(__name__)

def resolvent_analysis(
        z: np.ndarray, return_vec: bool = False,
        A: np.ndarray = None, B: np.ndarray = None, U: np.ndarray = None,
        ord: int = 1):
    """
    Standalone, naive implementation of resolvent analysis, mainly for
    sanity check and suitable for small-scale problems.
    Uses SVD to compute resolvent of `A` at a given point `z`,
    and return the first `ord` modes and gains.

    Args:
        z (np.ndarray): Point at which to compute resolvent
        return_vec (bool): If return I/O modes
        A (np.ndarray): The linear operator
        B (np.ndarray): The input matrix, only used if U is None
        U (np.ndarray): The constraining subspace
        ord (int): Number of modes and gains to return
    """
    if U is None:
        # No constraints on input/output space
        _N  = len(A)
        if B is None:
            _B = np.eye(_N)
        else:
            _B = np.array(B)
        _zI = np.diag(z*np.ones(_N,))
        _H = np.linalg.solve(_zI-A, _B)
        _U, _s, _Vh = np.linalg.svd(_H)
        _V = _Vh.conj().T
    else:
        # Constraints on both input & output spaces
        _Q, _R = np.linalg.qr(U)
        _Ar = spl.pinv(U).dot(A).dot(U)
        _Ir = np.diag(np.ones(U.shape[1],))
        _tmp = spl.solve(_R.dot(z*_Ir-_Ar).T, _R.T).T
        _U, _s, _Vh = np.linalg.svd(_tmp, full_matrices=False)
        _U = _Q.dot(_U)
        _V = _Q.dot(_Vh.conj().T)

        # # A computationally more expensive version
        # _zI = np.diag(z*np.ones(len(A),))
        # _H  = _Q.conj().T.dot(np.linalg.solve(_zI-A, _Q))
        # _U, _s, _Vh = np.linalg.svd(_H, full_matrices=False)
        # _U = _Q.dot(_U)
        # _V = _Q.dot(_Vh.conj().T)

    if return_vec:
        return _s[:ord], np.squeeze(_U[:,:ord]), np.squeeze(_V[:,:ord])
    return _s[:ord]

def estimate_pseudospectrum(grid: np.ndarray, estimator: Callable, return_vec: bool = False, **kwargs):
    """
    Estimate pseudospectrum over a grid.

    Args:
        grid (np.ndarray): List of points on complex plane
        estimator (Callable): Function to evaluate gain and I/O modes at a point
                              Args: complex point, return_vec, **kwargs
        verbose (bool): Whether to print info.
        return_vec (bool): If return I/O modes
        kwargs: Args for estimator
    """
    _Ng = len(grid)
    kwargs.update(return_vec=return_vec)

    _es, _us, _vs = [], [], []
    for _i, _z in enumerate(grid):
        logger.info(f"Estimating pseudospectrum at point {_i+1}/{_Ng}: {_z}")
        _res = estimator(_z, **kwargs)
        if return_vec:
            _es.append(_res[0])
            _us.append(_res[1])
            _vs.append(_res[2])
        else:
            _es.append(_res)
    _es = np.array(_es).reshape(-1)

    _m = np.isfinite(_es)
    if not np.all(_m):
        logger.info("Found inf gain(s) in pseudospectrum")
        _es[~_m] = np.max(_es)

    if return_vec:
        _us = np.array(_us).T
        _vs = np.array(_vs).T
        return _es, _us, _vs
    return _es

class RALowRank:
    """
    Resolvent analysis for low-rank linear systems

    For continuous-time system,
        \dot{x} = Ax+Bu
    Resolvent operator H(s) = (sI-A)^{-1} B

    Following cases are considered:
    (1) Low-dimensional full-order systems
    (1.1) Regular resolvent
        argmax || H(s)u || / ||u||, H(s) = (sI-A)^{-1} B

    (1.2) Constrained resolvent, where input & output are in subspace U
        argmax || UH_r(s)a || / ||Ua||, H_r(s) = (sI-PAU)^{-1}
        where PU=I_r.  B is ignored.
        An alternative view for constrained resolvent is to think as
        argmax || YH(s)Ua || / ||Ua||
            = || Q^HH(s)Qg || / ||g||, U=QR, Ra=g
        where Y=QQ^H is the projector onto U.

    (2) High-dimensional low-rank systems, where
        A = UTV^H, with V^H U = I_r
        The formulation is essentially the constrained resolvent,
        with subspace U,
        argmax || UH_r(s)a || / ||Ua||, H_r(s) = (sI-T)^{-1}
            = || RH_r(s)R^{-1}g || / ||g||, U=QR, Ra=g

    In mode='disc', discrete-time matrix will be used
    Ad = U exp(T*dt) V^H

    Args:
        U (np.ndarray): Left orthogonal factor of A
        T (np.ndarray): Eigen block of A
        V (np.ndarray): Right orthogonal factor of A
        dt (float): Time step for discrete-time systems, default is 1.0
    """
    def __init__(self, U: np.ndarray, T: np.ndarray, V: np.ndarray, dt: float = 1.0):
        self._U = U
        self._T = T
        self._V = V
        self._dt = dt
        self._Ndim, self._Nrnk = U.shape

        self._Td = spl.expm(self._T*self._dt)
        self._Q, self._R = np.linalg.qr(self._U)
        self._Ir = np.eye(self._Nrnk)

    def __call__(self, z: np.ndarray, return_vec: bool, mode: str = 'cont'):
        """
        Compute the resolvent norm at given complex point for pseudospectrum.
        Only the dominant one is returned.

        Args:
            z (np.ndarray): The point at which to evaluate the resolvent norm
            return_vec (bool): If return resolvent modes
            mode (str): 'cont' for c-t, 'disc' for d-t
        """
        _L = self._T if mode == 'cont' else self._Td
        _tmp = self._R.dot(z*self._Ir-_L).T
        try:
            _M = spl.solve(_tmp, self._R.T).T
        except np.linalg.LinAlgError:
            logger.info("RALowRank: Singular matrix encountered; using PInv")
            _M = spl.pinv(_tmp).dot(self._R.T).T
        # Since Nrnk is expected to be low, we just compute the entire set of SVD
        if return_vec:
            _U, _s, _Vh = np.linalg.svd(_M, full_matrices=False, compute_uv=True)
            _U = self._Q.dot(_U[:,0])
            _V = self._Q.dot(_Vh.conj()[0])
            return _s[0], _U, _V
        _s = np.linalg.svd(_M, full_matrices=False, compute_uv=False)
        return _s[0]