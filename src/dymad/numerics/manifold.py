import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sps
import torch
from scipy.optimize import minimize_scalar
from sklearn.neighbors import KDTree
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def tangent_1circle(x):
    _x = np.atleast_2d(x)
    _t = np.arctan2(_x[:, 1], _x[:, 0])
    _T = np.vstack([np.sin(_t), -np.cos(_t)]).T
    return _T.reshape(len(_x), 1, 2)


def tangent_2torus(x, R):
    _X = np.atleast_2d(x)
    _x, _y, _z = _X[:, 0], _X[:, 1], _X[:, 2]
    _p = np.arctan2(_y, _x)
    _c, _s = np.cos(_p), np.sin(_p)
    _r = np.sqrt(_x**2 + _y**2)
    _T1 = np.vstack([-_s, _c, np.zeros_like(_s)]).T
    _T2 = np.vstack([-_z * _c, -_z * _s, _r - R]).T
    _T2 /= np.linalg.norm(_T2, axis=1).reshape(-1, 1)
    _T = np.swapaxes(np.array([_T1, _T2]), 0, 1)
    return _T.reshape(len(_x), 2, 3)


class DimensionEstimator:
    """
    Estimate the intrinsic dimension of a point cloud using kernel method.

    Based on https://doi.org/10.1016/j.acha.2015.01.001
    See Fig. 6

    In implementation, we analytically evaluate

        d(log(S(e)))/d(log(e))

    There are three operation modes:

        - Given `data` only, Knn=None: use all pairwise distances
        - Given `data` and Knn=int: use kNN distances, a KDTree will be built
        - Given `tree` and Knn=int: use kNN distances from the tree, `data` will not be used

    Args:
        data: Input data, shape (N, d).
        Knn: Number of nearest neighbors to use. If None, use all pairwise distances.
        tree: A precomputed KDTree instance. If given, `data` is not used.
        bracket: Bracket for the scalar minimization
        tol: Tolerance for biased rounding of fractional dimension
    """

    def __init__(self, data=None, Knn=None, tree=None, bracket=None, tol=0.2):
        if bracket is None:
            bracket = [-20, 5]
        if tree is not None:
            self._tree = tree
            self._data = np.asarray(tree.data)
            self._Knn = Knn
        else:
            self._data = np.array(data)
            self._tree = KDTree(self._data)
            self._Knn = Knn
        self._Ndat, self._Ndim = self._data.shape

        self._bracket = bracket
        self._tol = tol

        # cache
        self._S = None
        self._dS = None

    def __call__(self):
        # Normalized squared distances
        if self._Knn is None:
            dst = sps.distance.pdist(self._data) ** 2  # Excludes self-distances
            scl = 2  # These are only half of the distances
        else:
            dst, _ = self._tree.query(self._data, k=self._Knn + 1)
            dst = dst[:, 1:].reshape(-1) ** 2  # Remove self-distances
            scl = 1  # Both sides of distances are included
        dmx = np.max(dst)
        dst /= dmx

        # Tuning functions
        def S(e):
            tmp = scl * np.sum(np.exp(-dst / e)) + self._Ndat
            return tmp / self._Ndat**2

        def dS(e):
            _k = np.exp(-dst / e)
            _S = scl * np.sum(_k) + self._Ndat
            _d = scl * dst.dot(_k)
            return _d / _S / e

        # Find the intrinsic dimension as the max of slope
        def func(e):
            return -2 * dS(2.0**e)

        res = minimize_scalar(func, bracket=self._bracket)
        est = -func(res.x)
        # For fractional dimensions, we do a biased rounding
        if np.ceil(est) - est <= self._tol:
            dim = int(np.ceil(est))
        else:
            dim = int(np.floor(est))
        tmp = (2**res.x * dmx) ** (1 / est)

        # Key results
        self._dim = dim
        self._est = est
        self._ref_bandwidth = tmp
        self._ref_l2dist = dmx
        self._ref_scalar = 2**res.x

        # Update cache
        self._S = S
        self._dS = dS

        logger.info(f"Dimension estimation of {self._Ndat} points in {self._Ndim}-dim")
        logger.info(f"Estimated intrinsic dimension: {dim} (est={est:4.3f})")
        logger.info(
            f"Reference bandwidth: {tmp:4.3e} (l2dist={dmx:4.3e}, scalar={self._ref_scalar:4.3e})"
        )

        return dim

    def plot(self, N=20, fig=None, sty="b-"):
        eps = 2.0 ** np.linspace(self._bracket[0], self._bracket[1], N)

        val = [self._S(_e) for _e in eps]
        slp = [2 * self._dS(_e) for _e in eps]

        if fig is None:
            f, ax = plt.subplots(nrows=2, sharex=True)
        else:
            f, ax = fig
        ax[0].loglog(eps, val, sty)
        ax[1].semilogx(eps, slp, sty)
        ax[1].semilogx(
            self._ref_scalar,
            self._est,
            sty[0] + "o",
            markerfacecolor="none",
            label=f"Estimated dim: {self._est:4.3f}",
        )
        ax[0].set_ylabel(r"$S(\epsilon)$")
        ax[1].set_xlabel(r"$\epsilon$")
        ax[1].set_ylabel(r"$d$")
        ax[1].legend()
        return f, ax

    def sanity_check(self, K=None, ifref=True, ifnrm=True):
        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        Nknn = tmp if K is None else K

        # Local PCA
        _, _i = self._tree.query(self._data, k=Nknn)
        _V = self._data[_i] - self._data[:, None, :]
        _, svs, _ = np.linalg.svd(_V, full_matrices=False)
        _avr = np.mean(svs, axis=0)
        _std = np.std(svs, axis=0)
        # Global PCA, as reference
        _tmp = self._data - np.mean(self._data, axis=0)
        _, sv, _ = np.linalg.svd(_tmp, full_matrices=False)

        scl = np.max(_avr) if ifnrm else 1.0
        ds = np.arange(len(_avr)) + 1
        f = plt.figure()
        plt.plot(ds, _avr / scl, label="Local PCA mean")
        plt.fill_between(ds, (_avr + _std) / scl, (_avr - _std) / scl, alpha=0.4)
        if ifref:
            scl = np.max(sv) if ifnrm else 1.0
            plt.plot(np.arange(len(sv)) + 1, sv / scl, "r--", label="Global PCA")
        plt.xlabel("SV Index")
        if ifnrm:
            plt.ylabel("Normalized SV")
        else:
            plt.ylabel("SV")
        plt.legend()

        return (_avr, _std), f


class Manifold:
    def __init__(self, data, d, K=None, g=None, T=None, iforit=False, extT=None):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape

        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        self._Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Intrinsic dimension
        self._Nman = d

        # Order of GMLS
        self._Nlsq = 2 if g is None else g
        self._fphi = PolynomialFeatures(self._Nlsq, include_bias=False)
        self._fpsi = PolynomialFeatures(self._Nlsq, include_bias=True)  # General GMLS

        # Order of tangent space estimation
        self._Ntan = 0 if T is None else T
        if self._Ntan > 0:
            self._ftau = PolynomialFeatures(self._Ntan, include_bias=False)
            self._estimate_tangent = self._estimate_tangent_ho
        else:
            self._estimate_tangent = self._estimate_tangent_1

        # Orientation of tangent vectors
        self._iforit = iforit

        if extT is None:
            self._ifprecomp = False  # Not precomputed yet
        else:
            self._ifprecomp = True
            self._T = np.array(extT)

        logger.info(
            f"Manifold info: {self._Ndat} points of {self._Ndim}-dim, intrinsic {self._Nman}-dim"
        )

    def _tree_query(self, x):
        _, _i = self._tree.query(x, k=self._Nknn)
        return _i

    def precompute(self):
        logger.info("  Precomputing")
        if self._ifprecomp:
            assert self._T.shape == (self._Ndat, self._Nman, self._Ndim)
            logger.info("  Already done, or T supplied externally; skipping")
            return

        self._T = self._estimate_tangent(self._data)
        if self._iforit:
            logger.info("  Orienting tangent vectors")
            if self._Nman == 1:
                rems = list(range(1, self._Ndat))
                curr = 0
                while len(rems) > 0:
                    _i = self._tree_query(self._data[curr])
                    _T = self._T[curr]
                    for _j in _i:
                        if _j in rems:
                            _d = self._T[_j].dot(_T.T)
                            if _d < 0:
                                self._T[_j] = -self._T[_j]
                            rems.remove(_j)
                            curr = _j
            else:
                raise NotImplementedError(
                    "Orientation for higher-dim manifold not implemented; \
                    supply oriented T externally"
                )
        self._ifprecomp = True
        logger.info("  Done")

    def gmls(self, x, Y, ret_der=False):
        # y = g(T^T (x-X))
        #
        # The derivative is approximate, as x impacts the tangent space estimation,
        # but only the gradients on polynomial are included
        #
        # dy = dg/dd * T^T
        _i = self._tree_query(np.atleast_2d(x))
        _T, _V = self._estimate_tangent(x, ret_V=True)
        _B = np.matmul(_V, np.swapaxes(_T, -2, -1))
        _P = self._poly_eval(self._fpsi, _B)
        _C = np.matmul(np.linalg.pinv(_P), np.atleast_3d(Y[_i]))
        _r = _C[..., 0, :].squeeze()
        if ret_der:
            _tmp = _C[..., 1 : self._Nman + 1, :]
            _rder = np.matmul(np.swapaxes(_tmp, -2, -1), _T)
            if _rder.ndim == 3 and _rder.shape[0] == 1:
                _rder = _rder[0]
            return _r, _rder
        return _r

    def _poly_eval(self, f, B):
        # PolynomialFeatures only supports 2D input
        # so we reshape input to 2D and then reshape back
        _s = B.shape[:-1]
        _P = f.fit_transform(B.reshape(-1, self._Nman)).reshape(*_s, -1)
        return _P

    def _estimate_normal(self, base, dx):
        _T, _V = self._estimate_tangent(base, ret_V=True)
        _B = np.matmul(_V, np.swapaxes(_T, -2, -1))
        _P = self._poly_eval(self._fphi, _B)
        _b = np.atleast_2d(np.matmul(dx, np.swapaxes(_T, -2, -1)))
        _p = self._poly_eval(self._fphi, _b)
        _tmp = np.matmul(_p, np.linalg.pinv(_P))
        _n = np.matmul(_tmp, _V - np.matmul(_B, _T))
        return _n.squeeze()

    def _estimate_tangent_1(self, x, ret_V=False):
        _i = self._tree_query(x)
        _V = self._data[_i] - x[..., None, :]
        _, _, _Vh = np.linalg.svd(_V, full_matrices=False)
        _T = _Vh.conj()[..., : self._Nman, :]
        if ret_V:
            return _T, _V
        return _T

    def _estimate_tangent_ho(self, x, ret_V=False):
        _T, _V = self._estimate_tangent_1(x, ret_V=True)
        _B = np.matmul(_V, np.swapaxes(_T, -2, -1))
        _P = self._poly_eval(self._ftau, _B)
        _C = np.matmul(np.linalg.pinv(_P), _V - np.matmul(_B, _T))
        _T += _C[..., : self._Nman, :]
        _tmp = np.linalg.qr(np.swapaxes(_T, -2, -1), mode="reduced")[0]
        _T = np.swapaxes(_tmp, -2, -1)
        if ret_V:
            return _T, _V
        return _T

    def plot2d(self, N, scl=1):
        assert self._Ndim == 2
        _d = self._data

        f, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(_d[:, 0], _d[:, 1], "b.", markersize=1)
        for _i in range(N):
            _p = _d[_i] + scl * self._T[_i]
            _c = np.vstack([_d[_i], _p]).T
            plt.plot(_c[0], _c[1], "k-")
        return f, ax

    def plot3d(self, N, scl=1):
        assert self._Ndim == 3
        _d = self._data

        f = plt.figure()
        ax = f.add_subplot(projection="3d")
        ax.plot(_d[:, 0], _d[:, 1], _d[:, 2], "b.", markersize=1)
        for _i in range(N):
            _p = _d[_i] + scl * self._T[_i]
            for _j in range(2):
                _c = np.vstack([_d[_i], _p[_j]]).T
                ax.plot(_c[0], _c[1], _c[2], "k-")
        return f, ax

    def to_tensors(self):
        # To interface with PyTorch for saving state dicts
        return {
            "data": torch.from_numpy(self._data),
            "d": torch.tensor(self._Nman, dtype=torch.int64),
            "K": torch.tensor(self._Nknn, dtype=torch.int64),
            "g": torch.tensor(self._Nlsq, dtype=torch.int64),
            "T": torch.tensor(self._Ntan, dtype=torch.int64),
            "iforit": torch.tensor(self._iforit, dtype=torch.bool),
            "extT": torch.from_numpy(self._T) if self._ifprecomp else None,
        }

    @classmethod
    def from_tensors(cls, t):
        # To interface with PyTorch to load a state dict
        data = t["data"].detach().cpu().numpy()
        d = t["d"].item()
        K = t["K"].item()
        g = t["g"].item()
        T = t["T"].item()
        iforit = t["iforit"].item()
        extT = t["extT"].detach().cpu().numpy() if t["extT"] is not None else None
        return cls(data, d, K, g, T, iforit, extT)


class ManifoldAltTree(Manifold):
    """
    The only difference from Manifold is that a KDTree on a different set of points is used
    to find the kNN points for GMLS.

    Specifically, instead of finding kNN points of x from the original data X,
    we first transform x to z using a given transform function, and then find
    the kNN points of z from a given set of points Z (with a KDTree built on Z).

    The rest is still the same as Manifold.
    """

    def __init__(
        self,
        data,
        d,
        K=None,
        g=None,
        T=None,
        iforit=False,
        extT=None,
        tree_data=None,
        tree_transform=None,
    ):
        super().__init__(data, d, K, g, T, iforit, extT)

        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(tree_data, leafsize=_leaf)
        self._ftree = tree_transform

    def _tree_query(self, x):
        z = self._ftree(x)
        _, _i = self._tree.query(z, k=self._Nknn)
        return _i


class ManifoldAnalytical(Manifold):
    def __init__(self, data, d, K=None, g=None, fT=None):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape

        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        self._Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Intrinsic dimension
        self._Nman = d

        # Order of GMLS
        self._Nlsq = 2 if g is None else g
        self._fphi = PolynomialFeatures(self._Nlsq, include_bias=False)
        self._fpsi = PolynomialFeatures(self._Nlsq, include_bias=True)  # General GMLS

        # Function giving tangent space basis
        self._tangent_func = fT

        # Possible data members
        self._T = []  # Tangent space basis for every data point
        self._ifprecomp = False  # Not precomputed yet

        logger.info(
            f"Manifold info: {self._Ndat} points of {self._Ndim}-dim, intrinsic {self._Nman}-dim"
        )

    def precompute(self):
        logger.info("  Precomputing")
        self._T = self._estimate_tangent(self._data)
        self._ifprecomp = True
        logger.info("  Done")

    def _estimate_tangent(self, x, ret_V=False):
        _T = self._tangent_func(x)
        if ret_V:
            _, _i = self._tree.query(x, k=self._Nknn)
            _V = self._data[_i] - x[..., None, :]
            return _T, _V
        return _T
