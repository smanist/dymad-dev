import logging
from typing import Any, cast

import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix, spdiags
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from dymad.numerics.manifold import DimensionEstimator

logger = logging.getLogger(__name__)


def dS(eps, D, F):
    tmp = np.exp(-D / (F * eps))
    fun = np.mean(tmp)
    grd = np.mean(D * tmp) / F
    return grd / fun / eps


def epsOpt(D, F, ret_val=False):
    # Normalize by max distance for epsilon at unit scale
    _max = np.max(D)

    def _fun(e):
        return -2 * dS(2.0**e, D / _max, F)

    _res = cast(Any, minimize_scalar(_fun, bracket=[-30, 10]))
    _eps = 2**_res.x * _max
    if ret_val:
        _dim = -_fun(_res.x)
        return _eps, _dim
    return _eps


def compDist(a, b, K):
    d2 = cdist(np.atleast_2d(a), np.atleast_2d(b), metric="sqeuclidean")
    if K is None:
        return d2
    idx = np.argsort(d2, axis=1)[:, :K]
    mask = np.ones_like(d2, dtype=bool)
    np.put_along_axis(mask, idx, False, axis=1)
    d2[mask] = np.inf
    return d2


def genMat(data, inds, N=None, ifsym=False):
    # Generate a MxN sparse matrix
    # data and inds are Mxk
    # If M=N and ifsym=True, the matrix is enforced to be symmetric
    M, k = inds.shape
    if N is None:
        N = M
    row = np.tile(np.arange(M), (k, 1)).T.reshape(-1)
    A = coo_matrix((data.reshape(-1), (row, inds.reshape(-1))), shape=(M, N))
    if ifsym:
        if M == N:
            A = (A + A.T) / 2
    return A


class DMF:
    """
    Diffusion map with dense matrix implementation

    A Knn option is added to emulate the DM class, that is a sparsified version of DM.

    Also includes a KRR implementation for sanity check of other classes.

    Rule of thumb comparing DM and DMF:

        - For eigenpairs, full DM gives accurate eigenvectors, but less accurate eigenvalues.
        - For KRR, full DM, i.e., without Knn, performs the best.
        - Whenever memory allows, use full DM.
    """

    def __init__(self, n_components, n_neighbors=None, alpha=1, epsilon=None):
        self._Npsi = n_components
        self._Knn = n_neighbors
        self._alpha = alpha
        self._epsilon = epsilon

    def _kernel(self, x, y=None):
        eps = self._require_epsilon()
        if y is None:
            d2 = compDist(x, x, self._Knn)
            W = np.exp(-d2 / (4 * eps))
            if self._Knn is not None:
                W = (W + W.T) / 2

            _qest = np.sum(W, axis=1).flatten()
            _D = _qest ** (-self._alpha)
            W = _D.reshape(-1, 1) * W * _D
            D = np.sum(W, axis=1).flatten()
            _Dinv1 = D ** (-0.5)
            W = _Dinv1.reshape(-1, 1) * W * _Dinv1

            return W, _qest, _D, _Dinv1

        d2 = compDist(y, x, self._Knn)
        W = np.exp(-d2 / (4 * eps))

        qest = np.sum(W, axis=1).flatten()
        D = qest ** (-self._alpha)
        W = D.reshape(-1, 1) * W * self._D
        D = np.sum(W, axis=1).flatten()
        Dinv1 = D ** (-0.5)
        W = Dinv1.reshape(-1, 1) * W * self._Dinv1

        return W, qest, D, Dinv1

    def _set_data(self, x):
        self._x = np.atleast_2d(x)
        self._N = self._x.shape[0]

        if self._epsilon is None:
            est = DimensionEstimator(data=x, Knn=self._Knn, bracket=[-30, 10])
            est()
            self._epsilon = est._ref_l2dist * est._ref_scalar / 4
            logger.info(f"Estimated epsilon: {self._epsilon}")

    def fit(self, x):
        self._set_data(x)

        W, self._qest, self._D, self._Dinv1 = self._kernel(self._x)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = spl.eigh(W, subset_by_index=[self._N - self._Npsi, self._N - 1])
        self._lmbd_raw = eigvals[::-1]
        self._psi_raw = eigvecs[:, ::-1]
        psi = self._Dinv1.reshape(-1, 1) * self._psi_raw
        self._lambda = -np.log(self._lmbd_raw) / self._require_epsilon()

        # Normalize eigenvectors
        peq = self._qest / self._Dinv1**2
        self._peq = peq / np.mean(peq / self._qest)
        self._norm_factors = np.sqrt(
            np.mean(psi**2 * (self._peq / self._qest).reshape(-1, 1), axis=0)
        )
        self._psi = psi / self._norm_factors

    def transform(self, x):
        """
        Nystrom extension for diffusion maps.
        """
        W, _, _, Dinv1 = self._kernel(self._x, x)
        eigvecs = Dinv1.reshape(-1, 1) * (W @ self._psi_raw) / self._lmbd_raw
        eigvecs_normalized = eigvecs / self._norm_factors
        return eigvecs_normalized

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def fit_krr(self, X, y, ridge):
        self._set_data(X)
        W, self._qest, self._D, self._Dinv1 = self._kernel(self._x)
        I = ridge * np.eye(W.shape[0])
        self._beta = spl.solve(W + I, y, assume_a="sym")

    def predict_krr(self, X):
        K = self._kernel(self._x, X)[0]
        return K @ self._beta

    def state_dict(self) -> dict[str, Any]:
        return {
            "x": self._x,
            "Npsi": self._Npsi,
            "alpha": self._alpha,
            "epsilon": self._epsilon,
            "N": self._N,
            "qest": self._qest,
            "D": self._D,
            "Dinv1": self._Dinv1,
            "lmbd_raw": self._lmbd_raw,
            "lambda": self._lambda,
            "peq": self._peq,
            "psi_raw": self._psi_raw,
            "psi": self._psi,
            "norm_factors": self._norm_factors,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._x = d["x"]
        self._Npsi = d["Npsi"]
        self._alpha = d["alpha"]
        self._epsilon = d["epsilon"]
        self._N = d["N"]
        self._qest = d["qest"]
        self._D = d["D"]
        self._Dinv1 = d["Dinv1"]
        self._lmbd_raw = d["lmbd_raw"]
        self._lambda = d["lambda"]
        self._peq = d["peq"]
        self._psi_raw = d["psi_raw"]
        self._psi = d["psi"]
        self._norm_factors = d["norm_factors"]

    def _require_epsilon(self) -> float:
        if self._epsilon is None:
            raise RuntimeError("Epsilon is not initialized.")
        return float(self._epsilon)


class DM:
    """
    Diffusion map

    Original MATLAB implementation by Tyrus Berry & John Harlim

    This implementation

        + is class-based
        + accelerates epsilon estimation by a minimizer
        + adds an approximate interpolation method

    Args:
        n_components: Number of diffusion map components
        n_neighbors: Number of nearest neighbors for graph construction
        alpha: Normalization parameter
        epsilon: Kernel bandwidth; if None, it will be estimated
    """

    def __init__(self, n_components, n_neighbors, alpha=1, epsilon=None):
        self._Npsi = n_components
        self._Knn = n_neighbors
        self._alpha = alpha
        self._epsilon = epsilon

    def fit(self, x):
        self._N = x.shape[0]

        # Step 1: Compute k-nearest neighbors
        self._tree = KDTree(x, leaf_size=self._Knn)
        distances, indices = self._tree.query(x, k=self._Knn)

        if self._epsilon is None:
            est = DimensionEstimator(tree=self._tree, Knn=self._Knn, bracket=[-30, 10])
            est()
            self._epsilon = est._ref_l2dist * est._ref_scalar / 4
            logger.info(f"Estimated epsilon: {self._epsilon}")

        # Step 2: Construct kernel matrix
        eps = self._require_epsilon()
        W = np.exp(-(distances**2) / (4 * eps))
        W = genMat(W, indices, self._N, ifsym=True)

        # Normalize
        self._qest = np.array(W.sum(axis=1)).flatten()
        self._D = spdiags(self._qest**-self._alpha, 0, self._N, self._N)
        W = self._D @ W @ self._D

        # Normalize further to create Markov matrix
        D = np.array(W.sum(axis=1)).flatten()
        self._Dinv1 = spdiags(D**-0.5, 0, self._N, self._N)
        W = self._Dinv1 @ W @ self._Dinv1

        # Step 4: Compute eigenvalues and eigenvectors
        eigvals, eigvecs = eigsh(W, k=self._Npsi, which="LM")
        self._lmbd_raw = eigvals[::-1]
        self._psi_raw = eigvecs[:, ::-1]
        psi = self._Dinv1 @ self._psi_raw
        self._lambda = -np.log(self._lmbd_raw) / eps

        # Normalize eigenvectors
        peq = self._qest * D
        self._peq = peq / np.mean(peq / self._qest)
        self._norm_factors = np.sqrt(
            np.mean(psi**2 * (self._peq / self._qest).reshape(-1, 1), axis=0)
        )
        self._psi = psi / self._norm_factors

    def transform(self, x, ifsym=False):
        """
        Nystrom extension for diffusion maps.
        """
        M = len(x)
        distances, indices = self._tree.query(x, k=self._Knn)

        W = np.exp(-(distances**2) / (4 * self._require_epsilon()))
        W = genMat(W, indices, self._N, ifsym=ifsym)

        qest = np.array(W.sum(axis=1)).flatten()
        D = spdiags(qest ** (-self._alpha), 0, M, M)
        W = D @ W @ self._D
        D = np.array(W.sum(axis=1)).flatten()

        eigvecs = (spdiags(1 / D, 0, M, M) @ W @ self._Dinv1) * self._psi_raw / self._lmbd_raw
        eigvecs_normalized = eigvecs / self._norm_factors

        return eigvecs_normalized

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "Npsi": self._Npsi,
            "Knn": self._Knn,
            "alpha": self._alpha,
            "epsilon": self._epsilon,
            "N": self._N,
            "qest": self._qest,
            "D": self._D.diagonal(),
            "Dinv1": self._Dinv1.diagonal(),
            "lmbd_raw": self._lmbd_raw,
            "lambda": self._lambda,
            "peq": self._peq,
            "psi_raw": self._psi_raw,
            "psi": self._psi,
            "tree": self._tree,
            "norm_factors": self._norm_factors,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._Npsi = d["Npsi"]
        self._Knn = d["Knn"]
        self._alpha = d["alpha"]
        self._epsilon = d["epsilon"]
        self._N = d["N"]
        self._qest = d["qest"]
        self._D = spdiags(d["D"], 0, self._N, self._N)
        self._Dinv1 = spdiags(d["Dinv1"], 0, self._N, self._N)
        self._lmbd_raw = d["lmbd_raw"]
        self._lambda = d["lambda"]
        self._peq = d["peq"]
        self._psi_raw = d["psi_raw"]
        self._psi = d["psi"]
        self._tree = d["tree"]
        self._norm_factors = d["norm_factors"]

    def _require_epsilon(self) -> float:
        if self._epsilon is None:
            raise RuntimeError("Epsilon is not initialized.")
        return float(self._epsilon)


class VBDM:
    """
    Variable bandwidth diffusion map

    https://www.sciencedirect.com/science/article/pii/S1063520315000020

    Original MATLAB implementation by Tyrus Berry & John Harlim

    This implementation

        + is class-based
        + accelerates epsilon estimation by a minimizer
        + adds an approximate interpolation method

    It works well when the sampling is dense

    Args:
        n_components: Number of diffusion map components
        n_neighbors: Number of nearest neighbors for graph construction
        Kb: Number of nearest neighbors for density estimation
        operator: 'lb' for Laplace-Beltrami, 'kb' for Kolmogorov backward
    """

    def __init__(self, n_components, n_neighbors, Kb=None, operator=None):
        self._op = operator
        self._Npsi = n_components
        self._Knn = n_neighbors
        if Kb is None:
            self._Kb = n_neighbors // 2
        else:
            assert Kb <= n_neighbors
            self._Kb = Kb

    def fit(self, x):
        # ---------------
        # Pre-processing
        # ---------------
        self._N = x.shape[0]
        self._tree = KDTree(x, leaf_size=2 * self._Knn)
        d, inds = self._tree.query(x, k=self._Knn)
        self._estimate_rho(d, inds)

        # ---------------
        # Determine the final kernel
        # ---------------
        # construct the exponent of K^S_epsilon
        d = d**2 / (self._rho.reshape(-1, 1) * self._rho[inds])

        # Tune epsilon for the final kernel
        self._epsilon = cast(float, epsOpt(d, 4, ret_val=False))

        # ---------------
        # Construct DM matrix and Solve Eigenvalue problem
        # ---------------
        # K^S_epsilon with final choice of epsilon
        tmp = np.exp(-d / (4 * self._epsilon))
        d = genMat(tmp, inds)

        # q^S_epsilon
        # The "real" one from VB kernel, but unnormalized
        self._qest_raw = np.asarray(d.sum(axis=1)).reshape(-1) / self._rho**self._dim

        # K = K^S_{epsilon,alpha}
        Dinv = spdiags(self._qest_raw ** (-self._alpha), 0, self._N, self._N)
        d = Dinv @ d @ Dinv

        # S^2 =P^2*D, where P = diag(rho), D = q^S_{epsilon,alpha}
        rho2 = self._rho * self._rho
        Ssquare = np.sum(d, axis=1).A1 * rho2

        # S^{-1}
        Sinv = spdiags(Ssquare ** (-0.5), 0, self._N, self._N)

        # epsilon*Lhat +eye(I) = Sinv*K*Sinv - P^{-2} + eye(I)
        d = Sinv @ d @ Sinv - spdiags(1 / rho2 - 1, 0, self._N, self._N)

        # this is eigenvalue of lambda = eig(eye(I) + epsilon*Lhat)
        # Since lambda^{1/epsilon} ---> e^(eig(Lhat))
        # and eig(Lhat) = log(lambda)/epsilon
        lmbd, psi = eigsh(d, k=self._Npsi, sigma=1.0, maxiter=2000)
        self._lmbd_raw = np.array(lmbd)[::-1]

        lmbd = np.log(lmbd) / self._epsilon
        self._lambda = lmbd[::-1]
        psi = Sinv @ psi
        self._proj = np.array(psi)[:, ::-1]

        # ---------------
        # Post-processing
        # ---------------
        # Normalize qest into a density by dividing by m0
        self._qest_fact = self._N * (4 * np.pi * float(self._epsilon)) ** (self._dim / 2)
        self._qest = self._qest_raw / self._qest_fact

        # U^TU = S^{-2} implies that U^TUS^2 = I
        # componentwise, this means \sum_j phi_j(x)^2 S^2(i,i) = 1 = <phi_j,\phi_j>_peq
        # but since phi_j is evaluated at x_i with sampling measure q(x),
        # then S^2(i,i) = peq(x_i)/q(x_i) which means, peq = S^2*q
        self._peqoversample = Ssquare
        peq = self._qest * Ssquare  # Invariant measure of the system

        # normalization factor Z = \frac{1}{N}sum_{j=1}^N peq(x_j)/qest(x_j)
        self._peq_fact = np.mean(peq / self._qest)
        self._peq = peq / self._peq_fact

        # normalize eigenfunctions such that \sum_i psi(x_i)^2 p(x_i)/q(x_i) = 1
        self._psi_fact = np.sqrt(np.mean(psi**2 * (self._peq / self._qest).reshape(-1, 1), axis=0))
        psi /= self._psi_fact
        self._psi = psi[:, ::-1]

    def transform_naive(self, x):
        d, inds = self._tree.query(x, k=1)
        msk = d.reshape(-1) > 1e-5
        if np.any(msk):
            logger.info(f"    VBDM: {np.sum(msk)} points not available in VBDM")
        return self._psi[inds.reshape(-1)]

    def transform(self, x, ret_den=False):
        """
        Interpolate eigenfunctions to new x
        """
        M = len(x)
        d, inds = self._tree.query(x, k=self._Knn)
        rho = self._compute_rho(d=d, inds=inds)

        # k^S_epsilon
        d = d**2 / (rho.reshape(-1, 1) * self._rho[inds])
        d = np.exp(-d / (4 * self._epsilon))
        a = np.ones(
            M,
        )

        # q^S_epsilon, unnormalized
        qest = np.sum(d, axis=1) / rho**self._dim

        # k = k^S_{epsilon,alpha}
        dinvL = qest ** (-self._alpha)
        dinvR = self._qest_raw ** (-self._alpha)
        d = dinvL.reshape(-1, 1) * d * dinvR[inds]
        a *= dinvL**2

        # S^2 = P^2*D
        rho2 = rho * rho
        Ssquare = np.sum(d, axis=1) * rho2

        # Projection to obtain eigenfunction
        d = d / Ssquare.reshape(-1, 1)
        a = a / Ssquare - 1 / rho2 + 1
        psi = np.zeros((M, self._Npsi))
        for _i in range(self._Npsi):
            psi[:, _i] = np.sum(d * self._proj[inds, _i], axis=1)
        psi = psi / (self._lmbd_raw - a.reshape(-1, 1))
        psi /= self._psi_fact

        if ret_den:
            # Normalize q_est
            qest /= self._qest_fact
            # Invariant measure of the system
            peq = qest * Ssquare / self._peq_fact
            return psi, (rho, qest, peq)
        return psi

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def _estimate_rho(self, d, inds):
        """
        Estimate intrinsic dimension;
        A preliminary estimate of sampling density;
        Compute bandwidth function
        """
        # Build ad hoc bandwidth function by autotuning epsilon for each point
        self._rho0 = np.sqrt(np.mean(d[:, 1 : self._Kb] ** 2, axis=1)).reshape(-1)

        # Pre-kernel used with ad hoc bandwidth only for estimating dimension and sampling density
        dt = d**2 / (self._rho0.reshape(-1, 1) * self._rho0[inds])

        # Tune epsilon on the pre-kernel
        self._rhoeps, self._dim = cast(tuple[float, float], epsOpt(dt, 2, ret_val=True))

        # Use ad hoc bandwidth function, rho0, to estimate the density
        tmp = np.exp(-dt / (2 * self._rhoeps)) / (2 * np.pi * self._rhoeps) ** (self._dim / 2)
        dt = genMat(tmp, inds)

        # A temporary estimate of sampling density q(x)
        # Operation on dt gives a Matrix object.  Using A1 to get a flattened array.
        qest = np.asarray(dt.sum(axis=1)).reshape(-1) / (self._N * self._rho0**self._dim)

        # Laplace-Beltrami, or Kolmogorov backward operator
        if self._op == "lb":
            # Laplace-Beltrami, c1 = 0
            self._beta = -0.5
            self._alpha = -self._dim / 4 + 0.5
        elif self._op == "kb":
            # Kolmogorov backward operator, c1 = 1
            self._beta = -0.5
            self._alpha = -self._dim / 4
        else:
            raise ValueError(f"Unknown operator {self._op}")

        # Theoretical coefficients
        self._c1 = 2 - 2 * self._alpha + self._dim * self._beta + 2 * self._beta
        self._c2 = (
            0.5
            - 2 * self._alpha
            + 2 * self._dim * self._alpha
            + self._dim * self._beta / 2
            + self._beta
        )

        # bandwidth function rho(x) from the sampling density estimate
        rho = qest**self._beta
        self._rho_fact = np.mean(rho)
        self._rho = rho.reshape(-1) / self._rho_fact

    def _compute_rho(self, x=None, d=None, inds=None):
        if d is None:
            d, inds = self._tree.query(x, k=self._Knn)
        rho0 = np.sqrt(np.mean(d[:, 1 : self._Kb] ** 2, axis=1)).reshape(-1)

        # Ad hoc bandwidth function for density estimate
        dt = d**2 / (rho0.reshape(-1, 1) * self._rho0[inds])
        dt = np.exp(-dt / (2 * self._rhoeps)) / (2 * np.pi * self._rhoeps) ** (self._dim / 2)

        # An estimate of sampling density q(x)
        qest = np.sum(dt, axis=1) / (self._N * rho0**self._dim)
        rho = qest**self._beta
        rho = rho.reshape(-1) / self._rho_fact
        return rho

    def state_dict(self) -> dict[str, Any]:
        return {
            "Knn": self._Knn,  # Transform
            "tree": self._tree,
            "rho": self._rho,
            "epsilon": self._epsilon,
            "dim": self._dim,
            "alpha": self._alpha,
            "qest_raw": self._qest_raw,
            "Npsi": self._Npsi,
            "proj": self._proj,
            "lmbd_raw": self._lmbd_raw,
            "psi_fact": self._psi_fact,
            "qest_fact": self._qest_fact,
            "peq_fact": self._peq_fact,
            "Kb": self._Kb,  # compute rho
            "rho0": self._rho0,
            "rhoeps": self._rhoeps,
            "N": self._N,
            "beta": self._beta,
            "rho_fact": self._rho_fact,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._Knn = d["Knn"]
        self._tree = d["tree"]
        self._rho = d["rho"]
        self._epsilon = d["epsilon"]
        self._dim = d["dim"]
        self._alpha = d["alpha"]
        self._qest_raw = d["qest_raw"]
        self._Npsi = d["Npsi"]
        self._proj = d["proj"]
        self._lmbd_raw = d["lmbd_raw"]
        self._psi_fact = d["psi_fact"]
        self._qest_fact = d["qest_fact"]
        self._peq_fact = d["peq_fact"]

        self._Kb = d["Kb"]
        self._rho0 = d["rho0"]
        self._rhoeps = d["rhoeps"]
        self._N = d["N"]
        self._beta = d["beta"]
        self._rho_fact = d["rho_fact"]
