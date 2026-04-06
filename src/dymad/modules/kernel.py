import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dymad.numerics import DimensionEstimator

logger = logging.getLogger(__name__)


# --------------------
# Utils
# --------------------
def scaled_cdist(
    X: torch.Tensor, Z: torch.Tensor, scale: float | torch.Tensor, p: float
) -> torch.Tensor:
    """
    Pairwise distance ||X/scale - Z/scale||^p with broadcasting-friendly scaling.

    Args:
        X (torch.Tensor): (N,d)
        Z (torch.Tensor): (M,d)
        scale (float or torch.Tensor): (d,) or scalar, positive
        p (float): order of the norm
    """
    Xn, Zn = X / scale, Z / scale
    dists = torch.cdist(Xn, Zn, p=p)  # (N,M)
    return dists


def inv_softplus(y: float, dtype) -> torch.Tensor:
    """Inverse of softplus, for initialization."""
    return torch.log(torch.exp(torch.tensor(float(y), dtype=dtype)) - 1)


# --------------------
# Kernels
#
# Besides base classes, naming convention: Kernel[A][B]
#   A: Sc (scalar) or Op (operator-valued)
#   B: Specific type of kernel, e.g., RBF, Separable, etc.
# --------------------


# Bases
class KernelAbstract(nn.Module, ABC):
    """
    Base interface for all kernels (scalar or operator-valued).
    """

    def __init__(self, in_dim: int, dtype=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.dtype = dtype if dtype is not None else torch.float64

    @abstractmethod
    def forward(self, X: torch.Tensor, Z: torch.Tensor = None) -> torch.Tensor:
        """
        Compute kernel between X (N,d) and Z (M,d).

        If Z is None, compute K(X,X).

        Returns:
          - Scalar kernels: (N, M)
          - Operator-valued kernels: (N, Dy, M, Dy)
        """
        pass

    @property
    @abstractmethod
    def is_operator_valued(self) -> bool:
        """True for operator-valued kernels; False for scalar kernels."""
        pass

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        """
        Prepare data-dependent structures from Xref (N,d).
        Must be differentiable if kernel params are learnable.

        By default the kernel is data-independent and does nothing.
        """
        pass


# Drived Bases
class KernelScalarValued(KernelAbstract, ABC):
    def __init__(self, in_dim: int, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self.out_dim = 1

    @property
    def is_operator_valued(self) -> bool:
        return False


class KernelOperatorValued(KernelAbstract, ABC):
    def __init__(self, in_dim: int, out_dim: int, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self.out_dim = int(out_dim)

    @property
    def is_operator_valued(self) -> bool:
        return True


class KernelOperatorValuedScalars(KernelOperatorValued):
    """
    Operator-valued kernel induced by scalar kernels
    Output shape: (..., N, Dy, M, Dy)
    """

    def __init__(self, kernels: KernelScalarValued | list, out_dim: int, dtype=None):
        if isinstance(kernels, KernelScalarValued):
            kernels = nn.ModuleList([kernels])
        elif isinstance(kernels, list):
            kernels = nn.ModuleList(kernels)
        self.n_kernels = len(kernels)
        self.in_dim = kernels[0].in_dim
        for k in kernels:
            assert isinstance(k, KernelScalarValued)
            assert k.in_dim == self.in_dim

        super().__init__(self.in_dim, out_dim, dtype=dtype)
        self.scalar_kernels = kernels

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        for _k in self.scalar_kernels:
            _k.set_reference_data(Xref)


# Actual kernels
## Scalar kernels
class KernelScRBF(KernelScalarValued):
    """
    Scalar RBF: k(x,z) = exp(-0.5 * ||x - z||^2 / ell^2)
    Learnable positive lengthscale.
    """

    def __init__(self, in_dim: int, lengthscale_init: float | None = None, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        if lengthscale_init is None:
            self._log_ell = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_ell = nn.Parameter(
                torch.tensor(float(lengthscale_init), dtype=self.dtype).log()
            )

    def __repr__(self) -> str:
        return f"KernelScRBF(in_dim={self.in_dim}, ell={self.ell}, dtype={self.dtype})"

    @property
    def ell(self):
        # positive via softplus
        return F.softplus(self._log_ell)

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        with torch.no_grad():
            if self._log_ell.numel() == 0:
                est = DimensionEstimator(
                    data=Xref.detach().cpu().numpy(), Knn=None, bracket=[-30, 10]
                )
                est()
                _tmp = np.sqrt(est._ref_l2dist * est._ref_scalar / 2)
                _tmp = inv_softplus(_tmp, self.dtype)
                self._log_ell.requires_grad = False
                self._log_ell.set_(_tmp)
                self._log_ell.requires_grad = True
                logger.info(f"Estimated lengthscale: {self.ell}")

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        sq = scaled_cdist(X, Z, self.ell, 2) ** 2
        return torch.exp(-0.5 * sq)


class KernelScExp(KernelScalarValued):
    """
    Scalar Exponential: k(x,z) = exp(-||x - z|| / ell)
    Learnable positive lengthscale.
    """

    def __init__(self, in_dim: int, lengthscale_init: float | None = None, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self._log_ell = nn.Parameter(torch.tensor(float(lengthscale_init), dtype=self.dtype).log())

    def __repr__(self) -> str:
        return f"KernelScExp(in_dim={self.in_dim}, ell={self.ell}, dtype={self.dtype})"

    @property
    def ell(self):
        # positive via softplus
        return F.softplus(self._log_ell)

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        sq = scaled_cdist(X, Z, self.ell, 2)
        return torch.exp(-sq)


class KernelScDM(KernelScalarValued):
    """
    Symmetric-normalized diffusion kernel via diffusion maps.

    Everything keeps autograd for eps and t.
    """

    def __init__(self, in_dim: int, eps_init: float | None = None, t_init: float = 1.0, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        if eps_init is None:
            self._log_eps = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_eps = nn.Parameter(torch.tensor(float(eps_init), dtype=self.dtype).log())
        _tmp = inv_softplus(t_init, self.dtype)
        self._log_t = nn.Parameter(_tmp)

        # caches
        self.register_parameter("_Xref", nn.Parameter(torch.empty(0, dtype=self.dtype)))
        self.register_parameter("_D", nn.Parameter(torch.empty(0, dtype=self.dtype)))
        self.register_parameter("_Dinv1", nn.Parameter(torch.empty(0, dtype=self.dtype)))

    def __repr__(self) -> str:
        return f"KernelScDM(in_dim={self.in_dim}, eps={self.eps}, t={self.t}, dtype={self.dtype})"

    @property
    def eps(self):  # eps > 0
        return F.softplus(self._log_eps)

    @property
    def t(self):  # t > 0
        return F.softplus(self._log_t)

    def _rbf(self, X, Z):
        # K_eps = exp(-||x-z||^2 / (4 eps))  (scaled so that bandwidth uses eps directly)
        scale = (4.0 * self.eps).sqrt()
        sq = scaled_cdist(X, Z, scale, 2) ** 2
        return torch.exp(-sq)

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        self._Xref.requires_grad = False
        self._Xref.set_(Xref)
        self._Xref.requires_grad = False

        with torch.no_grad():
            if self._log_eps.numel() == 0:
                est = DimensionEstimator(
                    data=Xref.detach().cpu().numpy(), Knn=None, bracket=[-30, 10]
                )
                est()
                _tmp = inv_softplus(est._ref_l2dist * est._ref_scalar / 4, self.dtype)
                self._log_eps.requires_grad = False
                self._log_eps.set_(_tmp)
                self._log_eps.requires_grad = True
                logger.info(f"Estimated epsilon: {self.eps}")

        self._D.requires_grad = False
        self._Dinv1.requires_grad = False

        W = self._rbf(Xref, Xref)
        self._D.set_(W.sum(axis=-1) ** (-self.t))
        W = self._D[..., None] * W * self._D[..., None, :]
        self._Dinv1.set_(W.sum(axis=-1) ** (-0.5))

    def forward(self, X, Z=None):
        if Z is None:
            Z = self._Xref

        if X.data_ptr() == Z.data_ptr() and X.data_ptr() == self._Xref.data_ptr():
            # K(X,X) with reference data, use cached
            W = self._rbf(X, X)
            W = self._D[..., None] * W * self._D[..., None, :]
            W = self._Dinv1[..., None] * W * self._Dinv1[..., None, :]
            return W

        W = self._rbf(X, Z)
        D = W.sum(axis=-1) ** (-self.t)
        W = D[..., None] * W * self._D[..., None, :]
        Dinv1 = W.sum(axis=-1) ** (-0.5)
        W = Dinv1[..., None] * W * self._Dinv1[..., None, :]
        return W


## Operator kernels
class KernelOpSeparable(KernelOperatorValuedScalars):
    """
    Separable operator-valued kernel K(x,z) = sum_i k_i(x,z; ell) * B_i
    where B_i = L_i L_i^T is PSD and learnable.
    Output shape: (..., N, Dy, M, Dy)
    """

    def __init__(
        self,
        kernels: KernelScalarValued | list,
        out_dim: int,
        Ls: torch.Tensor | list[torch.Tensor] = None,
        dtype=None,
    ):
        super().__init__(kernels, out_dim, dtype=dtype)

        if Ls is None:
            L0 = torch.stack(
                [torch.eye(out_dim, dtype=self.dtype) for _ in range(self.n_kernels)], dim=0
            )
            self.Ls = nn.Parameter(L0.clone())  # (n_kernels, Dy, Dy)
        else:
            Ls = torch.atleast_3d(torch.as_tensor(Ls, dtype=self.dtype))
            assert Ls.ndim == 3
            assert (
                Ls.shape[0] == self.n_kernels and Ls.shape[1] == out_dim and Ls.shape[2] == out_dim
            )
            self.Ls = nn.Parameter(Ls.clone())

    def __repr__(self) -> str:
        _s = [self.scalar_kernels[i].__repr__() for i in range(self.n_kernels)]
        return (
            f"KernelOpSeparable(in_dim={self.in_dim}, out_dim={self.out_dim}, n_kernels={self.n_kernels}, dtype={self.dtype})\n"
            f"\t\tLs_shapes={[self.Ls.shape]}\n\twith:\n\t\t" + "\n\t\t".join(_s)
        )

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        k = torch.stack([_k(X, Z) for _k in self.scalar_kernels], dim=0)  # (n_kernels, ..., M)
        L = torch.tril(self.Ls)
        B = torch.matmul(L, L.transpose(-1, -2))  # (n_kernels, Dy, Dy)
        # Output: (..., Dy, M, Dy) = sum_i k_i(x,z) * B_i
        out = torch.einsum("i ... m, i a b -> ... a m b", k, B)
        return out


class KernelOpTangent(KernelOperatorValued):
    """
    Operator-valued kernel for vector fields on a manifold

    For manifold of intrinsic dimension d and ambient dimension Dy:

        K(x,z) = k(x,z; ell) * T(x') O(x',z') T(z')^T

    where O(x',z') = T(x')^T T(z') and T, of (Dy,d), are tangent basis vectors at x' and z',
    and the ' denotes the state part of the input (the first out_dim dimensions).
    k is a scalar kernel that includes both states and inputs.

    Returns a factored representation of the kernel to stay in intrinsic dimension

        k(x,z; ell) O(x,z), T(x), T(z)

    of shapes: (..., d, M, d), (..., d, Dy), (M, d, Dy)
    """

    def __init__(self, kernel: KernelScalarValued, out_dim: int, dtype=None):
        assert isinstance(kernel, KernelScalarValued)
        self.in_dim = kernel.in_dim

        super().__init__(self.in_dim, out_dim, dtype=dtype)
        self.scalar_kernel = kernel

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        self.scalar_kernel.set_reference_data(Xref)

    def set_manifold(self, manifold) -> None:
        # Only requires manifold to provide an _estimate_tangent method
        # which can operate in batch, and give tangent bases of shape (...,d,Dy)
        self._manifold = manifold

    def __repr__(self) -> str:
        return (
            f"KernelOpTangent(in_dim={self.in_dim}, out_dim={self.out_dim}, dtype={self.dtype})\n"
            f"\t\twith:\n\t\t{self.scalar_kernel.__repr__()}"
        )

    def _tangent(self, X: torch.Tensor) -> torch.Tensor:
        _T = self._manifold._estimate_tangent(X[..., : self.out_dim].detach().cpu().numpy())
        return torch.as_tensor(_T, dtype=self.dtype, device=X.device)

    def forward(self, X, Z=None):
        k = self.scalar_kernel(X, Z)  # (..., M)

        if Z is None:
            Z = X

        _Tx = self._tangent(X)  # (..., d, Dy)
        _Tz = self._tangent(Z)  # (M, d, Dy)
        out = torch.einsum("... a i, m b i, ... m -> ... a m b", _Tx, _Tz, k)  # (..., d, M, d)

        return out, _Tx, _Tz
