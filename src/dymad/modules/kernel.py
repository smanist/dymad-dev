from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dymad.modules.helpers import _swap_parameter_storage
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


def inv_softplus(y: float | np.floating[Any], dtype: torch.dtype) -> torch.Tensor:
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

    def __init__(self, in_dim: int, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.dtype: torch.dtype = dtype if dtype is not None else torch.float64

    @abstractmethod
    def forward(self, X: torch.Tensor, Z: torch.Tensor | None = None) -> torch.Tensor:
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
    def __init__(self, in_dim: int, dtype: torch.dtype | None = None):
        super().__init__(in_dim, dtype=dtype)
        self.out_dim = 1

    @property
    def is_operator_valued(self) -> bool:
        return False


class KernelOperatorValued(KernelAbstract, ABC):
    def __init__(self, in_dim: int, out_dim: int, dtype: torch.dtype | None = None):
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

    def __init__(
        self,
        kernels: KernelScalarValued | Sequence[KernelScalarValued] | nn.ModuleList,
        out_dim: int,
        dtype: torch.dtype | None = None,
    ):
        if isinstance(kernels, KernelScalarValued):
            module_kernels = nn.ModuleList([kernels])
        elif isinstance(kernels, Sequence):
            module_kernels = nn.ModuleList(list(kernels))
        else:
            module_kernels = kernels
        self.n_kernels = len(module_kernels)
        first_kernel = cast(KernelScalarValued, module_kernels[0])
        self.in_dim = first_kernel.in_dim
        for k in module_kernels:
            assert isinstance(k, KernelScalarValued)
            assert k.in_dim == self.in_dim

        super().__init__(self.in_dim, out_dim, dtype=dtype)
        self.scalar_kernels: nn.ModuleList = module_kernels

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        for _k in self.scalar_kernels:
            cast(KernelScalarValued, _k).set_reference_data(Xref)


# Actual kernels
## Scalar kernels
class KernelScRBF(KernelScalarValued):
    """
    Scalar RBF: k(x,z) = exp(-0.5 * ||x - z||^2 / ell^2)
    Learnable positive lengthscale.
    """

    def __init__(
        self, in_dim: int, lengthscale_init: float | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__(in_dim, dtype=dtype)
        if lengthscale_init is None:
            self._log_ell: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype))
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
                _swap_parameter_storage(self._log_ell, _tmp, requires_grad=True)
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

    def __init__(
        self, in_dim: int, lengthscale_init: float | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__(in_dim, dtype=dtype)
        if lengthscale_init is None:
            self._log_ell: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_ell = nn.Parameter(
                torch.tensor(float(lengthscale_init), dtype=self.dtype).log()
            )

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

    def __init__(
        self,
        in_dim: int,
        eps_init: float | None = None,
        t_init: float = 1.0,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(in_dim, dtype=dtype)
        if eps_init is None:
            self._log_eps: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_eps = nn.Parameter(torch.tensor(float(eps_init), dtype=self.dtype).log())
        _tmp = inv_softplus(t_init, self.dtype)
        self._log_t: nn.Parameter = nn.Parameter(_tmp)

        # caches
        self._Xref: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype), requires_grad=False
        )
        self._D: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype), requires_grad=False)
        self._Dinv1: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype), requires_grad=False
        )

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
        _swap_parameter_storage(self._Xref, Xref, requires_grad=False)

        with torch.no_grad():
            if self._log_eps.numel() == 0:
                est = DimensionEstimator(
                    data=Xref.detach().cpu().numpy(), Knn=None, bracket=[-30, 10]
                )
                est()
                _tmp = inv_softplus(est._ref_l2dist * est._ref_scalar / 4, self.dtype)
                _swap_parameter_storage(self._log_eps, _tmp, requires_grad=True)
                logger.info(f"Estimated epsilon: {self.eps}")

        W = self._rbf(Xref, Xref)
        _swap_parameter_storage(self._D, W.sum(dim=-1) ** (-self.t))
        W = self._D[..., None] * W * self._D[..., None, :]
        _swap_parameter_storage(self._Dinv1, W.sum(dim=-1) ** (-0.5))

    def forward(self, X: torch.Tensor, Z: torch.Tensor | None = None):
        if Z is None:
            Z = self._Xref

        if X.data_ptr() == Z.data_ptr() and X.data_ptr() == self._Xref.data_ptr():
            # K(X,X) with reference data, use cached
            W = self._rbf(X, X)
            W = self._D[..., None] * W * self._D[..., None, :]
            W = self._Dinv1[..., None] * W * self._Dinv1[..., None, :]
            return W

        W = self._rbf(X, Z)
        D = W.sum(dim=-1) ** (-self.t)
        W = D[..., None] * W * self._D[..., None, :]
        Dinv1 = W.sum(dim=-1) ** (-0.5)
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
        kernels: KernelScalarValued | Sequence[KernelScalarValued] | nn.ModuleList,
        out_dim: int,
        Ls: torch.Tensor | Sequence[torch.Tensor] | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(kernels, out_dim, dtype=dtype)

        if Ls is None:
            L0 = torch.stack(
                [torch.eye(out_dim, dtype=self.dtype) for _ in range(self.n_kernels)], dim=0
            )
            self.Ls = nn.Parameter(L0.clone())  # (n_kernels, Dy, Dy)
        else:
            if isinstance(Ls, Sequence):
                Ls_tensor = torch.stack(
                    [torch.as_tensor(item, dtype=self.dtype) for item in Ls], dim=0
                )
            else:
                Ls_tensor = torch.as_tensor(Ls, dtype=self.dtype)
            Ls_tensor = torch.atleast_3d(Ls_tensor)
            assert Ls_tensor.ndim == 3
            assert (
                Ls_tensor.shape[0] == self.n_kernels
                and Ls_tensor.shape[1] == out_dim
                and Ls_tensor.shape[2] == out_dim
            )
            self.Ls = nn.Parameter(Ls_tensor.clone())

    def __repr__(self) -> str:
        _s = [
            cast(KernelScalarValued, self.scalar_kernels[i]).__repr__()
            for i in range(self.n_kernels)
        ]
        return (
            f"KernelOpSeparable(in_dim={self.in_dim}, out_dim={self.out_dim}, n_kernels={self.n_kernels}, dtype={self.dtype})\n"
            f"\t\tLs_shapes={[self.Ls.shape]}\n\twith:\n\t\t" + "\n\t\t".join(_s)
        )

    def forward(self, X: torch.Tensor, Z: torch.Tensor | None = None):
        if Z is None:
            Z = X
        k = torch.stack([cast(KernelScalarValued, _k)(X, Z) for _k in self.scalar_kernels], dim=0)
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

    def __init__(self, kernel: KernelScalarValued, out_dim: int, dtype: torch.dtype | None = None):
        assert isinstance(kernel, KernelScalarValued)
        self.in_dim = kernel.in_dim

        super().__init__(self.in_dim, out_dim, dtype=dtype)
        self.scalar_kernel: KernelScalarValued = kernel
        self._manifold: Any | None = None

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        self.scalar_kernel.set_reference_data(Xref)

    def set_manifold(self, manifold: Any) -> None:
        # Only requires manifold to provide an _estimate_tangent method
        # which can operate in batch, and give tangent bases of shape (...,d,Dy)
        self._manifold = manifold

    def __repr__(self) -> str:
        return (
            f"KernelOpTangent(in_dim={self.in_dim}, out_dim={self.out_dim}, dtype={self.dtype})\n"
            f"\t\twith:\n\t\t{self.scalar_kernel.__repr__()}"
        )

    def _tangent(self, X: torch.Tensor) -> torch.Tensor:
        manifold = self._manifold
        if manifold is None:
            raise RuntimeError("Tangent kernel requires manifold data before evaluation.")
        _T = manifold._estimate_tangent(X[..., : self.out_dim].detach().cpu().numpy())
        return torch.as_tensor(_T, dtype=self.dtype, device=X.device)

    def forward(self, X: torch.Tensor, Z: torch.Tensor | None = None):
        k = self.scalar_kernel(X, Z)  # (..., M)

        if Z is None:
            Z = X

        _Tx = self._tangent(X)  # (..., d, Dy)
        _Tz = self._tangent(Z)  # (M, d, Dy)
        out = torch.einsum("... a i, m b i, ... m -> ... a m b", _Tx, _Tz, k)  # (..., d, M, d)

        return out, _Tx, _Tz
