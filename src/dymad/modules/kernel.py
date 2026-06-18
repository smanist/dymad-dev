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
        *,
        metric: str = "euclidean",
        periodic_axes: tuple[int, ...] | None = None,
        density_bandwidth_factor: float = 1.0,
    ):
        super().__init__(in_dim, dtype=dtype)
        if metric not in {"euclidean", "periodic"}:
            raise ValueError("metric must be either 'euclidean' or 'periodic'.")
        if density_bandwidth_factor <= 0:
            raise ValueError("density_bandwidth_factor must be positive.")
        if periodic_axes is not None:
            if not all(isinstance(axis, int) for axis in periodic_axes):
                raise TypeError("periodic_axes entries must be integers.")
            axes = tuple(periodic_axes)
            if len(set(axes)) != len(axes):
                raise ValueError("periodic_axes must not contain duplicates.")
            if any(axis < 0 or axis >= self.in_dim for axis in axes):
                raise ValueError(
                    f"periodic_axes entries must be in [0, {self.in_dim}), got {axes}."
                )
        else:
            axes = None
        self.metric = metric
        self.periodic_axes = axes
        self.density_bandwidth_factor = float(density_bandwidth_factor)

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
        self._q_ref: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype), requires_grad=False
        )
        self._q_density_ref: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype), requires_grad=False
        )

    def __repr__(self) -> str:
        return (
            f"KernelScDM(in_dim={self.in_dim}, eps={self.eps}, t={self.t}, "
            f"metric={self.metric!r}, dtype={self.dtype})"
        )

    @property
    def eps(self):  # eps > 0
        return F.softplus(self._log_eps)

    @property
    def t(self):  # t > 0
        return F.softplus(self._log_t)

    @property
    def density_eps(self) -> torch.Tensor:
        return self.eps * self.density_bandwidth_factor

    def _tiny(self, tensor: torch.Tensor) -> float:
        return torch.finfo(tensor.dtype).tiny

    def _floor_positive(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clamp_min(self._tiny(tensor))

    def _periodic_axis_tuple(self) -> tuple[int, ...]:
        if self.periodic_axes is None:
            return tuple(range(self.in_dim))
        return self.periodic_axes

    def _squared_distances(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if self.metric == "euclidean":
            return torch.cdist(X, Z, p=2) ** 2

        delta = torch.abs(X.unsqueeze(-2) - Z.unsqueeze(-3))
        axes = self._periodic_axis_tuple()
        if axes:
            wrapped_delta = delta.clone()
            axis_index = torch.tensor(axes, device=delta.device)
            periodic_delta = torch.remainder(delta.index_select(-1, axis_index), 1.0)
            periodic_delta = torch.minimum(periodic_delta, 1.0 - periodic_delta)
            wrapped_delta.index_copy_(-1, axis_index, periodic_delta)
            delta = wrapped_delta
        return (delta**2).sum(dim=-1)

    def raw_kernel(
        self, X: torch.Tensor, Z: torch.Tensor | None = None, *, eps: torch.Tensor | None = None
    ) -> torch.Tensor:
        if Z is None:
            Z = X
        if eps is None:
            eps = self.eps
        sq = self._squared_distances(X, Z)
        return torch.exp(-sq / (4.0 * eps))

    def _rbf(self, X, Z):
        return self.raw_kernel(X, Z)

    def _require_reference_data(self) -> None:
        if self._Xref.numel() == 0:
            raise RuntimeError("Call set_reference_data before evaluating reference sections.")

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
        q_ref = self._floor_positive(W.sum(dim=-1))
        _swap_parameter_storage(self._q_ref, q_ref, requires_grad=False)
        _swap_parameter_storage(self._D, q_ref ** (-self.t))
        W = self._D[..., None] * W * self._D[..., None, :]
        _swap_parameter_storage(self._Dinv1, self._floor_positive(W.sum(dim=-1)) ** (-0.5))

        W_density = self.raw_kernel(Xref, Xref, eps=self.density_eps)
        q_density_ref = self._floor_positive(W_density.sum(dim=-1))
        _swap_parameter_storage(self._q_density_ref, q_density_ref, requires_grad=False)

    def reference_row_sums(self, *, density_bandwidth: bool = False) -> torch.Tensor:
        self._require_reference_data()
        if density_bandwidth:
            return self._q_density_ref
        return self._q_ref

    def row_sums(self, X: torch.Tensor, *, density_bandwidth: bool = False) -> torch.Tensor:
        self._require_reference_data()
        eps = self.density_eps if density_bandwidth else self.eps
        return self._floor_positive(self.raw_kernel(X, self._Xref, eps=eps).sum(dim=-1))

    def _as_alpha(
        self, alpha: torch.Tensor | float | None, *, device: torch.device
    ) -> torch.Tensor:
        if alpha is None:
            return self.t
        return torch.as_tensor(alpha, dtype=self.dtype, device=device)

    def markov_sections(
        self, X: torch.Tensor, *, alpha: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        self._require_reference_data()
        alpha_tensor = self._as_alpha(alpha, device=X.device)
        q_x = self.row_sums(X, density_bandwidth=True)
        q_ref = self.reference_row_sums(density_bandwidth=True)
        W = self.raw_kernel(X, self._Xref)
        W = W / (q_x[..., None] ** alpha_tensor * q_ref**alpha_tensor)
        return W / self._floor_positive(W.sum(dim=-1))[..., None]

    def volume_weights(self) -> torch.Tensor:
        self._require_reference_data()
        q_ref = self.reference_row_sums(density_bandwidth=True)
        weights = q_ref.reciprocal()
        return weights / self._floor_positive(weights.sum())

    def density_sections(
        self, X: torch.Tensor, *, alpha: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        P = self.markov_sections(X, alpha=alpha)
        return P / self.volume_weights()

    def heat_diagnostics(self) -> dict[str, torch.Tensor]:
        self._require_reference_data()
        return {
            "eps": self.eps,
            "density_eps": self.density_eps,
            "q_ref": self.reference_row_sums(),
            "q_density_ref": self.reference_row_sums(density_bandwidth=True),
            "volume_weights": self.volume_weights(),
            "t": self.t,
        }

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
        D = self._floor_positive(W.sum(dim=-1)) ** (-self.t)
        W = D[..., None] * W * self._D[..., None, :]
        Dinv1 = self._floor_positive(W.sum(dim=-1)) ** (-0.5)
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
