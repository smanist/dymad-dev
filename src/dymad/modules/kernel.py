from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dymad.modules.helpers import _swap_parameter_storage
from dymad.numerics import DimensionEstimator

logger = logging.getLogger(__name__)

KernelBackend = Literal["torch", "keops"]
_KEOPS_CACHE = Path(tempfile.gettempdir()) / "dymad_keops_cache"


def _validate_backend(backend: str) -> KernelBackend:
    if backend not in {"torch", "keops"}:
        raise ValueError("backend must be either 'torch' or 'keops'.")
    return cast(KernelBackend, backend)


@cache
def _lazy_tensor_cls() -> Any:
    _KEOPS_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))
    os.environ.setdefault("PYKEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))
    try:
        from pykeops.torch import LazyTensor
    except ImportError as exc:
        raise ImportError(
            "PyKeOps is required for backend='keops'. Install DyMAD with the "
            "'keops' extra or install pykeops."
        ) from exc
    return LazyTensor


def _keops_weighted_exp_sum(
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    exponent: Any,
) -> torch.Tensor:
    squeeze_result = values.ndim == 1
    if squeeze_result:
        values = values[:, None]
    if rows.ndim < 2 or cols.ndim < 2 or values.ndim < 2:
        raise ValueError(
            "KeOps reductions require inputs shaped (..., N, d), (..., M, d), and (..., M, R)."
        )
    if rows.shape[-1] != cols.shape[-1]:
        raise ValueError("KeOps reduction row and column point dimensions must match.")
    if values.shape[-2] != cols.shape[-2]:
        raise ValueError("KeOps reduction values must have the same column count as cols.")

    batch_shape = torch.broadcast_shapes(rows.shape[:-2], cols.shape[:-2], values.shape[:-2])
    rows = rows.expand(*batch_shape, *rows.shape[-2:])
    cols = cols.expand(*batch_shape, *cols.shape[-2:])
    values = values.expand(*batch_shape, *values.shape[-2:])

    LazyTensor = _lazy_tensor_cls()
    rows_i = LazyTensor(rows[..., :, None, :])
    cols_j = LazyTensor(cols[..., None, :, :])
    values_j = LazyTensor(values[..., None, :, :])
    kernel = exponent(rows_i, cols_j).exp()
    result = cast(torch.Tensor, (kernel * values_j).sum(axis=rows.ndim - 1))
    return result.squeeze(-1) if squeeze_result else result


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


def inv_softplus(
    y: float | np.floating[Any] | torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Inverse of softplus, for initialization."""
    y_tensor = torch.as_tensor(y, dtype=dtype, device=device)
    return torch.where(y_tensor > 20.0, y_tensor, torch.log(torch.expm1(y_tensor)))


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

    def require_fixed_parameters(self) -> None:
        """Raise when analysis would need to initialize kernel parameters.

        Generic kernels are assumed to be fully configured. Kernels with
        optional data-driven initialization override this hook so analysis
        objects never tune a kernel as a side effect of preparing references.
        """

        return None

    def materialize(self, X: torch.Tensor, Z: torch.Tensor | None = None) -> torch.Tensor:
        """Explicitly materialize the kernel block."""
        return self.forward(X, Z)

    def apply(self, X: torch.Tensor, Z: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        """Apply ``K(X, Z)`` to ``values``.

        The default implementation materializes the block. Kernels with
        backend-specific reductions override this method.
        """
        return self.materialize(X, Z) @ values


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
        self,
        in_dim: int,
        lengthscale_init: float | None = None,
        dtype: torch.dtype | None = None,
        *,
        backend: KernelBackend = "torch",
    ):
        super().__init__(in_dim, dtype=dtype)
        self.backend = _validate_backend(backend)
        if lengthscale_init is None:
            self._log_ell: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_ell = nn.Parameter(inv_softplus(lengthscale_init, self.dtype))

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

    def require_fixed_parameters(self) -> None:
        if self._log_ell.numel() == 0:
            raise RuntimeError("KernelScRBF requires lengthscale_init before kernel analysis.")

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        sq = scaled_cdist(X, Z, self.ell, 2) ** 2
        return torch.exp(-0.5 * sq)

    def apply(self, X: torch.Tensor, Z: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        if Z is None:
            Z = X
        if self.backend == "torch":
            return super().apply(X, Z, values)
        rows = torch.as_tensor(X, dtype=self.dtype)
        cols = torch.as_tensor(Z, dtype=self.dtype, device=rows.device)
        weights = torch.as_tensor(values, dtype=self.dtype, device=cols.device)

        def exponent(rows_i: Any, cols_j: Any) -> Any:
            delta = (rows_i - cols_j) / self.ell
            return -0.5 * delta.sqnorm2()

        return _keops_weighted_exp_sum(rows, cols, weights, exponent)


class KernelScExp(KernelScalarValued):
    """
    Scalar Exponential: k(x,z) = exp(-||x - z|| / ell)
    Learnable positive lengthscale.
    """

    def __init__(
        self,
        in_dim: int,
        lengthscale_init: float | None = None,
        dtype: torch.dtype | None = None,
        *,
        backend: KernelBackend = "torch",
    ):
        super().__init__(in_dim, dtype=dtype)
        self.backend = _validate_backend(backend)
        if lengthscale_init is None:
            self._log_ell: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_ell = nn.Parameter(inv_softplus(lengthscale_init, self.dtype))

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

    def require_fixed_parameters(self) -> None:
        if self._log_ell.numel() == 0:
            raise RuntimeError("KernelScExp requires lengthscale_init before kernel analysis.")

    def apply(self, X: torch.Tensor, Z: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        if Z is None:
            Z = X
        if self.backend == "torch":
            return super().apply(X, Z, values)
        rows = torch.as_tensor(X, dtype=self.dtype)
        cols = torch.as_tensor(Z, dtype=self.dtype, device=rows.device)
        weights = torch.as_tensor(values, dtype=self.dtype, device=cols.device)

        def exponent(rows_i: Any, cols_j: Any) -> Any:
            sq = ((rows_i - cols_j) / self.ell).sqnorm2()
            return -(sq + 1.0e-300).sqrt()

        return _keops_weighted_exp_sum(rows, cols, weights, exponent)


class KernelScDM(KernelScalarValued):
    """
    Symmetric-normalized diffusion kernel via diffusion maps.

    Everything keeps autograd for eps and alpha.
    """

    def __init__(
        self,
        in_dim: int,
        eps_init: float | None = None,
        alpha_init: float = 1.0,
        dtype: torch.dtype | None = None,
        *,
        metric: str = "euclidean",
        periodic_axes: tuple[int, ...] | None = None,
        backend: KernelBackend = "torch",
    ):
        super().__init__(in_dim, dtype=dtype)
        self.backend = _validate_backend(backend)
        if metric not in {"euclidean", "periodic"}:
            raise ValueError("metric must be either 'euclidean' or 'periodic'.")
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

        if eps_init is None:
            self._log_eps: nn.Parameter = nn.Parameter(torch.empty(0, dtype=self.dtype))
        else:
            self._log_eps = nn.Parameter(inv_softplus(eps_init, self.dtype))
        _tmp = inv_softplus(alpha_init, self.dtype)
        self._log_alpha: nn.Parameter = nn.Parameter(_tmp)

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

    def __repr__(self) -> str:
        return (
            f"KernelScDM(in_dim={self.in_dim}, eps={self.eps}, alpha={self.alpha}, "
            f"metric={self.metric!r}, dtype={self.dtype})"
        )

    @property
    def eps(self) -> torch.Tensor:  # eps > 0
        return cast(torch.Tensor, F.softplus(self._log_eps))

    @property
    def alpha(self) -> torch.Tensor:  # alpha > 0
        return cast(torch.Tensor, F.softplus(self._log_alpha))

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

    def _raw_kernel(
        self, X: torch.Tensor, Z: torch.Tensor | None = None, *, eps: torch.Tensor | None = None
    ) -> torch.Tensor:
        if Z is None:
            Z = X
        bandwidth = self.eps if eps is None else eps
        sq = self._squared_distances(X, Z)
        return torch.exp(-sq / (4.0 * bandwidth))

    def _rbf(self, X, Z):
        return self._raw_kernel(X, Z)

    def _require_reference_data(self) -> None:
        if self._Xref.numel() == 0:
            raise RuntimeError(
                "Call set_reference_data before evaluating reference-dependent kernels."
            )

    def require_fixed_parameters(self) -> None:
        if self._log_eps.numel() == 0:
            raise RuntimeError("KernelScDM requires eps_init before kernel analysis.")

    @property
    def reference_points(self) -> torch.Tensor:
        """Reference points prepared by :meth:`set_reference_data`."""

        self._require_reference_data()
        return self._Xref

    @property
    def reference_row_sums(self) -> torch.Tensor:
        """Raw-kernel row sums at the prepared reference points."""

        return self._reference_row_sums()

    def raw_apply(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        values: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the unnormalized diffusion kernel without materializing it."""

        if eps is None:
            eps = self.eps
        if self.backend == "keops":
            if self.metric != "euclidean":
                raise NotImplementedError(
                    "KernelScDM backend='keops' currently supports only Euclidean inputs."
                )
            return self._keops_raw_apply(rows, cols, values, eps=eps)
        return self._raw_kernel(rows, cols, eps=eps) @ values

    def raw_row_sums(self, rows: torch.Tensor, *, eps: torch.Tensor | None = None) -> torch.Tensor:
        """Return unnormalized row sums against the prepared references."""

        self._require_reference_data()
        if eps is None:
            eps = self.eps
        ones = torch.ones((*self._Xref.shape[:-1], 1), dtype=self.dtype, device=self._Xref.device)
        return self._floor_positive(self.raw_apply(rows, self._Xref, ones, eps=eps).squeeze(-1))

    def uniform_factors(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return density and symmetric-normalization factors for ``X``."""

        return self._uniform_factors(X)

    def uniform_symmetric_apply(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        values: torch.Tensor,
        d_rows: torch.Tensor | None = None,
        s_rows: torch.Tensor | None = None,
        d_cols: torch.Tensor | None = None,
        s_cols: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the symmetric-normalized diffusion kernel."""

        return self._uniform_symmetric_apply(rows, cols, values, d_rows, s_rows, d_cols, s_cols)

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

        if self.backend == "keops":
            if self.metric != "euclidean":
                raise NotImplementedError(
                    "KernelScDM backend='keops' currently supports only Euclidean inputs."
                )
            q_ref = self._floor_positive(self._keops_kernel_sum(Xref, Xref, eps=self.eps))
            _swap_parameter_storage(self._q_ref, q_ref, requires_grad=False)
            d_ref = q_ref ** (-self.alpha)
            _swap_parameter_storage(self._D, d_ref, requires_grad=False)
            ref_sum = self._keops_raw_apply(Xref, Xref, d_ref[:, None], eps=self.eps).squeeze(-1)
            s_ref = self._floor_positive(d_ref * ref_sum) ** (-0.5)
            _swap_parameter_storage(self._Dinv1, s_ref, requires_grad=False)
            return

        W = self._rbf(Xref, Xref)
        q_ref = self._floor_positive(W.sum(dim=-1))
        _swap_parameter_storage(self._q_ref, q_ref, requires_grad=False)
        _swap_parameter_storage(self._D, q_ref ** (-self.alpha))
        W = self._D[..., None] * W * self._D[..., None, :]
        _swap_parameter_storage(self._Dinv1, self._floor_positive(W.sum(dim=-1)) ** (-0.5))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name in ("_Xref", "_D", "_Dinv1", "_q_ref"):
            saved = state_dict.get(prefix + name)
            current = cast(nn.Parameter, getattr(self, name))
            if saved is not None and current.shape != saved.shape:
                current.data = torch.empty(saved.shape, dtype=current.dtype, device=current.device)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _reference_row_sums(self) -> torch.Tensor:
        self._require_reference_data()
        return self._q_ref

    def _row_sums(self, X: torch.Tensor) -> torch.Tensor:
        self._require_reference_data()
        if self.backend == "keops":
            if self.metric != "euclidean":
                raise NotImplementedError(
                    "KernelScDM backend='keops' currently supports only Euclidean inputs."
                )
            return self._floor_positive(self._keops_kernel_sum(X, self._Xref, eps=self.eps))
        return self._floor_positive(self._raw_kernel(X, self._Xref).sum(dim=-1))

    def _uniform_factors(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if X.data_ptr() == self._Xref.data_ptr() and X.shape == self._Xref.shape:
            return self._D, self._Dinv1

        q_x = self._row_sums(X)
        d_x = q_x ** (-self.alpha)
        if self.backend == "keops":
            ref_sum = self._keops_raw_apply(X, self._Xref, self._D[:, None], eps=self.eps).squeeze(
                -1
            )
            s_x = self._floor_positive(d_x * ref_sum) ** (-0.5)
            return d_x, s_x
        row_ref = d_x[..., None] * self._raw_kernel(X, self._Xref) * self._D
        s_x = self._floor_positive(row_ref.sum(dim=-1)) ** (-0.5)
        return d_x, s_x

    def _uniform_symmetric_block(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        d_rows, s_rows = self._uniform_factors(rows)
        d_cols, s_cols = self._uniform_factors(cols)
        if self.backend == "keops":
            n_cols = cols.shape[-2]
            eye = torch.eye(n_cols, dtype=self.dtype, device=cols.device)
            if cols.ndim > 2:
                eye = eye.expand(*cols.shape[:-2], n_cols, n_cols)
            return self._uniform_symmetric_apply(rows, cols, eye, d_rows, s_rows, d_cols, s_cols)
        block = self._raw_kernel(rows, cols)
        block = d_rows[..., None] * block * d_cols[..., None, :]
        return s_rows[..., None] * block * s_cols[..., None, :]

    def _uniform_symmetric_apply(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        values: torch.Tensor,
        d_rows: torch.Tensor | None = None,
        s_rows: torch.Tensor | None = None,
        d_cols: torch.Tensor | None = None,
        s_cols: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze_result = values.ndim == 1
        if squeeze_result:
            values = values[:, None]
        if d_rows is None or s_rows is None:
            d_rows, s_rows = self._uniform_factors(rows)
        if d_cols is None or s_cols is None:
            d_cols, s_cols = self._uniform_factors(cols)
        assert d_rows is not None and s_rows is not None
        assert d_cols is not None and s_cols is not None
        weighted = (d_cols * s_cols)[..., :, None] * values
        if self.backend == "keops":
            summed = self._keops_raw_apply(rows, cols, weighted, eps=self.eps)
        else:
            summed = self._raw_kernel(rows, cols) @ weighted
        result = (d_rows * s_rows)[..., :, None] * summed
        return result.squeeze(-1) if squeeze_result else result

    def forward(self, X: torch.Tensor, Z: torch.Tensor | None = None):
        reference = self._Xref if Z is None else Z

        if (
            X.data_ptr() == reference.data_ptr()
            and X.data_ptr() == self._Xref.data_ptr()
            and X.shape == self._Xref.shape
            and reference.shape == self._Xref.shape
        ):
            # K(X,X) with reference data, use cached
            W = self._rbf(X, X)
            W = self._D[..., None] * W * self._D[..., None, :]
            W = self._Dinv1[..., None] * W * self._Dinv1[..., None, :]
            return W

        W = self._rbf(X, reference)
        D = self._floor_positive(W.sum(dim=-1)) ** (-self.alpha)
        W = D[..., None] * W * self._D[..., None, :]
        Dinv1 = self._floor_positive(W.sum(dim=-1)) ** (-0.5)
        W = Dinv1[..., None] * W * self._Dinv1[..., None, :]
        return W

    def apply(self, X: torch.Tensor, Z: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        if Z is None:
            Z = self._Xref
        rows = torch.as_tensor(X, dtype=self.dtype, device=self._Xref.device)
        cols = torch.as_tensor(Z, dtype=self.dtype, device=self._Xref.device)
        weights = torch.as_tensor(values, dtype=self.dtype, device=self._Xref.device)
        if self.backend == "torch":
            return self.forward(rows, cols) @ weights
        if self.metric != "euclidean":
            raise NotImplementedError(
                "KernelScDM backend='keops' currently supports only Euclidean inputs."
            )
        return self._uniform_symmetric_apply(rows, cols, weights)

    def _keops_kernel_sum(
        self, rows: torch.Tensor, cols: torch.Tensor, *, eps: torch.Tensor
    ) -> torch.Tensor:
        ones = torch.ones((*cols.shape[:-1], 1), dtype=self.dtype, device=cols.device)
        return self._keops_raw_apply(rows, cols, ones, eps=eps).squeeze(-1)

    def _keops_raw_apply(
        self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, *, eps: torch.Tensor
    ) -> torch.Tensor:
        rows_tensor = torch.as_tensor(rows, dtype=self.dtype, device=self._Xref.device)
        cols_tensor = torch.as_tensor(cols, dtype=self.dtype, device=rows_tensor.device)
        values_tensor = torch.as_tensor(values, dtype=self.dtype, device=cols_tensor.device)

        def exponent(rows_i: Any, cols_j: Any) -> Any:
            return -(rows_i - cols_j).sqnorm2() / (4.0 * eps)

        return _keops_weighted_exp_sum(rows_tensor, cols_tensor, values_tensor, exponent)


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

    def apply(self, X: torch.Tensor, Z: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        if Z is None:
            Z = X
        L = torch.tril(self.Ls)
        B = torch.matmul(L, L.transpose(-1, -2))
        result = torch.zeros(
            (X.shape[-2], self.out_dim),
            dtype=self.dtype,
            device=torch.as_tensor(X).device,
        )
        for idx, kernel in enumerate(self.scalar_kernels):
            weighted = values @ B[idx].T
            result = result + cast(KernelScalarValued, kernel).apply(X, Z, weighted)
        return result


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
        out = torch.einsum("... n a i, ... m b i, ... n m -> ... n a m b", _Tx, _Tz, k)

        return out, _Tx, _Tz

    def intrinsic_apply(
        self,
        X: torch.Tensor,
        Z: torch.Tensor | None,
        values: torch.Tensor,
        *,
        Tx: torch.Tensor | None = None,
        Tz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if Z is None:
            Z = X
        if Tx is None:
            Tx = self._tangent(X)
        if Tz is None:
            Tz = self._tangent(Z)
        ambient_values = torch.einsum("... m b i, ... m b -> ... m i", Tz, values)
        ambient_result = self.scalar_kernel.apply(X, Z, ambient_values)
        return torch.einsum("... n a i, ... n i -> ... n a", Tx, ambient_result), Tx

    def apply(self, X: torch.Tensor, Z: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        if Z is None:
            Z = X
        Tx = self._tangent(X)
        Tz = self._tangent(Z)
        intrinsic_values = torch.einsum("... m b i, ... m i -> ... m b", Tz, values)
        intrinsic_result, _ = self.intrinsic_apply(X, Z, intrinsic_values, Tx=Tx, Tz=Tz)
        return torch.einsum("... n a, ... n a i -> ... n i", intrinsic_result, Tx)
