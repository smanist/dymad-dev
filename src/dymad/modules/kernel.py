from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

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
            self._log_eps = nn.Parameter(inv_softplus(eps_init, self.dtype))
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

    def _reference_row_sums(self, *, density_bandwidth: bool = False) -> torch.Tensor:
        self._require_reference_data()
        if density_bandwidth:
            return self._q_density_ref
        return self._q_ref

    def _row_sums(self, X: torch.Tensor, *, density_bandwidth: bool = False) -> torch.Tensor:
        self._require_reference_data()
        eps = self.density_eps if density_bandwidth else self.eps
        return self._floor_positive(self.raw_kernel(X, self._Xref, eps=eps).sum(dim=-1))

    def estimate_reference_volume(
        self,
        dim: int,
        *,
        method: Literal["median", "mean"] = "median",
        warn: bool = True,
        row_sum_cv_warn: float = 0.25,
        row_sum_spread_warn: float = 2.0,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int | str]]:
        """Estimate uniform-reference manifold volume from short-step row sums.

        The estimator assumes approximately uniform reference samples and uses
        ``sum_j exp(-||x_i-x_j||^2/(4 eps))``. Large row-sum variation triggers
        a warning because it often indicates non-uniform sampling, boundary
        bias, or an unsuitable bandwidth.
        """

        self._require_reference_data()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if method not in {"median", "mean"}:
            raise ValueError("method must be either 'median' or 'mean'.")

        q_ref = self._reference_row_sums()
        center = torch.median(q_ref) if method == "median" else torch.mean(q_ref)
        factor = torch.as_tensor(4.0 * np.pi, dtype=self.dtype, device=q_ref.device) * self.eps
        volume = q_ref.shape[-1] * factor ** (0.5 * dim) / self._floor_positive(center)

        q_mean = torch.mean(q_ref)
        q_std = torch.std(q_ref, unbiased=False)
        q_p05 = torch.quantile(q_ref, 0.05)
        q_p95 = torch.quantile(q_ref, 0.95)
        row_sum_cv = q_std / self._floor_positive(q_mean)
        row_sum_spread = q_p95 / self._floor_positive(q_p05)
        row_sum_cv_value = float(row_sum_cv.detach().cpu())
        row_sum_spread_value = float(row_sum_spread.detach().cpu())
        diagnostics: dict[str, float | int | str] = {
            "dim": dim,
            "method": method,
            "volume": float(volume.detach().cpu()),
            "row_sum_mean": float(q_mean.detach().cpu()),
            "row_sum_median": float(torch.median(q_ref).detach().cpu()),
            "row_sum_cv": row_sum_cv_value,
            "row_sum_p05": float(q_p05.detach().cpu()),
            "row_sum_p95": float(q_p95.detach().cpu()),
            "row_sum_p95_p05": row_sum_spread_value,
        }
        if warn and (
            row_sum_cv_value > row_sum_cv_warn or row_sum_spread_value > row_sum_spread_warn
        ):
            warnings.warn(
                "Reference row sums vary substantially "
                f"(cv={row_sum_cv_value:.3g}, "
                f"p95/p5={row_sum_spread_value:.3g}). "
                "The volume estimator assumes approximately uniform reference "
                "sampling and weak boundary bias.",
                RuntimeWarning,
                stacklevel=2,
            )

        if return_diagnostics:
            return volume, diagnostics
        return volume

    def _as_alpha(
        self, alpha: torch.Tensor | float | None, *, device: torch.device
    ) -> torch.Tensor:
        if alpha is None:
            return self.t
        return torch.as_tensor(alpha, dtype=self.dtype, device=device)

    def _heat_step_values(self, steps: int | Sequence[int]) -> tuple[tuple[int, ...], bool]:
        if isinstance(steps, int):
            step_values = (steps,)
            return_single = True
        elif isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            step_values = tuple(int(step) for step in steps)
            return_single = False
        else:
            raise TypeError("steps must be an int or a sequence of ints.")

        if not step_values:
            raise ValueError("steps must contain at least one value.")
        if any(step < 1 for step in step_values):
            raise ValueError("steps values must be positive.")
        return step_values, return_single

    def _as_heat_points(self, values: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=self.dtype, device=self._Xref.device)
        if tensor.ndim < 2 or tensor.shape[-1] != self.in_dim:
            raise ValueError(f"{name} must have shape (..., N, {self.in_dim}).")
        return tensor

    def _density_query_weights(self, X: torch.Tensor) -> torch.Tensor:
        q_x = self._row_sums(X, density_bandwidth=True)
        q_ref = self._reference_row_sums(density_bandwidth=True)
        return q_x.reciprocal() / self._floor_positive(q_ref.reciprocal().sum())

    def _density_markov_block(
        self, rows: torch.Tensor, cols: torch.Tensor, *, alpha: torch.Tensor
    ) -> torch.Tensor:
        q_rows = self._row_sums(rows, density_bandwidth=True)
        q_cols = self._row_sums(cols, density_bandwidth=True)
        q_ref = self._reference_row_sums(density_bandwidth=True)

        row_factor = q_rows[..., None] ** alpha
        ref_factor = q_ref**alpha
        row_ref = self.raw_kernel(rows, self._Xref) / (row_factor * ref_factor)
        normalizer = self._floor_positive(row_ref.sum(dim=-1))

        col_factor = q_cols[..., None, :] ** alpha
        block = self.raw_kernel(rows, cols) / (row_factor * col_factor)
        return block / normalizer[..., None]

    def _uniform_factors(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if X.data_ptr() == self._Xref.data_ptr():
            return self._D, self._Dinv1

        q_x = self._row_sums(X)
        d_x = q_x ** (-self.t)
        row_ref = d_x[..., None] * self.raw_kernel(X, self._Xref) * self._D
        s_x = self._floor_positive(row_ref.sum(dim=-1)) ** (-0.5)
        return d_x, s_x

    def _uniform_symmetric_block(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        d_rows, s_rows = self._uniform_factors(rows)
        d_cols, s_cols = self._uniform_factors(cols)
        block = self.raw_kernel(rows, cols)
        block = d_rows[..., None] * block * d_cols[..., None, :]
        return s_rows[..., None] * block * s_cols[..., None, :]

    def _heat_location_weights(
        self,
        locations: torch.Tensor,
        weights: torch.Tensor | None,
        *,
        volume_normalization: Literal["none", "explicit_volume", "estimate_volume"],
        volume: torch.Tensor | float | None,
        volume_dim: int | None,
        volume_estimate_warnings: bool,
        volume_row_sum_cv_warn: float,
        volume_row_sum_spread_warn: float,
    ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
        if weights is not None and volume_normalization != "none":
            raise ValueError("location_weights cannot be combined with volume_normalization.")

        if volume_normalization == "explicit_volume":
            if volume is None:
                raise ValueError("volume is required when volume_normalization='explicit_volume'.")
            volume_tensor = torch.as_tensor(volume, dtype=self.dtype, device=locations.device)
            if torch.any(volume_tensor <= 0):
                raise ValueError("volume must be positive.")
            diagnostics: dict[str, float | int | str] = {"volume_normalization": "explicit_volume"}
            if volume_tensor.numel() == 1:
                diagnostics["volume"] = float(volume_tensor.detach().cpu())
            else:
                diagnostics["volume_shape"] = str(tuple(volume_tensor.shape))
            location_weights = torch.ones(
                locations.shape[:-1], dtype=self.dtype, device=locations.device
            ) * (volume_tensor / locations.shape[-2])
            return location_weights, diagnostics

        if volume_normalization == "estimate_volume":
            if volume_dim is None:
                raise ValueError(
                    "volume_dim is required when volume_normalization='estimate_volume'."
                )
            volume_tensor, diagnostics = cast(
                tuple[torch.Tensor, dict[str, float | int | str]],
                self.estimate_reference_volume(
                    volume_dim,
                    warn=volume_estimate_warnings,
                    row_sum_cv_warn=volume_row_sum_cv_warn,
                    row_sum_spread_warn=volume_row_sum_spread_warn,
                    return_diagnostics=True,
                ),
            )
            diagnostics = {"volume_normalization": "estimate_volume", **diagnostics}
            location_weights = torch.ones(
                locations.shape[:-1], dtype=self.dtype, device=locations.device
            ) * (volume_tensor.to(locations.device) / locations.shape[-2])
            return location_weights, diagnostics

        if volume_normalization != "none":
            raise ValueError(
                "volume_normalization must be one of 'none', 'explicit_volume', "
                "or 'estimate_volume'."
            )

        if weights is None:
            return (
                torch.full(
                    locations.shape[:-1],
                    1.0 / locations.shape[-2],
                    dtype=self.dtype,
                    device=locations.device,
                ),
                {"volume_normalization": "none"},
            )

        weight_tensor = torch.as_tensor(weights, dtype=self.dtype, device=locations.device)
        if weight_tensor.ndim < 1 or weight_tensor.shape[-1] != locations.shape[-2]:
            raise ValueError("location_weights must have shape (..., Nloc).")
        return weight_tensor, {"volume_normalization": "none", "location_weights": "explicit"}

    def _normalize_heat(
        self,
        values: torch.Tensor,
        location_weights: torch.Tensor,
        normalization: Literal["source", "median", "none"],
    ) -> torch.Tensor:
        if normalization == "none":
            return values
        if normalization not in {"source", "median"}:
            raise ValueError("mass_normalization must be one of 'source', 'median', or 'none'.")

        mass = (values * location_weights[..., None]).sum(dim=-2)
        if normalization == "source":
            return values / self._floor_positive(mass)[..., None, :]

        scale = torch.median(mass.reshape(-1))
        return values / self._floor_positive(scale)

    def heat_kernel(
        self,
        locations: torch.Tensor,
        sources: torch.Tensor | None = None,
        *,
        mode: Literal["density", "uniform"] = "density",
        steps: int | Sequence[int] = 1,
        alpha: torch.Tensor | float | None = None,
        location_weights: torch.Tensor | None = None,
        mass_normalization: Literal["source", "median", "none"] = "source",
        volume_normalization: Literal["none", "explicit_volume", "estimate_volume"] = "none",
        volume: torch.Tensor | float | None = None,
        volume_dim: int | None = None,
        volume_estimate_warnings: bool = True,
        volume_row_sum_cv_warn: float = 0.25,
        volume_row_sum_spread_warn: float = 2.0,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int | str]]:
        """
        Evaluate diffusion-map heat kernels from sources to locations.

        The returned tensor is indexed as ``(..., Nloc, Nsrc)``. With a sequence of
        ``steps`` values, the result is stacked as ``(Nsteps, ..., Nloc, Nsrc)`` in
        the requested order. ``steps=1`` is the direct source-location kernel;
        larger values route through the reference data with ``steps - 2``
        reference-reference transitions.

        In ``density`` mode, directed Markov blocks are composed and the final
        target block is divided by query volume weights. In ``uniform`` mode,
        symmetric diffusion-map blocks are composed and scaled by the number of
        reference points. Both modes are mass-normalized on the returned location
        grid by default. ``location_weights`` supplies the quadrature weights for
        that grid; equal weights are used when it is omitted. Alternatively,
        ``volume_normalization`` can convert equal location weights to an
        explicit or estimated total volume, in which case the returned section
        is already scaled against that volume. Set ``return_diagnostics=True``
        to return the volume metadata used for this scaling.
        """
        self._require_reference_data()
        step_values, return_single = self._heat_step_values(steps)
        locations_tensor = self._as_heat_points(locations, name="locations")
        if sources is None:
            sources_tensor = locations_tensor
        else:
            sources_tensor = self._as_heat_points(sources, name="sources")

        mode_key = mode.lower()
        if volume_normalization == "estimate_volume" and mode_key != "uniform":
            raise ValueError(
                "volume_normalization='estimate_volume' is only supported in uniform mode."
            )
        weights, diagnostics = self._heat_location_weights(
            locations_tensor,
            location_weights,
            volume_normalization=volume_normalization,
            volume=volume,
            volume_dim=volume_dim,
            volume_estimate_warnings=volume_estimate_warnings,
            volume_row_sum_cv_warn=volume_row_sum_cv_warn,
            volume_row_sum_spread_warn=volume_row_sum_spread_warn,
        )
        if mode_key == "density":
            result_by_step = self._density_heat_kernel_by_step(
                sources_tensor,
                locations_tensor,
                step_values,
                alpha=alpha,
                location_weights=weights,
                mass_normalization=mass_normalization,
            )
        elif mode_key == "uniform":
            if alpha is not None:
                raise ValueError("alpha is only supported for density heat kernels.")
            result_by_step = self._uniform_heat_kernel_by_step(
                sources_tensor,
                locations_tensor,
                step_values,
                location_weights=weights,
                mass_normalization=mass_normalization,
            )
        else:
            raise ValueError("mode must be either 'density' or 'uniform'.")

        ordered = [result_by_step[step] for step in step_values]
        if return_single:
            result = ordered[0]
        else:
            result = torch.stack(ordered, dim=0)
        if return_diagnostics:
            return result, diagnostics
        return result

    def _density_heat_kernel_by_step(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        step_values: tuple[int, ...],
        *,
        alpha: torch.Tensor | float | None,
        location_weights: torch.Tensor,
        mass_normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        alpha_tensor = self._as_alpha(alpha, device=sources.device)
        target_weights = self._density_query_weights(locations)

        direct = self._density_markov_block(sources, locations, alpha=alpha_tensor)
        direct = direct / target_weights[..., None, :]
        result_by_step: dict[int, torch.Tensor] = {}
        if 1 in step_values:
            values = direct.transpose(-1, -2)
            result_by_step[1] = self._normalize_heat(values, location_weights, mass_normalization)

        larger_steps = sorted({step for step in step_values if step >= 2})
        if not larger_steps:
            return result_by_step

        source_ref = self._density_markov_block(sources, self._Xref, alpha=alpha_tensor)
        ref_ref = self._density_markov_block(self._Xref, self._Xref, alpha=alpha_tensor)
        ref_location = self._density_markov_block(self._Xref, locations, alpha=alpha_tensor)

        current = source_ref
        current_power = 0
        for step in larger_steps:
            target_power = step - 2
            while current_power < target_power:
                current = torch.matmul(current, ref_ref)
                current_power += 1
            values = torch.matmul(current, ref_location)
            values = values / target_weights[..., None, :]
            values = values.transpose(-1, -2)
            result_by_step[step] = self._normalize_heat(
                values, location_weights, mass_normalization
            )
        return result_by_step

    def _uniform_heat_kernel_by_step(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        step_values: tuple[int, ...],
        *,
        location_weights: torch.Tensor,
        mass_normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        scale = torch.as_tensor(self._Xref.shape[-2], dtype=self.dtype, device=self._Xref.device)
        result_by_step: dict[int, torch.Tensor] = {}
        if 1 in step_values:
            direct = scale * self._uniform_symmetric_block(sources, locations)
            values = direct.transpose(-1, -2)
            result_by_step[1] = self._normalize_heat(values, location_weights, mass_normalization)

        larger_steps = sorted({step for step in step_values if step >= 2})
        if not larger_steps:
            return result_by_step

        source_ref = self._uniform_symmetric_block(sources, self._Xref)
        ref_ref = self._uniform_symmetric_block(self._Xref, self._Xref)
        ref_location = self._uniform_symmetric_block(self._Xref, locations)

        current = source_ref
        current_power = 0
        for step in larger_steps:
            target_power = step - 2
            while current_power < target_power:
                current = torch.matmul(current, ref_ref)
                current_power += 1
            values = scale * torch.matmul(current, ref_location)
            values = values.transpose(-1, -2)
            result_by_step[step] = self._normalize_heat(
                values, location_weights, mass_normalization
            )
        return result_by_step

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
