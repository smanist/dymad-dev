from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn

from dymad.modules.helpers import _swap_parameter_storage
from dymad.modules.kernel import KernelBackend, KernelScDM


class KernelScDMHeat(KernelScDM):
    """Diffusion-map heat-section evaluator built on ``KernelScDM`` normalization."""

    def __init__(
        self,
        in_dim: int,
        eps_init: float | None = None,
        alpha_init: float = 1.0,
        dtype: torch.dtype | None = None,
        *,
        metric: str = "euclidean",
        periodic_axes: tuple[int, ...] | None = None,
        density_bandwidth_factor: float = 1.0,
        backend: KernelBackend = "torch",
    ):
        if density_bandwidth_factor <= 0:
            raise ValueError("density_bandwidth_factor must be positive.")
        super().__init__(
            in_dim=in_dim,
            eps_init=eps_init,
            alpha_init=alpha_init,
            dtype=dtype,
            metric=metric,
            periodic_axes=periodic_axes,
            backend=backend,
        )
        self.density_bandwidth_factor = float(density_bandwidth_factor)
        self._q_density_ref: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype), requires_grad=False
        )

    def __repr__(self) -> str:
        return (
            f"KernelScDMHeat(in_dim={self.in_dim}, eps={self.eps}, alpha={self.alpha}, "
            f"metric={self.metric!r}, dtype={self.dtype})"
        )

    @property
    def density_eps(self) -> torch.Tensor:
        return self.eps * self.density_bandwidth_factor

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        super().set_reference_data(Xref)
        if self.backend == "keops":
            q_density_ref = self._floor_positive(
                self._keops_kernel_sum(Xref, Xref, eps=self.density_eps)
            )
            _swap_parameter_storage(self._q_density_ref, q_density_ref, requires_grad=False)
            return

        w_density = self._raw_kernel(Xref, Xref, eps=self.density_eps)
        q_density_ref = self._floor_positive(w_density.sum(dim=-1))
        _swap_parameter_storage(self._q_density_ref, q_density_ref, requires_grad=False)

    def _density_reference_row_sums(self) -> torch.Tensor:
        self._require_reference_data()
        return self._q_density_ref

    def _density_row_sums(self, X: torch.Tensor) -> torch.Tensor:
        self._require_reference_data()
        if self.backend == "keops":
            if self.metric != "euclidean":
                raise NotImplementedError(
                    "KernelScDM backend='keops' currently supports only Euclidean inputs."
                )
            return self._floor_positive(self._keops_kernel_sum(X, self._Xref, eps=self.density_eps))
        return self._floor_positive(
            self._raw_kernel(X, self._Xref, eps=self.density_eps).sum(dim=-1)
        )

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
        """Estimate uniform-reference manifold volume from short-step row sums."""

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

    def _as_density_alpha(
        self, alpha: torch.Tensor | float | None, *, device: torch.device
    ) -> torch.Tensor:
        if alpha is None:
            return self.alpha
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
        q_x = self._density_row_sums(X)
        q_ref = self._density_reference_row_sums()
        return q_x.reciprocal() / self._floor_positive(q_ref.reciprocal().sum())

    def _density_markov_block(
        self, rows: torch.Tensor, cols: torch.Tensor, *, alpha: torch.Tensor
    ) -> torch.Tensor:
        q_rows = self._density_row_sums(rows)
        q_cols = self._density_row_sums(cols)
        q_ref = self._density_reference_row_sums()

        row_factor = q_rows[..., None] ** alpha
        ref_factor = q_ref**alpha
        row_ref = self._raw_kernel(rows, self._Xref) / (row_factor * ref_factor)
        normalizer = self._floor_positive(row_ref.sum(dim=-1))

        col_factor = q_cols[..., None, :] ** alpha
        block = self._raw_kernel(rows, cols) / (row_factor * col_factor)
        return block / normalizer[..., None]

    def _density_markov_apply(
        self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, *, alpha: torch.Tensor
    ) -> torch.Tensor:
        q_cols = self._density_row_sums(cols)
        q_ref = self._density_reference_row_sums()
        ref_weights = q_ref ** (-alpha)
        row_normalizer = self._floor_positive(
            self._keops_raw_apply(rows, self._Xref, ref_weights[:, None], eps=self.eps).squeeze(-1)
            if self.backend == "keops"
            else self._raw_kernel(rows, self._Xref) @ ref_weights
        )
        weighted = values / (q_cols**alpha)[..., :, None]
        if self.backend == "keops":
            applied = self._keops_raw_apply(rows, cols, weighted, eps=self.eps)
        else:
            applied = self._raw_kernel(rows, cols) @ weighted
        return applied / row_normalizer[..., :, None]

    def _density_markov_transpose_apply(
        self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, *, alpha: torch.Tensor
    ) -> torch.Tensor:
        q_cols = self._density_row_sums(cols)
        q_ref = self._density_reference_row_sums()
        ref_weights = q_ref ** (-alpha)
        row_normalizer = self._floor_positive(
            self._keops_raw_apply(rows, self._Xref, ref_weights[:, None], eps=self.eps).squeeze(-1)
            if self.backend == "keops"
            else self._raw_kernel(rows, self._Xref) @ ref_weights
        )
        weighted = values / row_normalizer[..., :, None]
        if self.backend == "keops":
            applied = self._keops_raw_apply(cols, rows, weighted, eps=self.eps)
        else:
            applied = self._raw_kernel(rows, cols).transpose(-1, -2) @ weighted
        return applied / (q_cols**alpha)[..., :, None]

    def _source_identity(self, sources: torch.Tensor) -> torch.Tensor:
        n_src = sources.shape[-2]
        eye = torch.eye(n_src, dtype=self.dtype, device=sources.device)
        if sources.ndim == 2:
            return eye
        return eye.expand(*sources.shape[:-2], n_src, n_src)

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
        if self.backend == "keops":
            return self._density_heat_kernel_by_step_keops(
                sources,
                locations,
                step_values,
                alpha=alpha,
                location_weights=location_weights,
                mass_normalization=mass_normalization,
            )
        return self._density_heat_kernel_by_step_dense(
            sources,
            locations,
            step_values,
            alpha=alpha,
            location_weights=location_weights,
            mass_normalization=mass_normalization,
        )

    def _density_heat_kernel_by_step_dense(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        step_values: tuple[int, ...],
        *,
        alpha: torch.Tensor | float | None,
        location_weights: torch.Tensor,
        mass_normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        alpha_tensor = self._as_density_alpha(alpha, device=sources.device)
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

    def _density_heat_kernel_by_step_keops(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        step_values: tuple[int, ...],
        *,
        alpha: torch.Tensor | float | None,
        location_weights: torch.Tensor,
        mass_normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        alpha_tensor = self._as_density_alpha(alpha, device=sources.device)
        target_weights = self._density_query_weights(locations)
        src_eye = self._source_identity(sources)
        result_by_step: dict[int, torch.Tensor] = {}
        if 1 in step_values:
            values = self._density_markov_transpose_apply(
                sources, locations, src_eye, alpha=alpha_tensor
            )
            values = values / target_weights[..., :, None]
            result_by_step[1] = self._normalize_heat(values, location_weights, mass_normalization)

        larger_steps = sorted({step for step in step_values if step >= 2})
        if not larger_steps:
            return result_by_step

        state = self._density_markov_transpose_apply(
            sources, self._Xref, src_eye, alpha=alpha_tensor
        )
        current_power = 0
        for step in larger_steps:
            target_power = step - 2
            while current_power < target_power:
                state = self._density_markov_transpose_apply(
                    self._Xref, self._Xref, state, alpha=alpha_tensor
                )
                current_power += 1
            values = self._density_markov_transpose_apply(
                self._Xref, locations, state, alpha=alpha_tensor
            )
            values = values / target_weights[..., :, None]
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
        if self.backend == "keops":
            return self._uniform_heat_kernel_by_step_keops(
                sources,
                locations,
                step_values,
                location_weights=location_weights,
                mass_normalization=mass_normalization,
            )
        return self._uniform_heat_kernel_by_step_dense(
            sources,
            locations,
            step_values,
            location_weights=location_weights,
            mass_normalization=mass_normalization,
        )

    def _uniform_heat_kernel_by_step_dense(
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

    def _uniform_heat_kernel_by_step_keops(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        step_values: tuple[int, ...],
        *,
        location_weights: torch.Tensor,
        mass_normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        scale = torch.as_tensor(self._Xref.shape[-2], dtype=self.dtype, device=self._Xref.device)
        d_src, s_src = self._uniform_factors(sources)
        d_loc, s_loc = self._uniform_factors(locations)
        src_eye = self._source_identity(sources)
        result_by_step: dict[int, torch.Tensor] = {}
        if 1 in step_values:
            values = scale * self._uniform_symmetric_apply(
                locations, sources, src_eye, d_loc, s_loc, d_src, s_src
            )
            result_by_step[1] = self._normalize_heat(values, location_weights, mass_normalization)

        larger_steps = sorted({step for step in step_values if step >= 2})
        if not larger_steps:
            return result_by_step

        state = self._uniform_symmetric_apply(
            self._Xref, sources, src_eye, self._D, self._Dinv1, d_src, s_src
        )
        current_power = 0
        for step in larger_steps:
            target_power = step - 2
            while current_power < target_power:
                state = self._uniform_symmetric_apply(
                    self._Xref, self._Xref, state, self._D, self._Dinv1, self._D, self._Dinv1
                )
                current_power += 1
            values = scale * self._uniform_symmetric_apply(
                locations, self._Xref, state, d_loc, s_loc, self._D, self._Dinv1
            )
            result_by_step[step] = self._normalize_heat(
                values, location_weights, mass_normalization
            )
        return result_by_step
