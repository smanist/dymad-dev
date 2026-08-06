"""Diffusion-map heat sections built from a diffusion kernel."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn

from dymad.modules.kernel import KernelScDM


class DiffusionHeatSections(nn.Module):
    """Evaluate heat sections using a composed :class:`KernelScDM`.

    ``kernel`` is the reusable KRR-capable diffusion kernel. This class owns
    only reference-dependent heat analysis state and is not itself a kernel.
    ``in_dim`` and the remaining kernel options are accepted as a concise
    convenience constructor for standalone analysis scripts.
    """

    def __init__(
        self,
        kernel: KernelScDM | None = None,
        *,
        in_dim: int | None = None,
        eps_init: float | None = None,
        alpha_init: float = 1.0,
        dtype: torch.dtype | None = None,
        metric: str = "euclidean",
        periodic_axes: tuple[int, ...] | None = None,
        density_bandwidth_factor: float = 1.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if density_bandwidth_factor <= 0:
            raise ValueError("density_bandwidth_factor must be positive.")
        if kernel is None:
            if in_dim is None:
                raise TypeError("kernel or in_dim is required.")
            kernel = KernelScDM(
                in_dim=in_dim,
                eps_init=eps_init,
                alpha_init=alpha_init,
                dtype=dtype,
                metric=metric,
                periodic_axes=periodic_axes,
                backend=cast(Literal["torch", "keops"], backend),
            )
        elif in_dim is not None:
            raise TypeError("Pass either kernel or in_dim, not both.")
        self.kernel = kernel
        self.density_bandwidth_factor = float(density_bandwidth_factor)
        self.register_buffer("density_reference_row_sums", torch.empty(0, dtype=kernel.dtype))

    @property
    def in_dim(self) -> int:
        return self.kernel.in_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.kernel.dtype

    @property
    def eps(self) -> torch.Tensor:
        return self.kernel.eps

    @property
    def alpha(self) -> torch.Tensor:
        return self.kernel.alpha

    @property
    def density_eps(self) -> torch.Tensor:
        return self.kernel.eps * self.density_bandwidth_factor

    @property
    def _Xref(self) -> torch.Tensor:
        """Compatibility for internal diagnostics; analysis code uses the kernel property."""

        return self.kernel.reference_points

    def _raw_kernel(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor | None = None,
        *,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel._raw_kernel(rows, cols, eps=eps)

    def _floor_positive(self, values: torch.Tensor) -> torch.Tensor:
        return values.clamp_min(torch.finfo(values.dtype).tiny)

    def prepare_reference(self, X_ref: torch.Tensor) -> DiffusionHeatSections:
        """Prepare fixed-kernel heat analysis state from reference locations."""

        self.kernel.require_fixed_parameters()
        reference = torch.as_tensor(X_ref, dtype=self.dtype)
        if reference.ndim != 2 or reference.shape[1] != self.in_dim:
            raise ValueError(f"X_ref must have shape (N, {self.in_dim}).")
        self.kernel.set_reference_data(reference)
        self.density_reference_row_sums = self.kernel.raw_row_sums(reference, eps=self.density_eps)
        return self

    def set_reference_data(self, X_ref: torch.Tensor) -> None:
        """Prepare reference state; retained as a neutral preparation spelling."""

        self.prepare_reference(X_ref)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        saved = state_dict.get(prefix + "density_reference_row_sums")
        if saved is not None and self.density_reference_row_sums.shape != saved.shape:
            self.density_reference_row_sums = torch.empty(
                saved.shape,
                dtype=self.density_reference_row_sums.dtype,
                device=self.density_reference_row_sums.device,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _require_reference_data(self) -> None:
        if self.density_reference_row_sums.numel() == 0:
            raise RuntimeError("Call prepare_reference before evaluating heat sections.")

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
        """Estimate a uniform-reference manifold volume from short-step row sums."""

        self._require_reference_data()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if method not in {"median", "mean"}:
            raise ValueError("method must be either 'median' or 'mean'.")
        q_ref = self.kernel.reference_row_sums
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
                "Reference row sums vary substantially; the volume estimator assumes "
                "approximately uniform reference sampling and weak boundary bias.",
                RuntimeWarning,
                stacklevel=2,
            )
        if return_diagnostics:
            return volume, diagnostics
        return volume

    def _as_points(self, values: torch.Tensor, *, name: str) -> torch.Tensor:
        points = torch.as_tensor(values, dtype=self.dtype, device=self._Xref.device)
        if points.ndim < 2 or points.shape[-1] != self.in_dim:
            raise ValueError(f"{name} must have shape (..., N, {self.in_dim}).")
        return points

    def _steps(self, steps: int | Sequence[int]) -> tuple[tuple[int, ...], bool]:
        if isinstance(steps, int):
            values = (steps,)
            single = True
        elif isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            values = tuple(int(step) for step in steps)
            single = False
        else:
            raise TypeError("steps must be an int or a sequence of ints.")
        if not values or any(step < 1 for step in values):
            raise ValueError("steps values must be positive.")
        return values, single

    def _density_row_sums(self, rows: torch.Tensor) -> torch.Tensor:
        return self.kernel.raw_row_sums(rows, eps=self.density_eps)

    def _density_query_weights(self, rows: torch.Tensor) -> torch.Tensor:
        return self._density_row_sums(rows).reciprocal() / self._floor_positive(
            self.density_reference_row_sums.reciprocal().sum()
        )

    def _density_apply(
        self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        q_cols = self._density_row_sums(cols)
        ref_weights = self.density_reference_row_sums ** (-alpha)
        normalizer = self._floor_positive(
            self.kernel.raw_apply(rows, self._Xref, ref_weights[:, None], eps=self.eps).squeeze(-1)
        )
        applied = self.kernel.raw_apply(
            rows, cols, values / (q_cols**alpha)[..., :, None], eps=self.eps
        )
        return applied / normalizer[..., :, None]

    def _density_transpose_apply(
        self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        q_cols = self._density_row_sums(cols)
        ref_weights = self.density_reference_row_sums ** (-alpha)
        normalizer = self._floor_positive(
            self.kernel.raw_apply(rows, self._Xref, ref_weights[:, None], eps=self.eps).squeeze(-1)
        )
        applied = self.kernel.raw_apply(cols, rows, values / normalizer[..., :, None], eps=self.eps)
        return applied / (q_cols**alpha)[..., :, None]

    def _identity(self, points: torch.Tensor) -> torch.Tensor:
        size = points.shape[-2]
        eye = torch.eye(size, dtype=self.dtype, device=points.device)
        return eye if points.ndim == 2 else eye.expand(*points.shape[:-2], size, size)

    def _location_weights(
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
            return (
                torch.ones(locations.shape[:-1], dtype=self.dtype, device=locations.device)
                * (volume_tensor / locations.shape[-2]),
                {"volume_normalization": "explicit_volume"},
            )
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
            return (
                torch.ones(locations.shape[:-1], dtype=self.dtype, device=locations.device)
                * (volume_tensor.to(locations.device) / locations.shape[-2]),
                {"volume_normalization": "estimate_volume", **diagnostics},
            )
        if volume_normalization != "none":
            raise ValueError("Unsupported volume_normalization.")
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
        value = torch.as_tensor(weights, dtype=self.dtype, device=locations.device)
        if value.ndim < 1 or value.shape[-1] != locations.shape[-2]:
            raise ValueError("location_weights must have shape (..., Nloc).")
        return value, {"volume_normalization": "none", "location_weights": "explicit"}

    def _normalize(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        normalization: Literal["source", "median", "none"],
    ) -> torch.Tensor:
        if normalization == "none":
            return values
        if normalization not in {"source", "median"}:
            raise ValueError("mass_normalization must be one of 'source', 'median', or 'none'.")
        mass = (values * weights[..., None]).sum(dim=-2)
        if normalization == "source":
            return values / self._floor_positive(mass)[..., None, :]
        return values / self._floor_positive(torch.median(mass.reshape(-1)))

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
        """Evaluate heat sections indexed as ``(..., Nloc, Nsrc)``."""

        self._require_reference_data()
        step_values, return_single = self._steps(steps)
        locations_tensor = self._as_points(locations, name="locations")
        sources_tensor = (
            locations_tensor if sources is None else self._as_points(sources, name="sources")
        )
        if volume_normalization == "estimate_volume" and mode != "uniform":
            raise ValueError(
                "volume_normalization='estimate_volume' is only supported in uniform mode."
            )
        weights, diagnostics = self._location_weights(
            locations_tensor,
            location_weights,
            volume_normalization=volume_normalization,
            volume=volume,
            volume_dim=volume_dim,
            volume_estimate_warnings=volume_estimate_warnings,
            volume_row_sum_cv_warn=volume_row_sum_cv_warn,
            volume_row_sum_spread_warn=volume_row_sum_spread_warn,
        )
        if mode == "density":
            alpha_value = (
                self.alpha
                if alpha is None
                else torch.as_tensor(alpha, dtype=self.dtype, device=self._Xref.device)
            )
            result = self._density_heat(
                sources_tensor,
                locations_tensor,
                step_values,
                alpha_value,
                weights,
                mass_normalization,
            )
        elif mode == "uniform":
            if alpha is not None:
                raise ValueError("alpha is only supported for density heat kernels.")
            result = self._uniform_heat(
                sources_tensor, locations_tensor, step_values, weights, mass_normalization
            )
        else:
            raise ValueError("mode must be either 'density' or 'uniform'.")
        ordered = [result[step] for step in step_values]
        output = ordered[0] if return_single else torch.stack(ordered, dim=0)
        return (output, diagnostics) if return_diagnostics else output

    def _density_heat(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        steps: tuple[int, ...],
        alpha: torch.Tensor,
        weights: torch.Tensor,
        normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        result: dict[int, torch.Tensor] = {}
        identities = self._identity(sources)
        target_weights = self._density_query_weights(locations)
        if 1 in steps:
            values = self._density_transpose_apply(sources, locations, identities, alpha)
            result[1] = self._normalize(
                values / target_weights[..., :, None], weights, normalization
            )
        larger = sorted({step for step in steps if step >= 2})
        if not larger:
            return result
        state = self._density_transpose_apply(sources, self._Xref, identities, alpha)
        current_power = 0
        for step in larger:
            while current_power < step - 2:
                state = self._density_transpose_apply(self._Xref, self._Xref, state, alpha)
                current_power += 1
            values = self._density_transpose_apply(self._Xref, locations, state, alpha)
            result[step] = self._normalize(
                values / target_weights[..., :, None], weights, normalization
            )
        return result

    def _uniform_heat(
        self,
        sources: torch.Tensor,
        locations: torch.Tensor,
        steps: tuple[int, ...],
        weights: torch.Tensor,
        normalization: Literal["source", "median", "none"],
    ) -> dict[int, torch.Tensor]:
        result: dict[int, torch.Tensor] = {}
        identities = self._identity(sources)
        scale = torch.as_tensor(self._Xref.shape[-2], dtype=self.dtype, device=self._Xref.device)
        if 1 in steps:
            values = scale * self.kernel.uniform_symmetric_apply(locations, sources, identities)
            result[1] = self._normalize(values, weights, normalization)
        larger = sorted({step for step in steps if step >= 2})
        if not larger:
            return result
        state = self.kernel.uniform_symmetric_apply(self._Xref, sources, identities)
        current_power = 0
        for step in larger:
            while current_power < step - 2:
                state = self.kernel.uniform_symmetric_apply(self._Xref, self._Xref, state)
                current_power += 1
            values = scale * self.kernel.uniform_symmetric_apply(locations, self._Xref, state)
            result[step] = self._normalize(values, weights, normalization)
        return result
