from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy import sparse, spatial

from dymad.modules.helpers import _swap_parameter_storage
from dymad.modules.kernel import KernelScDM


class KernelSparseScDM(KernelScDM):
    """Sparse Euclidean backend for uniform diffusion-map heat sections."""

    def __init__(
        self,
        in_dim: int,
        eps_init: float | None = None,
        t_init: float = 1.0,
        dtype: torch.dtype | None = None,
        *,
        kernel_tol: float = 1e-10,
        chunk_rows: int = 512,
        metric: str = "euclidean",
    ):
        if metric != "euclidean":
            raise NotImplementedError("KernelSparseScDM currently supports only Euclidean inputs.")
        if kernel_tol <= 0.0:
            raise ValueError("kernel_tol must be positive.")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive.")
        super().__init__(in_dim, eps_init=eps_init, t_init=t_init, dtype=dtype, metric=metric)
        self.kernel_tol = float(kernel_tol)
        self.chunk_rows = int(chunk_rows)
        self._x_ref_np = np.empty((0, self.in_dim), dtype=float)
        self._k_ref_ref: Any = None
        self._q_ref_np = np.empty(0, dtype=float)
        self._d_ref_np = np.empty(0, dtype=float)
        self._s_ref_np = np.empty(0, dtype=float)
        self._s_ref_ref: Any = None

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        tensor = torch.as_tensor(Xref, dtype=self.dtype)
        if tensor.ndim != 2 or tensor.shape[1] != self.in_dim:
            raise ValueError(f"Xref must have shape (N, {self.in_dim}).")
        if self._log_eps.numel() == 0:
            raise RuntimeError("KernelSparseScDM requires eps_init for sparse neighbor search.")
        _swap_parameter_storage(self._Xref, tensor, requires_grad=False)
        self._x_ref_np = tensor.detach().cpu().numpy()
        k_ref_ref = self._sparse_raw_block(self._x_ref_np, self._x_ref_np)
        q_ref = self._positive(np.asarray(k_ref_ref.sum(axis=1)).ravel())
        d_ref = q_ref ** (-float(self.t.detach().cpu()))
        base = sparse.diags(d_ref).dot(k_ref_ref).dot(sparse.diags(d_ref)).tocsr()
        s_ref = self._positive(np.asarray(base.sum(axis=1)).ravel()) ** (-0.5)
        self._k_ref_ref = k_ref_ref
        self._q_ref_np = q_ref
        self._d_ref_np = d_ref
        self._s_ref_np = s_ref
        self._s_ref_ref = sparse.diags(s_ref).dot(base).dot(sparse.diags(s_ref)).tocsr()

    def forward(self, X: torch.Tensor, Z: torch.Tensor | None = None) -> torch.Tensor:
        if Z is None:
            Z = X
        block = self._uniform_block_np(self._to_numpy(X), self._to_numpy(Z)).toarray()
        return torch.as_tensor(block, dtype=self.dtype, device=torch.as_tensor(X).device)

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
        self._require_sparse_reference()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        center = np.median(self._q_ref_np) if method == "median" else np.mean(self._q_ref_np)
        volume_value = (
            len(self._q_ref_np)
            * (4.0 * np.pi * float(self.eps.detach().cpu())) ** (0.5 * dim)
            / max(float(center), np.finfo(float).tiny)
        )
        q_mean = float(np.mean(self._q_ref_np))
        q_cv = float(np.std(self._q_ref_np) / max(q_mean, np.finfo(float).tiny))
        q_p05, q_p95 = np.quantile(self._q_ref_np, [0.05, 0.95])
        spread = float(q_p95 / max(q_p05, np.finfo(float).tiny))
        if warn and (q_cv > row_sum_cv_warn or spread > row_sum_spread_warn):
            warnings.warn("Reference row sums vary substantially.", RuntimeWarning, stacklevel=2)
        diagnostics: dict[str, float | int | str] = {
            "dim": dim,
            "method": method,
            "volume": volume_value,
            "row_sum_mean": q_mean,
            "row_sum_median": float(np.median(self._q_ref_np)),
            "row_sum_cv": q_cv,
            "row_sum_p05": float(q_p05),
            "row_sum_p95": float(q_p95),
            "row_sum_p95_p05": spread,
        }
        volume = torch.as_tensor(volume_value, dtype=self.dtype, device=self._Xref.device)
        return (volume, diagnostics) if return_diagnostics else volume

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
        if mode.lower() != "uniform":
            raise NotImplementedError(
                "KernelSparseScDM currently supports only uniform heat kernels."
            )
        if alpha is not None:
            raise ValueError("alpha is only supported for density heat kernels.")
        locations_tensor = self._as_heat_points(locations, name="locations")
        sources_tensor = (
            locations_tensor if sources is None else self._as_heat_points(sources, name="sources")
        )
        step_values, return_single = self._heat_step_values(steps)
        weights, diagnostics = self._sparse_location_weights(
            locations_tensor,
            location_weights,
            volume_normalization,
            volume,
            volume_dim,
            volume_estimate_warnings,
            volume_row_sum_cv_warn,
            volume_row_sum_spread_warn,
        )
        loc_np = locations_tensor.detach().cpu().numpy()
        src_np = sources_tensor.detach().cpu().numpy()
        result_by_step = {
            step: self._normalize_np(
                self._section_np(src_np, loc_np, step), weights, mass_normalization
            )
            for step in step_values
        }
        ordered = [
            torch.as_tensor(
                result_by_step[step].T, dtype=self.dtype, device=locations_tensor.device
            )
            for step in step_values
        ]
        result = ordered[0] if return_single else torch.stack(ordered, dim=0)
        return (result, diagnostics) if return_diagnostics else result

    def _section_np(self, sources: np.ndarray, locations: np.ndarray, steps: int) -> np.ndarray:
        self._require_sparse_reference()
        if steps == 1:
            return self._Xref.shape[0] * self._uniform_block_np(sources, locations).toarray()
        source_ref = self._uniform_block_np(sources, self._x_ref_np)
        ref_loc = self._uniform_block_np(self._x_ref_np, locations)
        state: Any = source_ref
        for _ in range(steps - 2):
            state = state @ self._s_ref_ref
        return self._Xref.shape[0] * cast(Any, state @ ref_loc).toarray()

    def _uniform_block_np(self, rows: np.ndarray, cols: np.ndarray) -> Any:
        k_rows_ref = self._sparse_raw_block(rows, self._x_ref_np)
        k_cols_ref = k_rows_ref if rows is cols else self._sparse_raw_block(cols, self._x_ref_np)
        q_rows = self._positive(np.asarray(k_rows_ref.sum(axis=1)).ravel())
        q_cols = self._positive(np.asarray(k_cols_ref.sum(axis=1)).ravel())
        d_rows = q_rows ** (-float(self.t.detach().cpu()))
        d_cols = q_cols ** (-float(self.t.detach().cpu()))
        row_base = sparse.diags(d_rows).dot(k_rows_ref).dot(sparse.diags(self._d_ref_np)).tocsr()
        col_base = sparse.diags(d_cols).dot(k_cols_ref).dot(sparse.diags(self._d_ref_np)).tocsr()
        s_rows = self._positive(np.asarray(row_base.sum(axis=1)).ravel()) ** (-0.5)
        s_cols = self._positive(np.asarray(col_base.sum(axis=1)).ravel()) ** (-0.5)
        block = self._sparse_raw_block(rows, cols)
        return sparse.diags(s_rows * d_rows).dot(block).dot(sparse.diags(d_cols * s_cols)).tocsr()

    def _sparse_raw_block(self, rows: np.ndarray, cols: np.ndarray) -> Any:
        eps = float(self.eps.detach().cpu())
        radius = math.sqrt(max(0.0, -4.0 * eps * math.log(self.kernel_tol)))
        tree: Any = spatial.KDTree(cols)
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        for start in range(0, rows.shape[0], self.chunk_rows):
            neighborhoods = tree.query_ball_point(rows[start : start + self.chunk_rows], radius)
            for offset, neigh in enumerate(neighborhoods):
                row_idx = start + offset
                if not neigh:
                    _, nearest = tree.query(rows[row_idx], k=1)
                    neigh = [int(nearest)]
                col_idx = np.asarray(neigh, dtype=np.int64)
                sq = np.sum((cols[col_idx] - rows[row_idx]) ** 2, axis=1)
                values = np.exp(-sq / (4.0 * eps))
                keep = values >= self.kernel_tol
                if not np.any(keep):
                    keep[np.argmax(values)] = True
                row_parts.append(np.full(int(np.sum(keep)), row_idx, dtype=np.int64))
                col_parts.append(col_idx[keep])
                data_parts.append(values[keep])
        row_all = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
        col_all = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int64)
        data_all = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=float)
        return sparse.csr_matrix(
            (data_all, (row_all, col_all)), shape=(rows.shape[0], cols.shape[0])
        )

    def _sparse_location_weights(
        self,
        locations: torch.Tensor,
        weights: torch.Tensor | None,
        volume_normalization: str,
        volume: torch.Tensor | float | None,
        volume_dim: int | None,
        volume_estimate_warnings: bool,
        volume_row_sum_cv_warn: float,
        volume_row_sum_spread_warn: float,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        if weights is not None and volume_normalization != "none":
            raise ValueError("location_weights cannot be combined with volume_normalization.")
        if volume_normalization == "none":
            if weights is None:
                return np.full(locations.shape[-2], 1.0 / locations.shape[-2]), {
                    "volume_normalization": "none"
                }
            return torch.as_tensor(weights, dtype=self.dtype).detach().cpu().numpy(), {
                "volume_normalization": "none",
                "location_weights": "explicit",
            }
        if volume_normalization == "explicit_volume":
            if volume is None:
                raise ValueError("volume is required when volume_normalization='explicit_volume'.")
            value = float(torch.as_tensor(volume, dtype=self.dtype).detach().cpu())
            return np.full(locations.shape[-2], value / locations.shape[-2]), {
                "volume_normalization": "explicit_volume",
                "volume": value,
            }
        if volume_normalization == "estimate_volume":
            if volume_dim is None:
                raise ValueError(
                    "volume_dim is required when volume_normalization='estimate_volume'."
                )
            vol, diagnostics = cast(
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
            return np.full(
                locations.shape[-2], float(vol.detach().cpu()) / locations.shape[-2]
            ), diagnostics
        raise ValueError(
            "volume_normalization must be one of 'none', 'explicit_volume', or 'estimate_volume'."
        )

    def _normalize_np(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        normalization: Literal["source", "median", "none"],
    ) -> np.ndarray:
        if normalization == "none":
            return values
        mass = values @ weights
        if normalization == "source":
            return values / self._positive(mass)[:, None]
        if normalization == "median":
            return values / max(float(np.median(mass)), np.finfo(float).tiny)
        raise ValueError("mass_normalization must be one of 'source', 'median', or 'none'.")

    def _require_sparse_reference(self) -> None:
        self._require_reference_data()
        if self._s_ref_ref is None:
            raise RuntimeError("Call set_reference_data before evaluating sparse sections.")

    def _to_numpy(self, value: torch.Tensor) -> np.ndarray:
        return torch.as_tensor(value, dtype=self.dtype).detach().cpu().numpy()

    @staticmethod
    def _positive(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, np.finfo(float).tiny)
