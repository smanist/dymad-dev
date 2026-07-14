"""Kernel Gram eigensystems and Nyström interpolation."""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from dymad.modules.kernel import KernelScalarValued

KernelEigenSolver = Literal["dense", "matrix_free", "scipy"]


class KernelEigenbasis(nn.Module):
    """Leading scalar-kernel eigensystem with Nyström interpolation.

    ``solve`` uses the raw Gram matrix by default. Passing ``sample_weights``
    instead solves the symmetric quadrature matrix ``sqrt(W) K sqrt(W)``.
    The resulting basis is fixed; ``transform`` remains differentiable with
    respect to its query points.
    """

    def __init__(
        self,
        kernel: KernelScalarValued,
        n_components: int,
        *,
        skip: int = 0,
        eigenvalue_rtol: float = 1.0e-10,
    ) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        if skip < 0:
            raise ValueError("skip must be nonnegative.")
        if eigenvalue_rtol <= 0:
            raise ValueError("eigenvalue_rtol must be positive.")
        if kernel.is_operator_valued:
            raise TypeError("KernelEigenbasis supports scalar-valued kernels only.")
        self.kernel = kernel
        self.n_components = int(n_components)
        self.skip = int(skip)
        self.eigenvalue_rtol = float(eigenvalue_rtol)
        self.register_buffer(
            "reference_points", torch.empty((0, kernel.in_dim), dtype=kernel.dtype)
        )
        self.register_buffer("sample_weights", torch.empty(0, dtype=kernel.dtype))
        self.register_buffer("eigenvalues", torch.empty(0, dtype=kernel.dtype))
        self.register_buffer("eigenvectors", torch.empty((0, 0), dtype=kernel.dtype))
        self.register_buffer("reference_eigenfunctions", torch.empty((0, 0), dtype=kernel.dtype))
        self.diagnostics: dict[str, Any] = {}

    @property
    def symmetric_eigenvectors(self) -> torch.Tensor:
        """Euclidean eigenvectors of the solved symmetric matrix."""

        return self.eigenvectors

    @property
    def is_weighted(self) -> bool:
        return self.sample_weights.numel() > 0

    def solve(
        self,
        X_ref: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None = None,
        solver: KernelEigenSolver = "dense",
        prepare_reference: bool = True,
        tol: float | None = None,
        max_iterations: int | None = None,
        block_size: int | None = None,
        seed: int | None = None,
    ) -> KernelEigenbasis:
        """Solve the requested reference Gram eigensystem."""

        if solver not in {"dense", "matrix_free", "scipy"}:
            raise ValueError("solver must be 'dense', 'matrix_free', or 'scipy'.")
        reference = torch.as_tensor(X_ref, dtype=self.kernel.dtype)
        self._validate_reference(reference)
        if prepare_reference:
            self.kernel.require_fixed_parameters()
            self.kernel.set_reference_data(reference)

        count = self.n_components + self.skip
        if count > reference.shape[0]:
            raise ValueError("n_components + skip cannot exceed the reference count.")
        weights = self._validate_weights(sample_weights, reference)
        if solver in {"matrix_free", "scipy"} and count >= reference.shape[0]:
            raise ValueError("matrix-free and SciPy solvers require n_components + skip < N.")

        with torch.no_grad():
            if solver == "dense":
                values, vectors, diagnostics = self._solve_dense(reference, weights, count)
            elif solver == "scipy":
                values, vectors, diagnostics = self._solve_scipy(reference, weights, count, tol)
            else:
                values, vectors, diagnostics = self._solve_matrix_free(
                    reference,
                    weights,
                    count,
                    tol=tol,
                    max_iterations=max_iterations,
                    block_size=block_size,
                    seed=seed,
                )

        values = values[self.skip :]
        vectors = vectors[:, self.skip :]
        self._validate_extendable(values)
        self.reference_points = reference.detach().clone()
        self.sample_weights = (
            torch.empty(0, dtype=reference.dtype, device=reference.device)
            if weights is None
            else weights.detach().clone()
        )
        self.eigenvalues = values.detach().clone()
        self.eigenvectors = vectors.detach().clone()
        if weights is None:
            self.reference_eigenfunctions = vectors.detach().clone()
        else:
            self.reference_eigenfunctions = (vectors / weights.sqrt()[:, None]).detach().clone()
        self.diagnostics = diagnostics
        return self

    def transform(self, X_new: torch.Tensor) -> torch.Tensor:
        """Nyström-interpolate the fixed reference eigenfunctions to ``X_new``."""

        if self.eigenvalues.numel() == 0:
            raise RuntimeError("Call solve before transform.")
        queries = torch.as_tensor(
            X_new, dtype=self.kernel.dtype, device=self.reference_points.device
        )
        if queries.ndim != 2 or queries.shape[1] != self.kernel.in_dim:
            raise ValueError(f"X_new must have shape (N, {self.kernel.in_dim}).")
        coefficients = self.eigenvectors
        if self.is_weighted:
            coefficients = self.sample_weights.sqrt()[:, None] * coefficients
        values = self.kernel.apply(queries, self.reference_points, coefficients)
        return values / self.eigenvalues[None, :]

    def _validate_reference(self, reference: torch.Tensor) -> None:
        if reference.ndim != 2 or reference.shape[1] != self.kernel.in_dim:
            raise ValueError(f"X_ref must have shape (N, {self.kernel.in_dim}).")
        if reference.shape[0] < 2:
            raise ValueError("X_ref must contain at least two points.")
        if not torch.is_floating_point(reference) or not torch.isfinite(reference).all():
            raise ValueError("X_ref must be a finite floating-point tensor.")

    def _validate_weights(
        self, sample_weights: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        if sample_weights is None:
            return None
        weights = torch.as_tensor(sample_weights, dtype=reference.dtype, device=reference.device)
        if weights.ndim != 1 or weights.shape[0] != reference.shape[0]:
            raise ValueError("sample_weights must have shape (N,).")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("sample_weights must be finite and strictly positive.")
        return weights

    def _operator(self, reference: torch.Tensor, weights: torch.Tensor | None) -> Any:
        if weights is None:
            return lambda values: self.kernel.apply(reference, reference, values)
        sqrt_weights = weights.sqrt()

        def apply(values: torch.Tensor) -> torch.Tensor:
            return sqrt_weights[:, None] * self.kernel.apply(
                reference, reference, sqrt_weights[:, None] * values
            )

        return apply

    def _solve_dense(
        self, reference: torch.Tensor, weights: torch.Tensor | None, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        matrix = self.kernel.materialize(reference, reference)
        if weights is not None:
            sqrt_weights = weights.sqrt()
            matrix = sqrt_weights[:, None] * matrix * sqrt_weights[None, :]
        scale = float(matrix.abs().max().clamp_min(1.0).detach().cpu())
        if not torch.allclose(
            matrix,
            matrix.transpose(-1, -2),
            rtol=100.0 * torch.finfo(matrix.dtype).eps,
            atol=100.0 * torch.finfo(matrix.dtype).eps * scale,
        ):
            raise ValueError("Kernel Gram matrix must be symmetric.")
        values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
        order = torch.arange(
            values.numel() - 1, values.numel() - count - 1, -1, device=values.device
        )
        return values[order], vectors[:, order], {"solver": "dense", "converged": True}

    def _solve_scipy(
        self,
        reference: torch.Tensor,
        weights: torch.Tensor | None,
        count: int,
        tol: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if reference.device.type != "cpu":
            raise ValueError("solver='scipy' requires CPU reference points and kernel state.")
        apply = self._operator(reference, weights)
        n = reference.shape[0]

        def matvec(values: np.ndarray) -> np.ndarray:
            tensor = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)[
                :, None
            ]
            return apply(tensor).squeeze(-1).detach().cpu().numpy()

        linear_operator_factory = cast(Any, LinearOperator)
        operator = linear_operator_factory(
            (n, n), matvec=matvec, dtype=reference.detach().cpu().numpy().dtype
        )
        kwargs: dict[str, Any] = {"k": count, "which": "LA"}
        if tol is not None:
            kwargs["tol"] = tol
        try:
            values_np, vectors_np = eigsh(operator, **kwargs)
        except ArpackNoConvergence as exc:
            warnings.warn("SciPy eigsh did not converge.", RuntimeWarning, stacklevel=2)
            raise exc
        order = np.argsort(values_np)[::-1]
        values = torch.as_tensor(values_np[order], dtype=reference.dtype, device=reference.device)
        vectors = torch.as_tensor(
            vectors_np[:, order], dtype=reference.dtype, device=reference.device
        )
        return values, vectors, {"solver": "scipy", "converged": True}

    def _solve_matrix_free(
        self,
        reference: torch.Tensor,
        weights: torch.Tensor | None,
        count: int,
        *,
        tol: float | None,
        max_iterations: int | None,
        block_size: int | None,
        seed: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        n = reference.shape[0]
        actual_block = min(n, max(2 * count, count + 8) if block_size is None else block_size)
        if actual_block < count:
            raise ValueError("block_size must be at least n_components + skip.")
        iterations = max(100, 10 * count) if max_iterations is None else max_iterations
        if iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        epsilon = torch.finfo(reference.dtype).eps
        relative_tol = epsilon**0.5 if tol is None else tol
        if relative_tol <= 0:
            raise ValueError("tol must be positive.")
        generator = torch.Generator(device=reference.device)
        if seed is not None:
            generator.manual_seed(seed)
        apply = self._operator(reference, weights)
        initial = torch.randn(
            (n, actual_block), dtype=reference.dtype, device=reference.device, generator=generator
        )
        current, _ = torch.linalg.qr(initial, mode="reduced")
        basis_blocks: list[torch.Tensor] = []
        applied_blocks: list[torch.Tensor] = []
        values = torch.empty(0, dtype=reference.dtype, device=reference.device)
        vectors = torch.empty((n, 0), dtype=reference.dtype, device=reference.device)
        residuals = torch.full((count,), torch.inf, dtype=reference.dtype, device=reference.device)
        for iteration in range(1, iterations + 1):
            applied = apply(current)
            basis_blocks.append(current)
            applied_blocks.append(applied)
            basis = torch.cat(basis_blocks, dim=1)
            applied_basis = torch.cat(applied_blocks, dim=1)
            projected = basis.T @ applied_basis
            projected = (projected + projected.T) * 0.5
            small_values, small_vectors = torch.linalg.eigh(projected)
            order = torch.arange(
                small_values.numel() - 1,
                small_values.numel() - count - 1,
                -1,
                device=reference.device,
            )
            values = small_values[order]
            vectors = basis @ small_vectors[:, order]
            residuals = torch.linalg.vector_norm(
                applied_basis @ small_vectors[:, order] - vectors * values[None, :], dim=0
            )
            scale = values.abs().clamp_min(1.0)
            if torch.all(residuals <= relative_tol * scale):
                return (
                    values,
                    vectors,
                    {
                        "solver": "matrix_free",
                        "converged": True,
                        "iterations": iteration,
                        "residuals": residuals.detach().cpu().tolist(),
                    },
                )
            next_block = applied
            for block in basis_blocks:
                next_block = next_block - block @ (block.T @ next_block)
            next_block, triangular = torch.linalg.qr(next_block, mode="reduced")
            rank = int((torch.abs(torch.diagonal(triangular)) > 100.0 * epsilon).sum())
            if rank == 0:
                break
            current = next_block[:, :rank]
        warnings.warn(
            "Matrix-free kernel eigensolver did not converge within the iteration budget.",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            values,
            vectors,
            {
                "solver": "matrix_free",
                "converged": False,
                "iterations": len(basis_blocks),
                "residuals": residuals.detach().cpu().tolist(),
            },
        )

    def _validate_extendable(self, values: torch.Tensor) -> None:
        threshold = self.eigenvalue_rtol * values.abs().max().clamp_min(1.0)
        if torch.any(values.abs() <= threshold):
            raise ValueError("Selected eigenvalues are too close to zero for Nyström extension.")
