"""Convergence of RBF-kernel eigensystems on a flat product torus.

Run this file directly. It writes a plot under ``runs/`` beside the script,
or under ``DYMAD_KERNEL_ANALYSIS_OUTPUT`` when that environment variable is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import iv

from dymad.kernel_analysis import KernelEigenbasis
from dymad.modules import KernelScRBF

ifrun = True
ifplt = True
GRID_SIDES = (4, 6, 8)
SIGMA = 0.55
N_COMPONENTS = 5
SOLVERS = ("dense", "matrix_free", "scipy")
OUTPUT_ROOT = Path(os.environ.get("DYMAD_KERNEL_ANALYSIS_OUTPUT", Path(__file__).with_name("runs")))


def flat_torus_points(theta: np.ndarray, phi: np.ndarray) -> torch.Tensor:
    return torch.tensor(
        np.column_stack((np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi))),
        dtype=torch.float64,
    )


def projector_error(estimated: torch.Tensor, exact: torch.Tensor) -> float:
    q_est, _ = torch.linalg.qr(estimated)
    q_exact, _ = torch.linalg.qr(exact)
    return float(torch.linalg.matrix_norm(q_est @ q_est.T - q_exact @ q_exact.T))


def align_first_eigenspace(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    aligned = estimate.clone()
    aligned[:, 0] = estimate[:, 0] if (estimate[:, 0] * truth[:, 0]).sum() >= 0 else -estimate[:, 0]
    left, _, right_t = torch.linalg.svd(estimate[:, 1:].T @ truth[:, 1:])
    aligned[:, 1:] = estimate[:, 1:] @ (left @ right_t)
    return aligned


def solve_all(points: torch.Tensor) -> dict[str, KernelEigenbasis]:
    weights = torch.full((points.shape[0],), 1.0 / points.shape[0], dtype=torch.float64)
    return {
        solver: KernelEigenbasis(
            KernelScRBF(in_dim=4, lengthscale_init=SIGMA, dtype=torch.float64), N_COMPONENTS
        ).solve(points, sample_weights=weights, solver=solver, seed=0)
        for solver in SOLVERS
    }


if __name__ == "__main__" and ifrun:
    records: list[tuple[int, float, float]] = []
    inverse_scale = 1.0 / SIGMA**2
    prefactor = np.exp(-2.0 * inverse_scale)
    exact_constant = prefactor * iv(0, inverse_scale) ** 2
    exact_first = prefactor * iv(1, inverse_scale) * iv(0, inverse_scale)
    for side in GRID_SIDES:
        axis = 2.0 * np.pi * np.arange(side) / side
        theta, phi = np.meshgrid(axis, axis, indexing="ij")
        theta, phi = theta.ravel(), phi.ravel()
        points = flat_torus_points(theta, phi)
        count = points.shape[0]
        truth = torch.tensor(
            np.column_stack(
                (
                    np.ones_like(theta),
                    np.sqrt(2.0) * np.cos(theta),
                    np.sqrt(2.0) * np.sin(theta),
                    np.sqrt(2.0) * np.cos(phi),
                    np.sqrt(2.0) * np.sin(phi),
                )
            ),
            dtype=torch.float64,
        ) / np.sqrt(count)
        solutions = solve_all(points)
        basis = solutions["dense"]
        value_error = max(
            abs(float(basis.eigenvalues[0]) - exact_constant),
            abs(float(basis.eigenvalues[1:].mean()) - exact_first),
        )
        records.append(
            (count, value_error, projector_error(basis.eigenvectors[:, 1:], truth[:, 1:]))
        )
        last_side, last_truth, last_solutions = side, truth, solutions

if __name__ == "__main__" and ifplt:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sizes, value_errors, subspace_errors = map(np.asarray, zip(*records, strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].loglog(sizes, np.maximum(value_errors, 1.0e-16), marker="o")
    axes[0].set(title="Eigenvalue error", xlabel="reference points", ylabel="absolute error")
    axes[1].loglog(sizes, np.maximum(subspace_errors, 1.0e-16), marker="o")
    axes[1].set(title="First eigenspace error", xlabel="reference points", ylabel="projector norm")
    figure.savefig(OUTPUT_ROOT / "flat_torus_rbf_eigenbasis_convergence.png", dpi=160)
    plt.close(figure)

    exact_values = np.asarray((exact_constant, exact_first, exact_first, exact_first, exact_first))
    mode = np.arange(N_COMPONENTS)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].plot(mode, exact_values, color="black", marker="x", label="truth")
    for solver, basis in last_solutions.items():
        axes[0].plot(mode, basis.eigenvalues.cpu(), marker="o", linestyle="--", label=solver)
    axes[0].set(title="First eigenvalues", xlabel="mode", ylabel="eigenvalue")
    axes[0].legend(fontsize="small")
    for solver, basis in last_solutions.items():
        axes[1].semilogy(
            mode,
            np.maximum(np.abs(basis.eigenvalues.cpu().numpy() - exact_values), 1.0e-16),
            marker="o",
            label=solver,
        )
    axes[1].set(title="Eigenvalue error", xlabel="mode", ylabel="absolute error")
    axes[1].legend(fontsize="small")
    figure.savefig(OUTPUT_ROOT / "flat_torus_rbf_eigenbasis_eigenvalues.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(N_COMPONENTS, 4, figsize=(8, 9), constrained_layout=True)
    aligned = {
        solver: align_first_eigenspace(basis.eigenvectors, last_truth)
        for solver, basis in last_solutions.items()
    }
    labels = ("truth", *SOLVERS)
    for mode_index in range(N_COMPONENTS):
        values = (
            last_truth[:, mode_index],
            *(aligned[solver][:, mode_index] for solver in SOLVERS),
        )
        limit = max(float(torch.abs(value).max()) for value in values)
        for axis, label, value in zip(axes[mode_index], labels, values, strict=True):
            image = axis.imshow(
                value.reshape(last_side, last_side), cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axis.set(xticks=[], yticks=[], title=label if mode_index == 0 else "")
            if label == "truth":
                axis.set_ylabel(f"mode {mode_index}")
        figure.colorbar(image, ax=axes[mode_index].tolist(), shrink=0.65)
    figure.savefig(OUTPUT_ROOT / "flat_torus_rbf_eigenbasis_eigenvectors.png", dpi=160)
    plt.close(figure)
