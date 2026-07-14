"""Dense RBF-kernel eigensystem convergence on a 3D donut torus.

The embedded torus has non-constant area density ``R + cos(theta)``. A small
dense study uses up to a 64-by-64 (4,096 point) quadrature grid.  This is
deliberately not a sparse or million-point KeOps example.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.kernel_analysis import KernelEigenbasis
from dymad.modules import KernelScRBF

ifrun = True
ifplt = True
MAJOR_RADIUS = 2.0
FAST_MODE = 0
GRID_SIDES = (4, 6, 8) if FAST_MODE else (8, 16, 32, 64)
REFERENCE_SIDE = GRID_SIDES[-1]
# Among sigma=(0.20, 0.30, 0.40), 0.30 is the smallest for which the first
# five weighted eigenvalues change by under 0.7% from 32-by-32 to 64-by-64.
SIGMA = 0.30
N_COMPONENTS = 5
OUTPUT_ROOT = Path(os.environ.get("DYMAD_KERNEL_ANALYSIS_OUTPUT", Path(__file__).with_name("runs")))


def donut_grid(side: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    axis = 2.0 * np.pi * np.arange(side) / side
    theta, phi = np.meshgrid(axis, axis, indexing="ij")
    theta, phi = theta.ravel(), phi.ravel()
    radius = MAJOR_RADIUS + np.cos(theta)
    points = torch.tensor(
        np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.sin(theta))),
        dtype=torch.float64,
    )
    weights = torch.tensor(radius / (MAJOR_RADIUS * side * side), dtype=torch.float64)
    return theta, phi, points, weights


def solve(points: torch.Tensor, weights: torch.Tensor) -> KernelEigenbasis:
    return KernelEigenbasis(
        KernelScRBF(in_dim=3, lengthscale_init=SIGMA, dtype=torch.float64), N_COMPONENTS
    ).solve(points, sample_weights=weights, solver="dense")


def normalize_columns(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    norms = torch.sqrt((weights[:, None] * values.square()).sum(dim=0))
    return values / norms[None, :]


def align_modes(
    numerical: torch.Tensor, reference: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    aligned = numerical.clone()
    sqrt_weights = weights.sqrt()[:, None]
    for indices in ((0,), (1, 2), (3, 4)):
        selected = list(indices)
        candidate = numerical[:, selected]
        truth = reference[:, selected]
        if len(selected) == 1:
            aligned[:, selected] = (
                candidate if (weights[:, None] * candidate * truth).sum() >= 0 else -candidate
            )
        else:
            left, _, right_t = torch.linalg.svd(
                (sqrt_weights * candidate).T @ (sqrt_weights * truth)
            )
            aligned[:, selected] = candidate @ (left @ right_t)
    return aligned


def weighted_projector_error(
    numerical: torch.Tensor, reference: torch.Tensor, weights: torch.Tensor
) -> float:
    q_numerical, _ = torch.linalg.qr(weights.sqrt()[:, None] * numerical)
    q_reference, _ = torch.linalg.qr(weights.sqrt()[:, None] * reference)
    return float(
        torch.linalg.matrix_norm(q_numerical @ q_numerical.T - q_reference @ q_reference.T)
        .detach()
        .cpu()
    )


def laplace_beltrami_spectral_modes(
    side: int, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Fourier--Galerkin modes of -Delta for the torus metric.

    Separation in phi gives a periodic one-dimensional weighted eigenproblem
    for each azimuthal Fourier order. The theta dependence is represented in
    a real Fourier basis and its mass and stiffness forms are integrated by
    the same periodic grid quadrature.
    """
    theta = 2.0 * torch.pi * torch.arange(side, dtype=torch.float64) / side
    radius = MAJOR_RADIUS + torch.cos(theta)
    step = 2.0 * torch.pi / side
    fourier_order = min(16, (side - 1) // 2)
    basis_columns = [torch.ones_like(theta)]
    derivative_columns = [torch.zeros_like(theta)]
    for order in range(1, fourier_order + 1):
        basis_columns.extend((torch.cos(order * theta), torch.sin(order * theta)))
        derivative_columns.extend(
            (-order * torch.sin(order * theta), order * torch.cos(order * theta))
        )
    basis = torch.column_stack(basis_columns)
    basis_derivative = torch.column_stack(derivative_columns)
    mass = step * (basis.T @ (radius[:, None] * basis))
    cholesky_mass = torch.linalg.cholesky(mass)
    candidates: list[tuple[float, torch.Tensor]] = []
    for azimuthal_order in range(N_COMPONENTS):
        stiffness = step * (
            basis_derivative.T @ (radius[:, None] * basis_derivative)
            + basis.T @ (((azimuthal_order**2 / radius)[:, None]) * basis)
        )
        left_solve = torch.linalg.solve_triangular(cholesky_mass, stiffness, upper=False)
        symmetric_stiffness = torch.linalg.solve_triangular(
            cholesky_mass, left_solve.T, upper=False
        ).T
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_stiffness)
        coefficients = torch.linalg.solve_triangular(cholesky_mass.T, eigenvectors, upper=True)
        theta_modes = basis @ coefficients
        for theta_index in range(N_COMPONENTS):
            theta_mode = theta_modes[:, theta_index]
            cosine = theta_mode[:, None] * torch.cos(azimuthal_order * theta)[None, :]
            candidates.append((float(eigenvalues[theta_index]), cosine.reshape(-1)))
            if azimuthal_order:
                sine = theta_mode[:, None] * torch.sin(azimuthal_order * theta)[None, :]
                candidates.append((float(eigenvalues[theta_index]), sine.reshape(-1)))

    candidates.sort(key=lambda candidate: candidate[0])
    selected = candidates[:N_COMPONENTS]
    modes = torch.column_stack([candidate[1] for candidate in selected])
    eigenvalues = torch.tensor([candidate[0] for candidate in selected], dtype=torch.float64)
    return normalize_columns(modes, weights), eigenvalues


if __name__ == "__main__" and ifrun:
    _, _, reference_points, reference_weights = donut_grid(REFERENCE_SIDE)
    reference = solve(reference_points, reference_weights)
    reference_modes = reference.reference_eigenfunctions
    laplace_modes, laplace_eigenvalues = laplace_beltrami_spectral_modes(
        REFERENCE_SIDE, reference_weights
    )
    numerical_modes = align_modes(reference_modes, laplace_modes, reference_weights)
    records: list[tuple[int, float, float]] = []
    for side in GRID_SIDES:
        _, _, points, weights = donut_grid(side)
        basis = reference if side == REFERENCE_SIDE else solve(points, weights)
        interpolated = normalize_columns(basis.transform(reference_points), reference_weights)
        aligned = align_modes(interpolated, reference_modes, reference_weights)
        eigenvalue_error = float(torch.max(torch.abs(basis.eigenvalues - reference.eigenvalues)))
        records.append(
            (
                points.shape[0],
                eigenvalue_error,
                weighted_projector_error(
                    aligned[:, 1:3], reference_modes[:, 1:3], reference_weights
                ),
            )
        )
        last_side, last_basis = side, basis

if __name__ == "__main__" and ifplt:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sizes, eigenvalue_errors, subspace_errors = map(np.asarray, zip(*records, strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].loglog(sizes, np.maximum(eigenvalue_errors, 1.0e-16), marker="o")
    axes[0].set(title="Dense eigenvalue convergence", xlabel="reference points", ylabel="max error")
    axes[1].loglog(sizes, np.maximum(subspace_errors, 1.0e-16), marker="o")
    axes[1].set(
        title="First-pair eigenspace convergence",
        xlabel="reference points",
        ylabel="projector norm",
    )
    figure.savefig(OUTPUT_ROOT / "donut_torus_rbf_eigenbasis_convergence.png", dpi=160)
    plt.close(figure)

    mode = np.arange(N_COMPONENTS)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].plot(
        mode, reference.eigenvalues.cpu(), color="black", marker="x", label="dense reference"
    )
    axes[0].plot(
        mode, last_basis.eigenvalues.cpu(), marker="o", linestyle="--", label=f"N={last_side**2}"
    )
    axes[0].set(title="First eigenvalues", xlabel="mode", ylabel="eigenvalue")
    axes[0].legend(fontsize="small")
    axes[1].semilogy(
        mode,
        np.maximum(
            np.abs(last_basis.eigenvalues.cpu().numpy() - reference.eigenvalues.cpu().numpy()),
            1.0e-16,
        ),
        marker="o",
    )
    axes[1].set(title="Reference error", xlabel="mode", ylabel="absolute error")
    figure.savefig(OUTPUT_ROOT / "donut_torus_rbf_eigenbasis_eigenvalues.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(
        3,
        N_COMPONENTS,
        figsize=(3 * N_COMPONENTS + 1.5, 7.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    difference_modes = numerical_modes - laplace_modes
    color_limit = max(
        float(torch.abs(values).max().detach().cpu())
        for values in (numerical_modes, laplace_modes, difference_modes)
    )
    tick_locations = (0.0, float(torch.pi), float(2.0 * torch.pi))
    tick_labels = ("0", r"$\pi$", r"$2\pi$")
    for mode_index in range(N_COMPONENTS):
        values = (
            numerical_modes[:, mode_index],
            laplace_modes[:, mode_index],
            difference_modes[:, mode_index],
        )
        labels = ("RBF Integral", "Laplace--Beltrami", "RBF minus LB")
        titles = (
            f"mode {mode_index}\nRBF λ={float(reference.eigenvalues[mode_index]):.3g}",
            f"LB λ={float(laplace_eigenvalues[mode_index]):.3g}",
            "difference",
        )
        for axis, label, title, value in zip(
            axes[:, mode_index], labels, titles, values, strict=True
        ):
            image = axis.imshow(
                value.detach().cpu().reshape(REFERENCE_SIDE, REFERENCE_SIDE),
                cmap="coolwarm",
                vmin=-color_limit,
                vmax=color_limit,
                extent=(0.0, float(2.0 * torch.pi), 0.0, float(2.0 * torch.pi)),
            )
            axis.set_ylabel(label if mode_index == 0 else "")
            axis.set_title(title)
    for axis in axes[-1]:
        axis.set(xlabel=r"$\phi$", xticks=tick_locations, xticklabels=tick_labels)
    for axis in axes[:, 0]:
        axis.set(ylabel=axis.get_ylabel(), yticks=tick_locations, yticklabels=tick_labels)
    figure.colorbar(image, ax=axes, shrink=0.82, label="field value")
    figure.savefig(OUTPUT_ROOT / "donut_torus_rbf_eigenbasis_eigenvectors.png", dpi=160)
    plt.close(figure)
