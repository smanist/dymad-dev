"""Dense RBF-kernel eigensystem convergence on a vertically cut semi-torus.

The parameter domain is periodic in theta and has phi in [0, pi], giving two
minor-circle boundaries. The LB comparison uses Neumann conditions there.
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
# Sigma=0.20 is the smallest screened bandwidth and changes the first five
# eigenvalues by under 0.02% from 32-by-32 to 64-by-64.
SIGMA = 0.20
N_COMPONENTS = 5
OUTPUT_ROOT = Path(os.environ.get("DYMAD_KERNEL_ANALYSIS_OUTPUT", Path(__file__).with_name("runs")))


def semi_torus_grid(side: int) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 2.0 * torch.pi * torch.arange(side, dtype=torch.float64) / side
    phi = torch.linspace(0.0, torch.pi, side, dtype=torch.float64)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    radius = MAJOR_RADIUS + torch.cos(theta_grid)
    points = torch.column_stack(
        (
            (radius * torch.cos(phi_grid)).reshape(-1),
            (radius * torch.sin(phi_grid)).reshape(-1),
            torch.sin(theta_grid).reshape(-1),
        )
    )
    phi_weights = torch.ones(side, dtype=torch.float64)
    phi_weights[[0, -1]] = 0.5
    weights = (radius / MAJOR_RADIUS / side * phi_weights[None, :] / (side - 1)).reshape(-1)
    return points, weights


def solve(points: torch.Tensor, weights: torch.Tensor) -> KernelEigenbasis:
    return KernelEigenbasis(
        KernelScRBF(in_dim=3, lengthscale_init=SIGMA, dtype=torch.float64), N_COMPONENTS
    ).solve(points, sample_weights=weights, solver="dense")


def normalize_columns(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    norms = torch.sqrt((weights[:, None] * values.square()).sum(dim=0))
    return values / norms[None, :]


def align_individual_modes(
    numerical: torch.Tensor, reference: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    signs = torch.sign((weights[:, None] * numerical * reference).sum(dim=0))
    return numerical * torch.where(signs == 0.0, torch.ones_like(signs), signs)[None, :]


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


def laplace_beltrami_dirichlet_modes(
    side: int, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Fourier--Galerkin LB modes with Dirichlet phi-boundary conditions."""
    theta = 2.0 * torch.pi * torch.arange(side, dtype=torch.float64) / side
    phi = torch.linspace(0.0, torch.pi, side, dtype=torch.float64)
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
    for azimuthal_order in range(1, N_COMPONENTS + 1):
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
            values = theta_modes[:, theta_index, None] * torch.sin(azimuthal_order * phi)[None, :]
            candidates.append((float(eigenvalues[theta_index]), values.reshape(-1)))

    candidates.sort(key=lambda candidate: candidate[0])
    selected = candidates[:N_COMPONENTS]
    modes = torch.column_stack([candidate[1] for candidate in selected])
    eigenvalues = torch.tensor([candidate[0] for candidate in selected], dtype=torch.float64)
    return normalize_columns(modes, weights), eigenvalues


if __name__ == "__main__" and ifrun:
    reference_points, reference_weights = semi_torus_grid(REFERENCE_SIDE)
    reference = solve(reference_points, reference_weights)
    reference_modes = reference.reference_eigenfunctions
    dirichlet_modes, dirichlet_eigenvalues = laplace_beltrami_dirichlet_modes(
        REFERENCE_SIDE, reference_weights
    )
    # The fourth and fifth RBF modes correspond to the fifth and fourth
    # Dirichlet modes, respectively. The latter needs the opposite orientation.
    comparison_order = torch.tensor((0, 1, 2, 4, 3))
    dirichlet_modes = dirichlet_modes[:, comparison_order]
    dirichlet_eigenvalues = dirichlet_eigenvalues[comparison_order]
    dirichlet_modes[:, 4] *= -1.0
    numerical_modes = align_individual_modes(reference_modes, dirichlet_modes, reference_weights)
    records: list[tuple[int, float, float]] = []
    for side in GRID_SIDES:
        points, weights = semi_torus_grid(side)
        basis = reference if side == REFERENCE_SIDE else solve(points, weights)
        interpolated = normalize_columns(basis.transform(reference_points), reference_weights)
        aligned = align_individual_modes(interpolated, reference_modes, reference_weights)
        eigenvalue_error = float(torch.max(torch.abs(basis.eigenvalues - reference.eigenvalues)))
        records.append(
            (
                points.shape[0],
                eigenvalue_error,
                weighted_projector_error(
                    aligned[:, 1:2], reference_modes[:, 1:2], reference_weights
                ),
            )
        )
        last_side, last_basis = side, basis

if __name__ == "__main__" and ifplt:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sizes, eigenvalue_errors, mode_errors = map(np.asarray, zip(*records, strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].loglog(sizes, np.maximum(eigenvalue_errors, 1.0e-16), marker="o")
    axes[0].set(title="Dense eigenvalue convergence", xlabel="reference points", ylabel="max error")
    axes[1].loglog(sizes, np.maximum(mode_errors, 1.0e-16), marker="o")
    axes[1].set(
        title="First nonconstant eigenfunction convergence",
        xlabel="reference points",
        ylabel="projector norm",
    )
    figure.savefig(OUTPUT_ROOT / "semi_torus_rbf_eigenbasis_convergence.png", dpi=160)
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
    figure.savefig(OUTPUT_ROOT / "semi_torus_rbf_eigenbasis_eigenvalues.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(
        3,
        N_COMPONENTS,
        figsize=(3 * N_COMPONENTS + 1.5, 7.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    difference_modes = numerical_modes - dirichlet_modes
    mode_color_limit = max(
        float(torch.abs(values).max().detach().cpu())
        for values in (numerical_modes, dirichlet_modes)
    )
    difference_color_limit = float(torch.abs(difference_modes).max().detach().cpu())
    tick_locations = (0.0, float(torch.pi / 2.0), float(torch.pi))
    tick_labels = ("0", r"$\pi/2$", r"$\pi$")
    for mode_index in range(N_COMPONENTS):
        values = (
            numerical_modes[:, mode_index],
            dirichlet_modes[:, mode_index],
            difference_modes[:, mode_index],
        )
        labels = ("RBF Integral", "LB Dirichlet", "RBF minus LB")
        titles = (
            f"mode {mode_index}\nRBF λ={float(reference.eigenvalues[mode_index]):.3g}",
            f"Dirichlet LB λ={float(dirichlet_eigenvalues[mode_index]):.3g}",
            "difference",
        )
        for row_index, (axis, label, title, value) in enumerate(
            zip(axes[:, mode_index], labels, titles, values, strict=True)
        ):
            image = axis.imshow(
                value.detach().cpu().reshape(REFERENCE_SIDE, REFERENCE_SIDE),
                cmap="coolwarm",
                vmin=-(difference_color_limit if row_index == 2 else mode_color_limit),
                vmax=difference_color_limit if row_index == 2 else mode_color_limit,
                extent=(0.0, float(torch.pi), 0.0, float(2.0 * torch.pi)),
                aspect="auto",
            )
            if row_index == 2:
                difference_image = image
            else:
                mode_image = image
            axis.set_ylabel(label if mode_index == 0 else "")
            axis.set_title(title, fontsize="small")
    for axis in axes[-1]:
        axis.set(xlabel=r"$\phi$", xticks=tick_locations, xticklabels=tick_labels)
    theta_ticks = (0.0, float(torch.pi), float(2.0 * torch.pi))
    theta_labels = ("0", r"$\pi$", r"$2\pi$")
    for axis in axes[:, 0]:
        axis.set(ylabel=axis.get_ylabel(), yticks=theta_ticks, yticklabels=theta_labels)
    figure.colorbar(mode_image, ax=axes[:2].ravel().tolist(), shrink=0.82, label="mode value")
    figure.colorbar(difference_image, ax=axes[2].tolist(), shrink=0.82, label="RBF minus LB")
    figure.savefig(OUTPUT_ROOT / "semi_torus_rbf_eigenbasis_eigenvectors.png", dpi=160)
    plt.close(figure)
