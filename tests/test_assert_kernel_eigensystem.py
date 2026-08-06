import math

import pytest
import torch

from dymad.kernel_analysis import KernelEigenbasis
from dymad.modules import KernelOpSeparable, KernelScDM, KernelScRBF


def _circle_angles(count: int = 48, *, phase: float = 0.0) -> torch.Tensor:
    return torch.arange(count, dtype=torch.float64) * (2.0 * math.pi / count) + phase


def _circle_points(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def _reference_points(count: int = 48) -> torch.Tensor:
    return _circle_points(_circle_angles(count))


def _kernel() -> KernelScRBF:
    return KernelScRBF(in_dim=2, lengthscale_init=0.22, dtype=torch.float64)


def _dm_kernel() -> KernelScDM:
    return KernelScDM(in_dim=2, eps_init=0.08, alpha_init=1.0, dtype=torch.float64)


def _circle_fourier_modes(angles: torch.Tensor, reference_count: int) -> torch.Tensor:
    scale = math.sqrt(2.0 / reference_count)
    return torch.stack(
        (
            torch.full_like(angles, 1.0 / math.sqrt(reference_count)),
            scale * angles.cos(),
            scale * angles.sin(),
        ),
        dim=1,
    )


def _circle_fourier_derivatives(angles: torch.Tensor, reference_count: int) -> torch.Tensor:
    scale = math.sqrt(2.0 / reference_count)
    return torch.stack(
        (
            torch.zeros_like(angles),
            -scale * angles.sin(),
            scale * angles.cos(),
        ),
        dim=1,
    )


def _orthogonal_alignment(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(actual.T @ expected)
    return left @ right


def _circle_eigenvalues(kernel: KernelScRBF | KernelScDM, reference: torch.Tensor) -> torch.Tensor:
    """Return the constant and first-harmonic eigenvalues of a circulant Gram matrix."""

    gram = kernel(reference, reference)
    angles = _circle_angles(reference.shape[0])
    return torch.stack(tuple((gram[0] * (mode * angles).cos()).sum() for mode in (0, 1, 1)))


def test_dense_gram_eigenbasis_matches_unit_circle_fourier_eigensystem() -> None:
    reference = _reference_points()
    basis = KernelEigenbasis(_kernel(), 3).solve(reference)
    query_angles = _circle_angles(17, phase=0.17)
    recovered = basis.transform(_circle_points(query_angles))
    expected = _circle_fourier_modes(query_angles, reference.shape[0])
    alignment = _orthogonal_alignment(recovered, expected)

    assert basis.eigenvalues.shape == (3,)
    assert basis.eigenvectors.shape == (reference.shape[0], 3)
    assert torch.allclose(
        basis.eigenvalues,
        _circle_eigenvalues(basis.kernel, reference),
        rtol=1e-11,
        atol=1e-12,
    )
    assert torch.allclose(recovered @ alignment, expected, rtol=1e-10, atol=1e-11)
    assert torch.allclose(
        basis.transform(reference), basis.reference_eigenfunctions, rtol=1e-10, atol=1e-10
    )
    assert torch.allclose(
        basis.eigenvectors.T @ basis.eigenvectors,
        torch.eye(3, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_nystrom_extension_gradients_match_unit_circle_fourier_derivatives() -> None:
    reference = _reference_points()
    basis = KernelEigenbasis(_kernel(), 3).solve(reference)
    query_angles = _circle_angles(17, phase=0.17).requires_grad_()
    recovered = basis.transform(_circle_points(query_angles))
    expected = _circle_fourier_modes(query_angles, reference.shape[0])
    alignment = _orthogonal_alignment(recovered.detach(), expected.detach())
    derivatives = torch.stack(
        tuple(
            torch.autograd.grad(recovered[:, component].sum(), query_angles, retain_graph=True)[0]
            for component in range(recovered.shape[1])
        ),
        dim=1,
    )

    assert torch.allclose(
        derivatives @ alignment,
        _circle_fourier_derivatives(query_angles, reference.shape[0]),
        rtol=1e-9,
        atol=1e-10,
    )


def test_weighted_eigenbasis_uses_symmetric_quadrature_operator() -> None:
    reference = _reference_points(15)
    weights = torch.linspace(1.0, 2.0, 15, dtype=torch.float64)
    basis = KernelEigenbasis(_kernel(), 3).solve(reference, sample_weights=weights)

    gram = _kernel()(reference, reference)
    matrix = weights.sqrt()[:, None] * gram * weights.sqrt()[None, :]
    expected_values, _ = torch.linalg.eigh(matrix)

    assert torch.allclose(basis.eigenvalues, expected_values[-3:].flip(0))
    assert torch.allclose(
        basis.transform(reference), basis.reference_eigenfunctions, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("solver", ["scipy", "matrix_free"])
def test_iterative_solvers_match_dense_eigenvalues(solver: str) -> None:
    reference = _reference_points(32)
    dense = KernelEigenbasis(_kernel(), 3).solve(reference)
    iterative = KernelEigenbasis(_kernel(), 3).solve(
        reference, solver=solver, max_iterations=20 if solver == "matrix_free" else None, seed=0
    )

    assert torch.allclose(iterative.eigenvalues, dense.eigenvalues, rtol=1e-8, atol=1e-10)
    assert iterative.diagnostics["converged"] is True


@pytest.mark.parametrize("solver", ["dense", "scipy", "matrix_free"])
def test_dm_kernel_eigenvalues_match_unit_circle_fourier_spectrum(solver: str) -> None:
    reference = _reference_points(32)
    expected_kernel = _dm_kernel()
    expected_kernel.set_reference_data(reference)
    expected = _circle_eigenvalues(expected_kernel, reference)
    basis = KernelEigenbasis(_dm_kernel(), 3).solve(
        reference,
        solver=solver,
        max_iterations=20 if solver == "matrix_free" else None,
        seed=0,
    )

    assert torch.allclose(basis.eigenvalues, expected, rtol=1e-8, atol=1e-10)
    assert basis.diagnostics["converged"] is True


def test_keops_matrix_free_solver_matches_dense_when_available() -> None:
    pytest.importorskip("pykeops")
    reference = _reference_points(20)
    dense = KernelEigenbasis(_kernel(), 3).solve(reference)
    keops_kernel = KernelScRBF(
        in_dim=2, lengthscale_init=0.22, dtype=torch.float64, backend="keops"
    )
    iterative = KernelEigenbasis(keops_kernel, 3).solve(
        reference, solver="matrix_free", max_iterations=20, seed=0
    )

    assert torch.allclose(iterative.eigenvalues, dense.eigenvalues, rtol=1e-8, atol=1e-10)


def test_analysis_rejects_uninitialized_kernel_parameters_without_mutation() -> None:
    kernel = KernelScRBF(in_dim=2, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="lengthscale_init"):
        KernelEigenbasis(kernel, 2).solve(_reference_points())

    assert kernel._log_ell.numel() == 0


def test_eigenbasis_rejects_operator_valued_kernel() -> None:
    scalar = _kernel()
    operator = KernelOpSeparable(scalar, out_dim=2, dtype=torch.float64)

    with pytest.raises(TypeError, match="scalar-valued"):
        KernelEigenbasis(operator, 2)  # type: ignore[arg-type]


def test_eigenbasis_state_dict_contains_fixed_reference_state() -> None:
    basis = KernelEigenbasis(_kernel(), 2).solve(_reference_points())
    state = basis.state_dict()

    assert "reference_points" in state
    assert "eigenvalues" in state
    assert "kernel._log_ell" in state


@pytest.mark.parametrize("kernel_factory", [_kernel, _dm_kernel], ids=("rbf", "dm"))
def test_eigenbasis_state_dict_round_trip_restores_fixed_basis(kernel_factory) -> None:
    reference = _reference_points(18)
    weights = torch.linspace(1.0, 2.0, reference.shape[0], dtype=torch.float64)
    basis = KernelEigenbasis(kernel_factory(), 3).solve(reference, sample_weights=weights)
    query = _circle_points(_circle_angles(11, phase=0.13))
    expected = basis.transform(query)

    restored = KernelEigenbasis(kernel_factory(), 3)
    restored.load_state_dict(basis.state_dict())

    assert torch.equal(restored.reference_points, basis.reference_points)
    assert torch.equal(restored.sample_weights, basis.sample_weights)
    assert torch.equal(restored.eigenvalues, basis.eigenvalues)
    assert torch.allclose(restored.transform(query), expected, rtol=1e-12, atol=1e-12)
