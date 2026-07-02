import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.modules import KernelOpSeparable, KernelOpTangent, KernelScDM, KernelScExp, KernelScRBF

_KEOPS_CACHE = Path(tempfile.gettempdir()) / "dymad_keops_cache"
_KEOPS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))
os.environ.setdefault("PYKEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))


def _points() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.9, 0.1]], dtype=torch.float64)
    Z = torch.tensor([[0.1, 0.2], [0.45, 0.65], [0.8, 0.25], [1.0, 0.9]], dtype=torch.float64)
    values = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.4, 0.3], [1.2, -0.1]],
        dtype=torch.float64,
    )
    return X, Z, values


@pytest.mark.parametrize(
    "kernel",
    [
        KernelScRBF(in_dim=2, lengthscale_init=0.4, dtype=torch.float64),
        KernelScExp(in_dim=2, lengthscale_init=0.6, dtype=torch.float64),
    ],
)
def test_scalar_apply_matches_materialized_block(kernel) -> None:
    X, Z, values = _points()

    actual = kernel.apply(X, Z, values)
    expected = kernel(X, Z) @ values

    assert torch.allclose(actual, expected)


def test_scdm_apply_matches_materialized_block() -> None:
    Xref = torch.tensor(
        [[0.0, 0.0], [0.25, 0.5], [0.5, 0.25], [0.9, 0.1], [1.0, 1.0]],
        dtype=torch.float64,
    )
    X = Xref[:3]
    values = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.4, 0.3], [1.2, -0.1], [0.1, 0.4]],
        dtype=torch.float64,
    )
    kernel = KernelScDM(in_dim=2, eps_init=0.2, dtype=torch.float64)
    kernel.set_reference_data(Xref)

    actual = kernel.apply(X, Xref, values)
    expected = kernel(X, Xref) @ values

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    "torch_kernel,keops_kernel",
    [
        (
            KernelScRBF(in_dim=2, lengthscale_init=0.4, dtype=torch.float64),
            KernelScRBF(in_dim=2, lengthscale_init=0.4, dtype=torch.float64, backend="keops"),
        ),
        (
            KernelScExp(in_dim=2, lengthscale_init=0.6, dtype=torch.float64),
            KernelScExp(in_dim=2, lengthscale_init=0.6, dtype=torch.float64, backend="keops"),
        ),
    ],
)
def test_keops_scalar_apply_matches_torch_apply(torch_kernel, keops_kernel) -> None:
    pytest.importorskip("pykeops")
    X, Z, values = _points()

    expected = torch_kernel.apply(X, Z, values)
    actual = keops_kernel.apply(X, Z, values)

    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_keops_scdm_apply_matches_torch_apply() -> None:
    pytest.importorskip("pykeops")
    Xref = torch.tensor(
        [[0.0, 0.0], [0.25, 0.5], [0.5, 0.25], [0.9, 0.1], [1.0, 1.0]],
        dtype=torch.float64,
    )
    X = Xref[:3]
    values = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.4, 0.3], [1.2, -0.1], [0.1, 0.4]],
        dtype=torch.float64,
    )
    dense = KernelScDM(in_dim=2, eps_init=0.2, dtype=torch.float64)
    keops = KernelScDM(in_dim=2, eps_init=0.2, dtype=torch.float64, backend="keops")
    dense.set_reference_data(Xref)
    keops.set_reference_data(Xref)

    expected = dense.apply(X, Xref, values)
    actual = keops.apply(X, Xref, values)

    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_keops_scdm_batched_apply_matches_torch_apply() -> None:
    pytest.importorskip("pykeops")
    Xref = torch.tensor(
        [[0.0, 0.0], [0.25, 0.5], [0.5, 0.25], [0.9, 0.1], [1.0, 1.0]],
        dtype=torch.float64,
    )
    X = torch.tensor(
        [
            [[0.02, 0.03], [0.31, 0.45], [0.82, 0.18]],
            [[0.08, 0.10], [0.42, 0.30], [0.95, 0.70]],
        ],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [
            [[1.0, -0.5], [0.2, 0.7], [-0.4, 0.3], [1.2, -0.1], [0.1, 0.4]],
            [[0.3, 0.6], [-0.8, 0.1], [0.4, -0.2], [0.5, 0.9], [-0.1, 0.2]],
        ],
        dtype=torch.float64,
    )
    dense = KernelScDM(in_dim=2, eps_init=0.2, dtype=torch.float64)
    keops = KernelScDM(in_dim=2, eps_init=0.2, dtype=torch.float64, backend="keops")
    dense.set_reference_data(Xref)
    keops.set_reference_data(Xref)

    expected = dense.apply(X, Xref, values)
    actual = keops.apply(X, Xref, values)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_separable_operator_apply_matches_materialized_contraction() -> None:
    X, Z, _ = _points()
    values = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.4, 0.3], [1.2, -0.1]],
        dtype=torch.float64,
    )
    kernel = KernelOpSeparable(
        [
            KernelScRBF(in_dim=2, lengthscale_init=0.4, dtype=torch.float64),
            KernelScRBF(in_dim=2, lengthscale_init=0.8, dtype=torch.float64),
        ],
        out_dim=2,
        Ls=torch.tensor(
            [
                [[1.0, 0.0], [0.3, 0.5]],
                [[0.7, 0.0], [-0.2, 0.9]],
            ],
            dtype=torch.float64,
        ),
        dtype=torch.float64,
    )

    actual = kernel.apply(X, Z, values)
    expected = torch.einsum("n a m b, m b -> n a", kernel(X, Z), values)

    assert torch.allclose(actual, expected)


class _ConstantTangentManifold:
    _Nman = 1

    def _estimate_tangent(self, X: np.ndarray) -> np.ndarray:
        basis = np.zeros((X.shape[0], 1, 2), dtype=float)
        basis[:, 0, 0] = 1.0
        return basis


def test_tangent_intrinsic_apply_matches_materialized_contraction() -> None:
    X, Z, _ = _points()
    values = torch.tensor([[1.0], [0.2], [-0.4], [1.2]], dtype=torch.float64)
    kernel = KernelOpTangent(
        KernelScRBF(in_dim=2, lengthscale_init=0.4, dtype=torch.float64),
        out_dim=2,
        dtype=torch.float64,
    )
    kernel.set_manifold(_ConstantTangentManifold())

    actual, _Tx = kernel.intrinsic_apply(X, Z, values)
    materialized, _Tx, _Tz = kernel(X, Z)
    expected = torch.einsum("n a m b, m b -> n a", materialized, values)

    assert torch.allclose(actual, expected)
