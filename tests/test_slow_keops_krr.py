import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.modules import make_krr
from dymad.numerics import ManifoldAnalytical, tangent_2torus

_KEOPS_CACHE = Path(tempfile.gettempdir()) / "dymad_keops_cache"
_KEOPS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))
os.environ.setdefault("PYKEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))

pytestmark = pytest.mark.slow


def _grid_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(axis, axis)
    X = np.column_stack((xx.ravel(), yy.ravel()))
    Y = np.column_stack(
        (
            X[:, 0] ** 2 + np.sin(X[:, 1]),
            0.5 * X[:, 0] - X[:, 1] ** 2,
        )
    )
    query_axis = np.linspace(0.05, 0.95, 5)
    qx, qy = np.meshgrid(query_axis, query_axis)
    Xquery = np.column_stack((qx.ravel(), qy.ravel()))
    return X, Y, Xquery


def _assert_keops_matrix_free_matches_dense(dense_cfg, keops_cfg) -> None:
    X, Y, Xquery = _grid_data()
    dense = make_krr(**dense_cfg, solver="dense_cholesky")
    matrix_free = make_krr(
        **keops_cfg,
        solver="matrix_free_cg",
        cg_rtol=1e-12,
        cg_max_iter=500,
    )
    dense.set_train_data(X, Y)
    matrix_free.set_train_data(X, Y)

    dense.fit()
    matrix_free.fit()
    with torch.no_grad():
        query = torch.as_tensor(Xquery, dtype=torch.float64)
        dense_pred = dense(query)
        matrix_free_pred = matrix_free(query)

    assert matrix_free._cg_diagnostics is not None
    assert matrix_free._cg_diagnostics["converged"]
    assert torch.allclose(matrix_free_pred, dense_pred, rtol=1e-7, atol=1e-8)


def test_keops_matrix_free_shared_krr_matches_dense() -> None:
    pytest.importorskip("pykeops")
    dense_cfg = {
        "type": "share",
        "kernel": {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": 0.6},
        "dtype": torch.float64,
        "ridge_init": 1e-6,
    }
    keops_cfg = {
        **dense_cfg,
        "kernel": {
            **dense_cfg["kernel"],
            "backend": "keops",
        },
    }

    _assert_keops_matrix_free_matches_dense(dense_cfg, keops_cfg)


def test_keops_matrix_free_operator_krr_matches_dense() -> None:
    pytest.importorskip("pykeops")
    scalar_cfg = {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": 0.6}
    dense_cfg = {
        "type": "opval",
        "kernel": {
            "type": "op_sep",
            "input_dim": 2,
            "output_dim": 2,
            "kopts": [scalar_cfg],
            "Ls": np.asarray([[[1.0, 0.0], [0.25, 0.8]]]),
        },
        "dtype": torch.float64,
        "ridge_init": 1e-2,
    }
    keops_cfg = {
        **dense_cfg,
        "kernel": {
            **dense_cfg["kernel"],
            "kopts": [{**scalar_cfg, "backend": "keops"}],
        },
    }

    _assert_keops_matrix_free_matches_dense(dense_cfg, keops_cfg)


def test_keops_matrix_free_tangent_krr_matches_dense() -> None:
    pytest.importorskip("pykeops")
    rng = np.random.default_rng(5)
    angles = rng.random((80, 2)) * 2.0 * np.pi
    major_radius = 2.0
    X = np.column_stack(
        (
            (np.cos(angles[:, 0]) + major_radius) * np.cos(angles[:, 1]),
            (np.cos(angles[:, 0]) + major_radius) * np.sin(angles[:, 1]),
            np.sin(angles[:, 0]),
        )
    )
    T = tangent_2torus(X, major_radius)
    F = np.column_stack((X[:, 0] ** 2 / 4.0 + X[:, 1] ** 2, X[:, 1] * X[:, 2]))
    Y = np.einsum("ij,ijk->ik", F, T)
    Xtrain, Ytrain = X[:56], Y[:56]
    Xquery = X[56:72]
    dense_cfg = {
        "type": "tangent",
        "kernel": {
            "type": "op_tan",
            "input_dim": 3,
            "output_dim": 3,
            "kopts": {"type": "sc_rbf", "input_dim": 3, "lengthscale_init": 1.0},
        },
        "dtype": torch.float64,
        "ridge_init": 1e-2,
    }
    keops_cfg = {
        **dense_cfg,
        "kernel": {
            **dense_cfg["kernel"],
            "kopts": {
                **dense_cfg["kernel"]["kopts"],
                "backend": "keops",
            },
        },
    }
    dense = make_krr(**dense_cfg, solver="dense_cholesky")
    matrix_free = make_krr(
        **keops_cfg,
        solver="matrix_free_cg",
        cg_rtol=1e-12,
        cg_max_iter=500,
    )
    dense.set_train_data(Xtrain, Ytrain)
    matrix_free.set_train_data(Xtrain, Ytrain)
    dense_manifold = ManifoldAnalytical(Xtrain, d=2, fT=lambda x: tangent_2torus(x, major_radius))
    keops_manifold = ManifoldAnalytical(
        Xtrain,
        d=2,
        fT=lambda x: tangent_2torus(x, major_radius),
    )
    dense_manifold.precompute()
    keops_manifold.precompute()
    dense.set_manifold(dense_manifold)
    matrix_free.set_manifold(keops_manifold)

    dense.fit()
    matrix_free.fit()
    with torch.no_grad():
        query = torch.as_tensor(Xquery, dtype=torch.float64)
        dense_pred = dense(query)
        matrix_free_pred = matrix_free(query)

    assert matrix_free._cg_diagnostics is not None
    assert matrix_free._cg_diagnostics["converged"]
    assert torch.allclose(matrix_free_pred, dense_pred, rtol=1e-7, atol=1e-8)
