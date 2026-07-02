"""Compare dense and KeOps matrix-free KRR solve paths."""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

_KEOPS_CACHE = Path(tempfile.gettempdir()) / "dymad_keops_cache"
_KEOPS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))
os.environ.setdefault("PYKEOPS_CACHE_FOLDER", str(_KEOPS_CACHE))
_MPL_CACHE = Path(tempfile.gettempdir()) / "dymad_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from dymad.modules import make_krr
from dymad.numerics import ManifoldAnalytical, tangent_2torus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cg-rtol", type=float, default=1e-10)
    parser.add_argument("--cg-max-iter", type=int, default=1000)
    return parser.parse_args()


def _time_call(fn: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _grid_data(n_train: int, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.random((n_train, 2))
    Y = np.column_stack((X[:, 0] ** 2 + np.sin(X[:, 1]), 0.5 * X[:, 0] - X[:, 1] ** 2))
    Xtest = rng.random((n_test, 2))
    return X, Y, Xtest


def _torus_data(n_train: int, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = rng.random((n_train + n_test, 2)) * 2.0 * np.pi
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
    return X[:n_train], Y[:n_train], X[n_train:]


def _run_case(
    name: str,
    dense_cfg: dict,
    keops_cfg: dict,
    X: np.ndarray,
    Y: np.ndarray,
    Xtest: np.ndarray,
    *,
    cg_rtol: float,
    cg_max_iter: int,
    manifold_factory: Callable[[], ManifoldAnalytical] | None = None,
) -> dict[str, float | int | str | bool]:
    dense = make_krr(**dense_cfg, solver="dense_cholesky")
    keops_dense = make_krr(**keops_cfg, solver="dense_cholesky")
    matrix_free = make_krr(
        **keops_cfg,
        solver="matrix_free_cg",
        cg_rtol=cg_rtol,
        cg_max_iter=cg_max_iter,
    )
    for model in (dense, keops_dense, matrix_free):
        model.set_train_data(X, Y)
        if manifold_factory is not None:
            model.set_manifold(manifold_factory())

    _, dense_solve = _time_call(dense.fit)
    _, keops_materialized_solve = _time_call(keops_dense.fit)
    _, cg_solve = _time_call(matrix_free.fit)
    query = torch.as_tensor(Xtest, dtype=torch.float64)
    dense_pred, dense_predict = _time_call(lambda: dense(query).detach())
    _keops_pred, keops_materialized_predict = _time_call(lambda: keops_dense(query).detach())
    cg_pred, cg_predict = _time_call(lambda: matrix_free(query).detach())
    rel_error = torch.linalg.norm(cg_pred - dense_pred) / torch.linalg.norm(dense_pred)
    residual = matrix_free._cg_diagnostics or {}
    return {
        "case": name,
        "dense_solve_s": dense_solve,
        "keops_materialized_solve_s": keops_materialized_solve,
        "cg_solve_s": cg_solve,
        "dense_predict_s": dense_predict,
        "keops_materialized_predict_s": keops_materialized_predict,
        "cg_predict_s": cg_predict,
        "cg_iterations": int(residual.get("iterations", -1)),
        "cg_residual": float(residual.get("residual_norm", float("nan"))),
        "cg_converged": bool(residual.get("converged", False)),
        "dense_relative_error": float(rel_error),
    }


def main() -> int:
    args = parse_args()
    try:
        import pykeops  # noqa: F401
    except ImportError:
        print("PyKeOps is not installed. Install dymad[keops] to run this benchmark.")
        return 2

    X, Y, Xtest = _grid_data(args.n_train, args.n_test, args.seed)
    scalar_dense = {
        "type": "share",
        "kernel": {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": 0.6},
        "dtype": torch.float64,
        "ridge_init": 1e-6,
    }
    scalar_keops = {
        **scalar_dense,
        "kernel": {**scalar_dense["kernel"], "backend": "keops"},
    }
    op_dense = {
        "type": "opval",
        "kernel": {
            "type": "op_sep",
            "input_dim": 2,
            "output_dim": 2,
            "kopts": [scalar_dense["kernel"]],
            "Ls": np.asarray([[[1.0, 0.0], [0.25, 0.8]]]),
        },
        "dtype": torch.float64,
        "ridge_init": 1e-2,
    }
    op_keops = {
        **op_dense,
        "kernel": {
            **op_dense["kernel"],
            "kopts": [{**scalar_dense["kernel"], "backend": "keops"}],
        },
    }
    rows = [
        _run_case(
            "shared_scalar",
            scalar_dense,
            scalar_keops,
            X,
            Y,
            Xtest,
            cg_rtol=args.cg_rtol,
            cg_max_iter=args.cg_max_iter,
        ),
        _run_case(
            "separable_operator",
            op_dense,
            op_keops,
            X,
            Y,
            Xtest,
            cg_rtol=args.cg_rtol,
            cg_max_iter=args.cg_max_iter,
        ),
    ]

    major_radius = 2.0
    Xt, Yt, Xttest = _torus_data(args.n_train, args.n_test, args.seed + 1)
    tangent_dense = {
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
    tangent_keops = {
        **tangent_dense,
        "kernel": {
            **tangent_dense["kernel"],
            "kopts": {**tangent_dense["kernel"]["kopts"], "backend": "keops"},
        },
    }
    rows.append(
        _run_case(
            "tangent",
            tangent_dense,
            tangent_keops,
            Xt,
            Yt,
            Xttest,
            cg_rtol=args.cg_rtol,
            cg_max_iter=args.cg_max_iter,
            manifold_factory=lambda: ManifoldAnalytical(
                Xt,
                d=2,
                fT=lambda x: tangent_2torus(x, major_radius),
            ),
        )
    )

    fields = list(rows[0])
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[field]) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
