"""
Numerics package API.

Public API:
- Numerical routines and helpers listed in ``__all__``.

Internal-only guidance:
- Internal modules should import concrete modules (for example,
  ``dymad.numerics.linalg``) instead of package re-exports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "central_diff",
    "check_direction",
    "check_orthogonality",
    "complex_grid",
    "complex_map",
    "complex_plot",
    "complex_step",
    "DimensionEstimator",
    "disc2cont",
    "DM",
    "DMF",
    "eig_low_rank",
    "expm_full_rank",
    "expm_low_rank",
    "fe_step",
    "generate_coef",
    "generate_weak_weights",
    "logm_low_rank",
    "make_random_matrix",
    "Manifold",
    "ManifoldAltTree",
    "ManifoldAnalytical",
    "mode_split",
    "randomized_svd",
    "rational_kernel",
    "real_lowrank_from_eigpairs",
    "rk4_step",
    "scaled_eig",
    "tangent_1circle",
    "tangent_2torus",
    "torch_jacobian",
    "truncate_sequence",
    "truncated_svd",
    "VBDM",
]

_EXPORTS = {
    "central_diff": "dymad.numerics.gradients",
    "check_direction": "dymad.numerics.linalg",
    "check_orthogonality": "dymad.numerics.linalg",
    "complex_grid": "dymad.numerics.complex",
    "complex_map": "dymad.numerics.complex",
    "complex_plot": "dymad.numerics.complex",
    "complex_step": "dymad.numerics.gradients",
    "DimensionEstimator": "dymad.numerics.manifold",
    "disc2cont": "dymad.numerics.complex",
    "DM": "dymad.numerics.dm",
    "DMF": "dymad.numerics.dm",
    "eig_low_rank": "dymad.numerics.linalg",
    "expm_full_rank": "dymad.numerics.linalg",
    "expm_low_rank": "dymad.numerics.linalg",
    "fe_step": "dymad.numerics.time_int",
    "generate_coef": "dymad.numerics.spectrum",
    "generate_weak_weights": "dymad.numerics.weak",
    "logm_low_rank": "dymad.numerics.linalg",
    "make_random_matrix": "dymad.numerics.linalg",
    "Manifold": "dymad.numerics.manifold",
    "ManifoldAltTree": "dymad.numerics.manifold",
    "ManifoldAnalytical": "dymad.numerics.manifold",
    "mode_split": "dymad.numerics.linalg",
    "randomized_svd": "dymad.numerics.linalg",
    "rational_kernel": "dymad.numerics.spectrum",
    "real_lowrank_from_eigpairs": "dymad.numerics.linalg",
    "rk4_step": "dymad.numerics.time_int",
    "scaled_eig": "dymad.numerics.linalg",
    "tangent_1circle": "dymad.numerics.manifold",
    "tangent_2torus": "dymad.numerics.manifold",
    "torch_jacobian": "dymad.numerics.gradients",
    "truncate_sequence": "dymad.numerics.linalg",
    "truncated_svd": "dymad.numerics.linalg",
    "VBDM": "dymad.numerics.dm",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
