"""
Utility package API.

Public API:
- Common utility helpers listed in ``__all__``.

Internal-only guidance:
- Internal modules should import from concrete modules in ``dymad.utils``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "adj_to_edge",
    "animate",
    "compare_contour",
    "config_logger",
    "ControlInterpolator",
    "CTRL_MAP",
    "JaxWrapper",
    "load_config",
    "make_scheduler",
    "plot_contour",
    "plot_cv_results",
    "plot_hist",
    "plot_multi_trajs",
    "plot_summary",
    "plot_trajectory",
    "setup_logging",
    "TrajectorySampler",
    "X0_MAP",
]

_EXPORTS = {
    "adj_to_edge": "dymad.utils.graph",
    "animate": "dymad.utils.plot",
    "compare_contour": "dymad.utils.plot",
    "config_logger": "dymad.utils.misc",
    "ControlInterpolator": "dymad.utils.control",
    "CTRL_MAP": "dymad.utils.sampling",
    "JaxWrapper": "dymad.utils.wrapper",
    "load_config": "dymad.utils.misc",
    "make_scheduler": "dymad.utils.scheduler",
    "plot_contour": "dymad.utils.plot",
    "plot_cv_results": "dymad.utils.plot",
    "plot_hist": "dymad.utils.plot",
    "plot_multi_trajs": "dymad.utils.plot",
    "plot_summary": "dymad.utils.plot",
    "plot_trajectory": "dymad.utils.plot",
    "setup_logging": "dymad.utils.misc",
    "TrajectorySampler": "dymad.utils.sampling",
    "X0_MAP": "dymad.utils.sampling",
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
