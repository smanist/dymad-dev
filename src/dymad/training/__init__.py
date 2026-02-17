"""
Training package API.

Public API:
- Driver, optimizer, and trainer utilities listed in ``__all__``.

Internal-only guidance:
- Internal modules should import from concrete modules in ``dymad.training``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "aggregate_cv_results",
    "CVResult",
    "DriverBase",
    "iter_param_grid",
    "LinearTrainer",
    "LSUpdater",
    "NODETrainer",
    "OptBase",
    "OptLinear",
    "OptNODE",
    "OptWeakForm",
    "RunState",
    "set_by_dotted_key",
    "SingleSplitDriver",
    "SOL_MAP",
    "StackedOpt",
    "StackedTrainer",
    "TrainerBase",
    "WeakFormTrainer",
]

_EXPORTS = {
    "aggregate_cv_results": "dymad.training.helper",
    "CVResult": "dymad.training.helper",
    "DriverBase": "dymad.training.driver",
    "iter_param_grid": "dymad.training.helper",
    "LinearTrainer": "dymad.training.trainer",
    "LSUpdater": "dymad.training.ls_update",
    "NODETrainer": "dymad.training.trainer",
    "OptBase": "dymad.training.opt_base",
    "OptLinear": "dymad.training.opt_linear",
    "OptNODE": "dymad.training.opt_node",
    "OptWeakForm": "dymad.training.opt_weak_form",
    "RunState": "dymad.training.helper",
    "set_by_dotted_key": "dymad.training.helper",
    "SingleSplitDriver": "dymad.training.driver",
    "SOL_MAP": "dymad.training.ls_update",
    "StackedOpt": "dymad.training.stacked_opt",
    "StackedTrainer": "dymad.training.trainer",
    "TrainerBase": "dymad.training.driver",
    "WeakFormTrainer": "dymad.training.trainer",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        if name == "TrainerBase":
            value = getattr(module, "DriverBase")
        else:
            value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
