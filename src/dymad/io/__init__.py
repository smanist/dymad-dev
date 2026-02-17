"""
I/O package API.

Public API:
- DynData, DataInterface
- load_model, visualize_model
- TrajectoryManager, TrajectoryManagerGraph

Internal-only guidance:
- Internal modules should import from concrete modules (for example, ``dymad.io.data``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DataInterface",
    "DynData",
    "load_model",
    "TrajectoryManager",
    "TrajectoryManagerGraph",
    "visualize_model",
]

_EXPORTS = {
    "DataInterface": "dymad.io.checkpoint",
    "DynData": "dymad.io.data",
    "load_model": "dymad.io.checkpoint",
    "TrajectoryManager": "dymad.io.trajectory_manager",
    "TrajectoryManagerGraph": "dymad.io.trajectory_manager",
    "visualize_model": "dymad.io.checkpoint",
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
