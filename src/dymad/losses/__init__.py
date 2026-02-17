"""
Loss package API.

Public API:
- LOSS_MAP
- vpt_loss, VPTLoss
- wmse_loss, WMSELoss

Internal-only guidance:
- Internal modules should import from ``dymad.losses.losses`` directly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LOSS_MAP",
    "vpt_loss",
    "VPTLoss",
    "wmse_loss",
    "WMSELoss",
]

_EXPORTS = {
    "LOSS_MAP": "dymad.losses.losses",
    "vpt_loss": "dymad.losses.losses",
    "VPTLoss": "dymad.losses.losses",
    "wmse_loss": "dymad.losses.losses",
    "WMSELoss": "dymad.losses.losses",
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
