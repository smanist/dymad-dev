"""
SAKO package API.

Public API:
- Spectral analysis interfaces and utilities listed in ``__all__``.

Internal-only guidance:
- Internal modules should import from concrete modules in ``dymad.sako``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "estimate_pseudospectrum",
    "filter_spectrum",
    "per_state_err",
    "RALowRank",
    "resolvent_analysis",
    "SAInterface",
    "SAKO",
    "SpectralAnalysis",
]

_EXPORTS = {
    "estimate_pseudospectrum": "dymad.sako.rals",
    "filter_spectrum": "dymad.sako.base",
    "per_state_err": "dymad.sako.base",
    "RALowRank": "dymad.sako.rals",
    "resolvent_analysis": "dymad.sako.rals",
    "SAInterface": "dymad.sako.base",
    "SAKO": "dymad.sako.sako",
    "SpectralAnalysis": "dymad.sako.base",
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
