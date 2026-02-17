"""
Transform package API.

Public API:
- Transform classes and factory utilities listed in ``__all__``.

Internal-only guidance:
- Internal modules should import from concrete modules in ``dymad.transform``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AddOne",
    "Autoencoder",
    "Compose",
    "DelayEmbedder",
    "DiffMap",
    "DiffMapVB",
    "Identity",
    "Isomap",
    "Lift",
    "make_transform",
    "Scaler",
    "SVD",
    "TRN_MAP",
]

_EXPORTS = {
    "AddOne": "dymad.transform.base",
    "Autoencoder": "dymad.transform.base",
    "Compose": "dymad.transform.collection",
    "DelayEmbedder": "dymad.transform.base",
    "DiffMap": "dymad.transform.ndr",
    "DiffMapVB": "dymad.transform.ndr",
    "Identity": "dymad.transform.base",
    "Isomap": "dymad.transform.ndr",
    "Lift": "dymad.transform.base",
    "make_transform": "dymad.transform.collection",
    "Scaler": "dymad.transform.base",
    "SVD": "dymad.transform.base",
    "TRN_MAP": "dymad.transform.collection",
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
