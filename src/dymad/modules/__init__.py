"""
Neural module package API.

Public API:
- Module factories, maps, and module classes listed in ``__all__``.

Internal-only guidance:
- Internal modules should import concrete modules (for example,
  ``dymad.modules.collections`` or ``dymad.modules.kernel``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ACT_MAP",
    "GCL_MAP",
    "INIT_MAP_B",
    "INIT_MAP_W",
    "AE_MAP",
    "FlexLinear",
    "GNN",
    "IdenCatGNN",
    "IdenCatMLP",
    "KernelAbstract",
    "KernelOperatorValued",
    "KernelOperatorValuedScalars",
    "KernelOpSeparable",
    "KernelOpTangent",
    "KernelScalarValued",
    "KernelScDM",
    "KernelScExp",
    "KernelScRBF",
    "KRRBase",
    "KRRMultiOutputIndep",
    "KRRMultiOutputShared",
    "KRROperatorValued",
    "KRRTangent",
    "make_autoencoder",
    "make_kernel",
    "make_krr",
    "make_network",
    "MLP",
    "NN_MAP",
    "ResBlockGNN",
    "ResBlockMLP",
    "scaled_cdist",
    "SequentialBase",
    "SimpleRNN",
    "StepwiseModel",
    "VanillaRNN",
]

_EXPORTS = {
    "ACT_MAP": "dymad.modules.helpers",
    "GCL_MAP": "dymad.modules.helpers",
    "INIT_MAP_B": "dymad.modules.helpers",
    "INIT_MAP_W": "dymad.modules.helpers",
    "AE_MAP": "dymad.modules.collections",
    "FlexLinear": "dymad.modules.linear",
    "GNN": "dymad.modules.gnn",
    "IdenCatGNN": "dymad.modules.gnn",
    "IdenCatMLP": "dymad.modules.mlp",
    "KernelAbstract": "dymad.modules.kernel",
    "KernelOperatorValued": "dymad.modules.kernel",
    "KernelOperatorValuedScalars": "dymad.modules.kernel",
    "KernelOpSeparable": "dymad.modules.kernel",
    "KernelOpTangent": "dymad.modules.kernel",
    "KernelScalarValued": "dymad.modules.kernel",
    "KernelScDM": "dymad.modules.kernel",
    "KernelScExp": "dymad.modules.kernel",
    "KernelScRBF": "dymad.modules.kernel",
    "KRRBase": "dymad.modules.krr",
    "KRRMultiOutputIndep": "dymad.modules.krr",
    "KRRMultiOutputShared": "dymad.modules.krr",
    "KRROperatorValued": "dymad.modules.krr",
    "KRRTangent": "dymad.modules.krr",
    "make_autoencoder": "dymad.modules.collections",
    "make_kernel": "dymad.modules.collections",
    "make_krr": "dymad.modules.collections",
    "make_network": "dymad.modules.collections",
    "MLP": "dymad.modules.mlp",
    "NN_MAP": "dymad.modules.collections",
    "ResBlockGNN": "dymad.modules.gnn",
    "ResBlockMLP": "dymad.modules.mlp",
    "scaled_cdist": "dymad.modules.kernel",
    "SequentialBase": "dymad.modules.sequential",
    "SimpleRNN": "dymad.modules.sequential",
    "StepwiseModel": "dymad.modules.sequential",
    "VanillaRNN": "dymad.modules.sequential",
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
