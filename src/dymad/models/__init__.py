"""
Model package API.

Public API:
- Model collections, recipes, and prediction helpers listed in ``__all__``.

Internal-only guidance:
- Internal modules should import from concrete modules (for example,
  ``dymad.models.collections`` or ``dymad.models.prediction``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "build_model",
    "get_dims",
    "ComposedDynamics",
    "Composer",
    "Decoder",
    "Encoder",
    "Features",
    "Predictor",
    "DEC_MAP",
    "DYN_MAP",
    "ENC_MAP",
    "FZU_MAP",
    "LIN_MAP",
    "CD_KM",
    "CD_KMM",
    "CD_KMSK",
    "CD_LDM",
    "CD_LFM",
    "CD_SDM",
    "DGKBF",
    "DGKM",
    "DGKMSK",
    "DGLDM",
    "DGLTI",
    "DKBF",
    "DKM",
    "DKMSK",
    "DLDM",
    "DLDMG",
    "DLTI",
    "DSDM",
    "DSDMG",
    "GKBF",
    "GKM",
    "GLDM",
    "GLTI",
    "KBF",
    "KM",
    "KMM",
    "LDM",
    "LDMG",
    "LTI",
    "PredefinedModel",
    "TemplateCorrAlg",
    "TemplateCorrDif",
    "predict_continuous",
    "predict_continuous_exp",
    "predict_continuous_fenc",
    "predict_continuous_np",
    "predict_discrete",
    "predict_discrete_exp",
]

_EXPORTS = {
    "build_model": "dymad.models.helpers",
    "get_dims": "dymad.models.helpers",
    "ComposedDynamics": "dymad.models.model_base",
    "Composer": "dymad.models.model_base",
    "Decoder": "dymad.models.model_base",
    "Encoder": "dymad.models.model_base",
    "Features": "dymad.models.model_base",
    "Predictor": "dymad.models.model_base",
    "DEC_MAP": "dymad.models.components",
    "DYN_MAP": "dymad.models.components",
    "ENC_MAP": "dymad.models.components",
    "FZU_MAP": "dymad.models.components",
    "LIN_MAP": "dymad.models.components",
    "CD_KM": "dymad.models.recipes",
    "CD_KMM": "dymad.models.recipes",
    "CD_KMSK": "dymad.models.recipes",
    "CD_LDM": "dymad.models.recipes",
    "CD_LFM": "dymad.models.recipes",
    "CD_SDM": "dymad.models.recipes",
    "DGKBF": "dymad.models.collections",
    "DGKM": "dymad.models.collections",
    "DGKMSK": "dymad.models.collections",
    "DGLDM": "dymad.models.collections",
    "DGLTI": "dymad.models.collections",
    "DKBF": "dymad.models.collections",
    "DKM": "dymad.models.collections",
    "DKMSK": "dymad.models.collections",
    "DLDM": "dymad.models.collections",
    "DLDMG": "dymad.models.collections",
    "DLTI": "dymad.models.collections",
    "DSDM": "dymad.models.collections",
    "DSDMG": "dymad.models.collections",
    "GKBF": "dymad.models.collections",
    "GKM": "dymad.models.collections",
    "GLDM": "dymad.models.collections",
    "GLTI": "dymad.models.collections",
    "KBF": "dymad.models.collections",
    "KM": "dymad.models.collections",
    "KMM": "dymad.models.collections",
    "LDM": "dymad.models.collections",
    "LDMG": "dymad.models.collections",
    "LTI": "dymad.models.collections",
    "PredefinedModel": "dymad.models.collections",
    "TemplateCorrAlg": "dymad.models.recipes_corr",
    "TemplateCorrDif": "dymad.models.recipes_corr",
    "predict_continuous": "dymad.models.prediction",
    "predict_continuous_exp": "dymad.models.prediction",
    "predict_continuous_fenc": "dymad.models.prediction",
    "predict_continuous_np": "dymad.models.prediction",
    "predict_discrete": "dymad.models.prediction",
    "predict_discrete_exp": "dymad.models.prediction",
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
