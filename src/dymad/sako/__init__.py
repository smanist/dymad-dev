"""Public SAKO barrel with lazy imports to avoid boundary-package cycles."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dymad.sako.adapter import SpectralAnalysisAdapter, SpectralEigensystem
    from dymad.sako.base import SAInterface, SpectralAnalysis, filter_spectrum
    from dymad.sako.plotting import SpectralPlottingAdapter, per_state_err
    from dymad.sako.rals import RALowRank, estimate_pseudospectrum, resolvent_analysis
    from dymad.sako.sako import SAKO
    from dymad.sako.snapshot import KoopmanWeightSnapshot, SpectralSnapshot, build_spectral_snapshot

__all__ = [
    "estimate_pseudospectrum",
    "filter_spectrum",
    "per_state_err",
    "RALowRank",
    "resolvent_analysis",
    "SpectralAnalysisAdapter",
    "SpectralPlottingAdapter",
    "SpectralEigensystem",
    "SAInterface",
    "build_spectral_snapshot",
    "KoopmanWeightSnapshot",
    "SAKO",
    "SpectralSnapshot",
    "SpectralAnalysis",
]

_EXPORTS = {
    "estimate_pseudospectrum": ("dymad.sako.rals", "estimate_pseudospectrum"),
    "filter_spectrum": ("dymad.sako.base", "filter_spectrum"),
    "per_state_err": ("dymad.sako.plotting", "per_state_err"),
    "RALowRank": ("dymad.sako.rals", "RALowRank"),
    "resolvent_analysis": ("dymad.sako.rals", "resolvent_analysis"),
    "SpectralAnalysisAdapter": ("dymad.sako.adapter", "SpectralAnalysisAdapter"),
    "SpectralPlottingAdapter": ("dymad.sako.plotting", "SpectralPlottingAdapter"),
    "SpectralEigensystem": ("dymad.sako.adapter", "SpectralEigensystem"),
    "SAInterface": ("dymad.sako.base", "SAInterface"),
    "build_spectral_snapshot": ("dymad.sako.snapshot", "build_spectral_snapshot"),
    "KoopmanWeightSnapshot": ("dymad.sako.snapshot", "KoopmanWeightSnapshot"),
    "SAKO": ("dymad.sako.sako", "SAKO"),
    "SpectralSnapshot": ("dymad.sako.snapshot", "SpectralSnapshot"),
    "SpectralAnalysis": ("dymad.sako.base", "SpectralAnalysis"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
