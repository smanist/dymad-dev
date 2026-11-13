"""Shared defaults and guards for optimizer choices within existing training phases."""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any


def _finite_number(config: dict[str, Any], key: str, *, positive: bool = False) -> None:
    value = config[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{key} must be a finite {bound} number")


def _positive_integer(config: dict[str, Any], key: str) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")


def _validate_criterion(criterion: object) -> None:
    if not isinstance(criterion, dict):
        raise ValueError("bfgs/lbfgs criterion must be a mapping")
    for name, options in criterion.items():
        if not isinstance(options, dict):
            raise ValueError(f"criterion.{name} must be a mapping")
        weight = options.get("weight", 1.0)
        _finite_number({"weight": weight}, "weight")
        if name != "dynamics" and weight == 0:
            continue
        if name == "dynamics" and weight <= 0:
            raise ValueError("bfgs/lbfgs require a positive dynamics weight")
        params = options.get("params", {})
        if (
            options.get("type", "mse") != "mse"
            or not isinstance(params, dict)
            or params.get("reduction", "mean") != "mean"
            or set(params) - {"reduction"}
        ):
            raise ValueError("bfgs/lbfgs require mean MSE for every active training loss")


def normalize_optimizer_config(config: dict[str, Any], trainer: str) -> dict[str, Any]:
    """Copy a phase config, preserving Adam defaults and checking supported QN options.

    The compiler applies this to explicit optimizer overrides before profile merging, so
    inherited Adam learning-rate and curriculum settings cannot become QN defaults. The
    phase runtime uses the same helper for direct library configs.
    """
    normalized = copy.deepcopy(config)
    optimizer = normalized.get("optimizer", "adam")
    if optimizer not in ("adam", "bfgs", "lbfgs"):
        raise ValueError("optimizer must be one of adam, bfgs, lbfgs")
    if optimizer == "adam":
        return normalized
    if trainer not in ("NODE", "Weak"):
        raise ValueError("bfgs/lbfgs currently require trainer NODE or Weak")

    defaults: dict[str, Any] = {
        "learning_rate": 1.0,
        "decay_rate": 1.0,
        "min_learning_rate": 0.0,
        "max_eval": 25,
        "tolerance_grad": 1e-9,
        "tolerance_change": 1e-12,
        "curvature_floor": 1e-10,
    }
    if trainer == "NODE":
        defaults.update(sweep_lengths=[None], sweep_tols=None, chop_mode="initial")
    for key, value in defaults.items():
        normalized.setdefault(key, value)

    _finite_number(normalized, "learning_rate", positive=True)
    for key in ("decay_rate", "min_learning_rate", "tolerance_grad", "tolerance_change"):
        _finite_number(normalized, key)
    _finite_number(normalized, "curvature_floor")
    _positive_integer(normalized, "max_eval")
    if normalized["decay_rate"] != 1 or normalized["min_learning_rate"] != 0:
        raise ValueError("bfgs/lbfgs require decay_rate=1 and min_learning_rate=0")
    if trainer == "NODE":
        if normalized["sweep_lengths"] != [None] or normalized["sweep_tols"] is not None:
            raise ValueError(
                "bfgs/lbfgs require fixed full horizon: sweep_lengths=[None], sweep_tols=None"
            )
        if normalized["chop_mode"] != "initial":
            raise ValueError("bfgs/lbfgs require initial-time shooting: chop_mode='initial'")
    if optimizer == "lbfgs":
        normalized.setdefault("history_size", 100)
        _positive_integer(normalized, "history_size")
    elif "history_size" in normalized:
        raise ValueError("history_size is only supported by lbfgs")
    if "criterion" in normalized:
        _validate_criterion(normalized["criterion"])
    return normalized
