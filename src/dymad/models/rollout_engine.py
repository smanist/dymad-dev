"""Typed rollout-engine selection for migrated model-spec families."""

from dataclasses import dataclass
from typing import Callable

from dymad.models.model_spec import ModelSpec, ModelSpecValidationError, PredictorKey
from dymad.models.prediction import (
    predict_continuous,
    predict_continuous_exp,
    predict_continuous_fenc,
    predict_continuous_np,
    predict_discrete,
    predict_discrete_exp,
)


@dataclass(frozen=True)
class RolloutEngineSelection:
    """Resolved rollout engine for a typed model spec."""

    predictor: Callable
    source: str


_PREDICTOR_BY_NAME = {
    "continuous": predict_continuous,
    "continuous_np": predict_continuous_np,
    "continuous_exp": predict_continuous_exp,
    "continuous_fenc": predict_continuous_fenc,
    "discrete": predict_discrete,
    "discrete_exp": predict_discrete_exp,
}


def _requested_predictor_key(model_spec: ModelSpec, model_config: dict, dims: dict) -> PredictorKey:
    rollout = model_spec.rollout
    requested = model_config.get("predictor_type", "ode")
    if rollout.default_predictor == "continuous_fenc":
        return "continuous_fenc"

    if model_spec.continuous_time:
        if requested == "exp":
            predictor_key: PredictorKey = "continuous_exp"
        elif requested == "np":
            predictor_key = "continuous_np"
        else:
            predictor_key = "continuous"
    else:
        if requested in {"exp", "np"}:
            predictor_key = "discrete_exp"
        else:
            predictor_key = "discrete"

    if predictor_key == "continuous_exp" and dims.get("u", 0) > 0:
        raise ModelSpecValidationError("Exponential rollout does not support control inputs.")
    if predictor_key == "discrete_exp" and requested == "exp" and dims.get("u", 0) > 0 and model_spec.recipe.kind == "sdm":
        raise ModelSpecValidationError("SDM exponential rollout does not support control inputs.")
    return predictor_key


def select_rollout_engine(model_spec: ModelSpec, model_config: dict, dims: dict) -> RolloutEngineSelection:
    """Resolve the rollout engine allowed by the typed model spec."""
    predictor_key = _requested_predictor_key(model_spec, model_config, dims)
    rollout = model_spec.rollout
    if predictor_key not in rollout.allowed_predictors:
        raise ModelSpecValidationError(
            f"Predictor '{predictor_key}' is not allowed for rollout family '{rollout.family}'."
        )
    predictor = _PREDICTOR_BY_NAME.get(predictor_key)
    if predictor is None:
        raise ModelSpecValidationError(
            f"Unsupported typed rollout predictor '{predictor_key}' "
            f"for family '{rollout.family}'."
        )
    if dims.get("u", 0) > 0 and not rollout.supports_control_inputs:
        raise ModelSpecValidationError(
            f"Rollout family '{rollout.family}' does not support control inputs."
        )
    if predictor_key in {"continuous", "discrete"} and not rollout.supports_control_inputs:
        raise ValueError(
            "Typed rollout predictor requires control-input support for "
            f"'{predictor_key}'."
        )
    return RolloutEngineSelection(
        predictor=predictor,
        source=f"{rollout.family}:{predictor_key}",
    )
