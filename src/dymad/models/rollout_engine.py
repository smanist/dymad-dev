"""Typed rollout-engine selection for migrated model-spec families."""

from dataclasses import dataclass
from typing import Callable, Optional

from dymad.models.model_spec import ModelSpec
from dymad.models.prediction import (
    predict_continuous,
    predict_continuous_exp,
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
    "discrete": predict_discrete,
    "discrete_exp": predict_discrete_exp,
}


def select_rollout_engine(model_spec: ModelSpec) -> Optional[RolloutEngineSelection]:
    """Return a typed rollout engine for migrated families, if available."""
    rollout = model_spec.rollout
    if rollout is None:
        return None
    if rollout.family != "lti":
        return None
    predictor = _PREDICTOR_BY_NAME.get(rollout.predictor)
    if predictor is None:
        raise ValueError(
            f"Unsupported typed rollout predictor '{rollout.predictor}' "
            f"for family '{rollout.family}'."
        )
    if rollout.predictor in {"continuous", "discrete"} and not rollout.supports_control_inputs:
        raise ValueError(
            "Typed LTI rollout predictor requires control-input support for "
            f"'{rollout.predictor}'."
        )
    return RolloutEngineSelection(
        predictor=predictor,
        source=f"{rollout.family}:{rollout.predictor}",
    )
