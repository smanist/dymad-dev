import copy
import logging
from typing import Any

import torch

from dymad.training.driver import SingleSplitDriver

logger = logging.getLogger(__name__)


def _select_single_stage_phase(
    base_config: dict[str, Any], *, name: str, trainer: str
) -> dict[str, Any]:
    phases = base_config.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Config must contain at least one phase after load_config normalization.")
    cfg = copy.deepcopy(phases[0])
    cfg.update({"name": name, "trainer": trainer})
    cfg.setdefault("type", "optimizer")
    return cfg


class NODETrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by NODE.
    """

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )

        self.base_config["phases"] = [
            _select_single_stage_phase(self.base_config, name="NODE", trainer="NODE")
        ]


class WeakFormTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by Weak Form.
    """

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )

        self.base_config["phases"] = [
            _select_single_stage_phase(self.base_config, name="WeakForm", trainer="Weak")
        ]


class LinearTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by Linear regression.
    """

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )

        self.base_config["phases"] = [
            _select_single_stage_phase(self.base_config, name="Linear", trainer="Linear")
        ]


class OneStepTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by nonlinear one-step optimization.
    """

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )

        self.base_config["phases"] = [
            _select_single_stage_phase(self.base_config, name="OneStep", trainer="OneStep")
        ]


class StackedTrainer(SingleSplitDriver):
    """
    Simple interface for single-split phased training.
    """

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )
