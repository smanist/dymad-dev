from typing import Dict, Any, Type
import copy
import logging
import torch

from dymad.training.driver import SingleSplitDriver

logger = logging.getLogger(__name__)

class NODETrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by NODE.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
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

        # By default, we don't specify the phases
        # So we manually create a single NODE phase here
        cfg = copy.deepcopy(self.base_config["training"])
        del self.base_config["training"]
        cfg.update({"name": "NODE", "trainer": "NODE"})
        self.base_config.update({"phases": [cfg]})


class WeakFormTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by Weak Form.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
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

        # By default, we don't specify the phases
        # So we manually create a single NODE phase here
        cfg = copy.deepcopy(self.base_config["training"])
        del self.base_config["training"]
        cfg.update({"name": "WeakForm", "trainer": "Weak"})
        self.base_config.update({"phases": [cfg]})


class LinearTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by Linear regression.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
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

        # By default, we don't specify the phases
        # So we manually create a single NODE phase here
        cfg = copy.deepcopy(self.base_config["training"])
        del self.base_config["training"]
        cfg.update({"name": "Linear", "trainer": "Linear"})
        self.base_config.update({"phases": [cfg]})


class StackedTrainer(SingleSplitDriver):
    """
    Simple interface for single-split phased training.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
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
