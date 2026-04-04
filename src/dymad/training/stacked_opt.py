import torch
from typing import Any, Dict, List, Type

from dymad.training.execution_services import ExecutionServices
from dymad.training.helper import RunState
from dymad.training.phase_pipeline import OPT_REGISTRY as _OPT_REGISTRY, PhasePipeline, PhaseResult

OPT_REGISTRY = _OPT_REGISTRY

class StackedOpt:
    """
    Stack multiple optimization phases (e.g., WF -> NODE -> LR)
    on a (potentially precomputed) RunState.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_class: Type,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.execution_services = ExecutionServices.from_config(config, default_device=device)
        self.config = self.execution_services.apply_to_config(config)
        self.model_class = model_class
        self.device = self.execution_services.device
        self.dtype = dtype
        self.pipeline = PhasePipeline(
            config=self.config,
            model_class=model_class,
            device=self.device,
            dtype=dtype,
            execution_services=self.execution_services,
        )
        self.config = self.pipeline.config
        self.phases = self.pipeline.phases

    def run(self, initial_state: RunState) -> List[PhaseResult]:
        """Compatibility wrapper around :class:`PhasePipeline`."""
        return self.pipeline.run(initial_state=initial_state)
