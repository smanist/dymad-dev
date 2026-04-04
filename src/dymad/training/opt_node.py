import logging
import torch
from typing import Any, Dict, Type, Union

from dymad.training.batch_adapter import TrainerBatch, batch_to_runtime
from dymad.training.execution_services import ExecutionServices
from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase
from dymad.utils import make_scheduler

logger = logging.getLogger(__name__)

def _determine_chop_step(window: int, step: Union[int, float]) -> int:
    """
    Determine the chop step based on the window size and step value.
    """
    if isinstance(step, int):
        return step
    elif isinstance(step, float):
        stp = int(window * step)
        return min(max(stp, 1), window)
    else:
        raise ValueError(f"Invalid step type: {type(step)}. Expected int or float.")


class OptNODE(OptBase):
    """
    Optimization using Neural ODE approach.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_phase: Dict[str, Any],
        model_class: Type[torch.nn.Module],
        run_state: RunState,
        device: torch.device,
        dtype: torch.dtype,
        execution_services: ExecutionServices | None = None,
    ):
        super().__init__(
            config,
            config_phase,
            model_class,
            run_state,
            device,
            dtype,
            execution_services=execution_services,
        )

        # Trajectory chopping
        self.chop_mode = self.config_phase.get("chop_mode", "initial")
        assert self.chop_mode in ["initial", "unfold"], f"Invalid chop_mode: {self.chop_mode}"
        self.chop_step = self.config_phase.get("chop_step", 1.0)
        assert self.chop_step > 0, f"Chop step must be positive. Got: {self.chop_step}"

        # Optional: minimum LR default for NODE
        self.config_phase.setdefault("min_learning_rate", 1e-6)

        # Sweep settings
        sweep_lengths = self.config_phase.get("sweep_lengths", [None])
        epoch_step = self.config_phase.get(
            "sweep_epoch_step", self.config_phase["n_epochs"]
        )
        sweep_tols = self.config_phase.get("sweep_tols", None)
        sweep_mode = self.config_phase.get("sweep_mode", "skip")

        sweep_scheduler = make_scheduler(
            scheduler_type="sweep",
            sweep_lengths=sweep_lengths,
            sweep_tols=sweep_tols,
            epoch_step=epoch_step,
            mode=sweep_mode,
        )
        self.schedulers.append(sweep_scheduler)

        if self.chop_mode == 'initial':
            logger.info(f"Chop mode: {self.chop_mode}, initial steps only")
        else:
            logger.info(f"Chop mode: {self.chop_mode}, window stride: {self.chop_step}")
        logger.info(f"Added sweep scheduler: {self.schedulers[-1].diagnostic_info()}")

    def _process_batch(self, batch: TrainerBatch) -> Dict[str, torch.Tensor]:
        """
        Compute NODE loss terms on a batch and return a dict of named losses.

        The TrainerBase will aggregate:
          - If 'total' in dict -> use directly.
          - Else, sum loss_weights[name] * term.
        Here we expose atomic losses "dynamics" and "recon", and let the
        base class aggregate according to config["training"]["loss_weights"].
        """
        if hasattr(batch, "is_ragged") and batch.is_ragged:
            return self._average_loss_lists(
                [self._process_batch(sample) for sample in batch.iter_single_batches()]
            )

        num_steps = self.schedulers[1].get_length()
        if num_steps is None:
            runtime = batch_to_runtime(batch)
            num_steps = runtime.x.size(1)

        # Chop trajectories through the typed-batch path when possible.
        if self.chop_mode == "initial":
            if hasattr(batch, "truncate"):
                B = batch.truncate(num_steps).to(self.device)
                runtime = batch_to_runtime(B)
            else:
                runtime = batch_to_runtime(batch)
                runtime = runtime.truncate(num_steps).to(self.device)
        else:
            chop_step = _determine_chop_step(num_steps, self.chop_step)
            if hasattr(batch, "window"):
                B = batch.window(num_steps, chop_step).to(self.device)
                runtime = batch_to_runtime(B)
            else:
                runtime = batch_to_runtime(batch)
                runtime = runtime.unfold(num_steps, chop_step).to(self.device)

        # Initial states and time vector
        init_states = runtime.x[:, 0, :]  # (batch_size, n_total_state_features)
        # Use the actual time points from trajectory manager
        ts = runtime.t[:, :num_steps]
        ts = ts.to(self.device)

        # Batched NODE prediction
        predictions = self.model.predict(
            init_states,
            runtime,
            ts,
            method=self.ode_method,
            **self.ode_args,
        )

        # Base dynamics criterion
        dynamics_loss = self.criteria[0](predictions, runtime.x)
        loss_list = [dynamics_loss]

        # Other criteria
        # x_hat is computed inside criteria evaluation if needed
        x_hat = None
        _list = self._additional_criteria_evaluation(x_hat, predictions, runtime)
        loss_list.extend(_list)

        return loss_list
