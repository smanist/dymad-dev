import logging
import torch
from typing import Any, Dict, Type, Union

from dymad.io import DynData
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
    ):
        super().__init__(config, config_phase, model_class, run_state, device, dtype)

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

    def _process_batch(self, batch: DynData) -> Dict[str, torch.Tensor]:
        """
        Compute NODE loss terms on a batch and return a dict of named losses.

        The TrainerBase will aggregate:
          - If 'total' in dict -> use directly.
          - Else, sum loss_weights[name] * term.
        Here we expose atomic losses "dynamics" and "recon", and let the
        base class aggregate according to config["training"]["loss_weights"].
        """
        num_steps = self.schedulers[1].get_length()
        if num_steps is None:
            num_steps = batch.x.size(1)

        # Chop trajectories
        if self.chop_mode == "initial":
            B = batch.truncate(num_steps)
        else:
            B = batch.unfold(num_steps, _determine_chop_step(num_steps, self.chop_step))

        B = B.to(self.device)

        # Initial states and time vector
        init_states = B.x[:, 0, :]  # (batch_size, n_total_state_features)
        # Use the actual time points from trajectory manager
        ts = B.t[:, :num_steps]
        if ts.dim() == 3 and ts.size(0) == 1:
            # Expect this to be the graph case with broadcasted time
            # For now we only take the first batch entry, assuming all are identical
            ts = ts[..., 0]
        ts = ts.to(self.device)

        # Batched NODE prediction
        predictions = self.model.predict(
            init_states,
            B,
            ts,
            method=self.ode_method,
            **self.ode_args,
        )

        # Base dynamics criterion
        dynamics_loss = self.criteria[0](predictions, B.x)
        loss_list = [dynamics_loss]

        # Other criteria
        # x_hat is computed inside criteria evaluation if needed
        x_hat = None
        _list = self._additional_criteria_evaluation(x_hat, predictions, B)
        loss_list.extend(_list)

        return loss_list
