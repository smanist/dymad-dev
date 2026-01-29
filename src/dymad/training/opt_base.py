import copy
import logging
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Type, Union

from dymad.io import DynData
from dymad.losses import LOSS_MAP
from dymad.training.helper import RunState
from dymad.training.ls_update import LSUpdater
from dymad.utils import make_scheduler, plot_hist, plot_trajectory

logger = logging.getLogger(__name__)

class OptBase:
    """
    Base Optimization class: owns data, model, optimizer, schedulers, training
    loop, checkpointing, history, LSUpdater, and RunState.

    Note:
        This class is not meant to be used directly; one should use the Driver classes.

    Key features of this class

      - Initialization in three modes: Data-only RunState (fresh start),
        Full RunState (continue from other opt), or from-checkpoint (restart from same opt).
      - Multiple training criteria are supported, and linearly combined via weights.
        There is also a separate prediction criterion for evaluation.
        All the training criteria can be different between different phases of optimization,
        but the prediction criterion is shared.
      - All the history of criteria are stored and plotted during training for monitoring;
        similarly one random validation trajectory is predicted and plotted regularly.

    Inderited Opt classes (NODE, WF, LR, ...) should

      - implement `_process_batch(batch)` (return list of losses),
      - optionally customize model / optimizer / schedulers via config.

    Args:
        config (Dict): Overall experiment configuration
        config_phase (Dict): Phase-specific configuration
        model_class (Type[torch.nn.Module]): Class of the model to train
        run_state (RunState): Existing RunState to attach (data-only or full)
        device (torch.device): Device to use
        dtype (torch.dtype): Data type to use
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Dict[str, Any],
        config_phase: Dict[str, Any],
        model_class: Type[torch.nn.Module],
        run_state: RunState,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = copy.deepcopy(config)
        self.config_phase = copy.deepcopy(config_phase)
        self.model_class = model_class
        self.device = device
        self.dtype = dtype

        if 'save_interval' not in self.config_phase:
            self.config_phase['save_interval'] = 10

        self.convergence_tolerance_reached = False

        # ODE solver settings, potentially needed for losses
        self.ode_method = self.config_phase.get("ode_method", "dopri5")
        self.ode_args = self.config_phase.get("ode_args", {})

        # Setup paths
        self.model_name = self.config["model"]["name"]
        self.checkpoint_path = self.config["path"]["checkpoint_prefix"] + f"/{self.model_name}_checkpoint.pt"
        self.best_model_path = self.config["path"]["checkpoint_prefix"] + f"/{self.model_name}.pt"
        os.makedirs(self.config["path"]["results_prefix"], exist_ok=True)
        self.results_prefix  = self.config["path"]["results_prefix"]

        # Create model, optimizer, schedulers, criteria
        self.attach_run_state(run_state)
        self.load_checkpoint()

        # logging summary
        logger.info("Opt Initialized:")
        logger.info(f"Model name: {self.model_name}")
        logger.info(self.model.diagnostic_info())
        logger.info("Optimization settings:")
        logger.info(self.optimizer)
        logger.info("Criteria settings:")
        for _n, _c, _w in zip(self.criteria_names[:-1], self.criteria[:-1], self.criteria_weights):
            logger.info(f" - {_n}: weight={_w}, criterion={_c}")
        logger.info("Prediction criterion settings:")
        logger.info(f" - {self.criteria_names[-1]}: criterion={self.criteria[-1]}")
        logger.info("Scheduler info:")
        for _s in self.schedulers:
            logger.info(_s.diagnostic_info())
        logger.info(f"Using device: {self.device}")
        logger.info(f"Double precision: {self.config['data'].get('double_precision', False)}")
        logger.info(
            f"Epochs: {self.config_phase['n_epochs']}, "
            f"Save interval: {self.config_phase['save_interval']}"
        )
        logger.info(f"ODE method: {self.ode_method}, Options: {self.ode_args}")

    def attach_run_state(self, state: RunState) -> None:
        """
        Attach an existing RunState.

        Cases:
          - data-only: model/optimizer/schedulers None -> we set them up fresh.
          - full: everything present -> we reuse all.
        """
        # Data
        self.train_set = state.train_set
        self.valid_set = state.valid_set
        self.train_loader = state.train_loader
        self.valid_loader = state.valid_loader
        self.train_md = state.train_md     # Only need dimension info
        self.valid_md = state.valid_md

        self._setup_model()
        if state.model is None or state.optimizer is None or not state.schedulers:
            # If model is None, this is "data-only" state
            # Start history from scratch
            self.start_epoch = 0
            self.best_loss = {"valid_total" : float("inf")}
            self.hist = []
            self.crit = []
            self.epoch_times = []
        else:
            # Full RunState: reuse model/optimizer, but not schedulers and criteria
            self.model.load_state_dict(state.model.state_dict())
            self.optimizer.load_state_dict(state.optimizer.state_dict())

            self.start_epoch = state.epoch
            self.best_loss = {"valid_total" : float("inf")}     # reset best loss as the losses may differ
            self.hist = copy.deepcopy(state.hist)
            self.crit = list(state.crit)
            self.epoch_times = []
        self._setup_ls()

    def load_checkpoint(self) -> None:
        """If checkpoint exists and config requires, load it into the current model/optimizer/schedulers."""
        checkpoint_path = None
        load_from_checkpoint = self.config_phase.get('load_checkpoint', False)
        if isinstance(load_from_checkpoint, str):
            checkpoint_path = load_from_checkpoint
        elif load_from_checkpoint:
            checkpoint_path = self.checkpoint_path

        flag = True
        if checkpoint_path is None:
            logger.info(f"Got load_from_checkpoint={load_from_checkpoint}, resulting in checkpoint_path=None. Starting from scratch.")
            flag = False
        elif not os.path.exists(checkpoint_path):
            logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            flag = False
        if not flag:
            return flag

        ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
        state = RunState.from_checkpoint(
            ckpt, self.model, self.optimizer, self.schedulers, self.criteria)

        # Attach the persistent parts
        self.start_epoch = state.epoch + 1  # resume from next epoch
        self.best_loss = copy.deepcopy(state.best_loss)
        self.hist = copy.deepcopy(state.hist)
        self.crit = state.crit
        self.convergence_tolerance_reached = False  # reset on load
        self.epoch_times = state.epoch_times

        logger.info(f"Loaded checkpoint from {self.checkpoint_path}, starting at epoch {self.start_epoch}")
        return flag

    # ------------------------------------------------------------------
    # Initialization - internal API's
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        """Setup model, optimizer, schedulers, and criteria."""
        # Model
        self.model = self.model_class(
            self.config["model"], self.train_md, dtype=self.dtype, device=self.device
        ).to(self.device)
        if self.config["data"].get("double_precision", False):
            self.model = self.model.double()

        # Optimizer
        lr = float(self.config_phase.get("learning_rate", 1e-3))
        gamma = float(self.config_phase.get("decay_rate", 0.999))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # By default there is only one scheduler.
        # There might be more, e.g., in OptNODE with sweep scheduler.
        self.schedulers = [make_scheduler(
            torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        )]

        # Criteria - for training
        crit_dict  = copy.deepcopy(self.config.get("criterion", {}))
        crit_dict.update(self.config_phase.get("criterion", {}))   # Possible phase override

        self.criteria = []
        self.criteria_names = ['dynamics']   # There is always dynamics
        self.criteria_weights = []
        if 'dynamics' in crit_dict:
            crit_cfg = crit_dict['dynamics']
            loss_class = LOSS_MAP[crit_cfg.get("type", "mse")]
            self.criteria.append(loss_class(**crit_cfg.get("params", {})))
            self.criteria_weights.append(crit_cfg.get("weight", 1.0))
        else:
            # Default dynamics criterion
            self.criteria.append(torch.nn.MSELoss(reduction="mean"))
            self.criteria_weights.append(1.0)
        if 'recon' in crit_dict:
            crit_cfg = crit_dict['recon']
            loss_class = LOSS_MAP.get(crit_cfg.get("type", "mse"))
            self.criteria.append(loss_class(**crit_cfg.get("params", {})))
            self.criteria_weights.append(crit_cfg.get("weight", 1.0))
            self.criteria_names.append('recon')
        for key in crit_dict:
            if key in ['dynamics', 'recon']:
                continue
            crit_cfg = crit_dict[key]
            loss_class = LOSS_MAP.get(crit_cfg.get("type", "mse"))
            self.criteria.append(loss_class(**crit_cfg.get("params", {})))
            self.criteria_weights.append(crit_cfg.get("weight", 1.0))
            self.criteria_names.append(key)

        # Criteria - for monitoring, possibly different from training
        crit_dict = copy.deepcopy(self.config.get("prediction_criterion", {}))
        crit_dict.update(self.config_phase.get("prediction_criterion", {}))   # Possible phase override
        if len(crit_dict) > 0:
            key = crit_dict.get("type", "mse")
            loss_class = LOSS_MAP.get(key)
            self.criteria.append(loss_class(**crit_dict.get("params", {})))
            self.criteria_names.append(key)
        else:
            self.criteria.append(self.criteria[0])
            self.criteria_names.append(str(self.criteria_names[0]))

    def _setup_ls(self) -> None:
        """Setup LSUpdater if requested."""
        if self.config_phase.get("ls_update", False):
            cfg = self.config_phase["ls_update"]
            self._ls = LSUpdater(
                method=cfg.get("method", "full"),
                model=self.model,
                dt=self.train_md["dt_and_n_steps"][0][0],
                params=cfg.get("params", None),
                **cfg.get("kwargs", {}),
            )
            self._ls_update_interval = cfg.get("update_interval", 1)
            self._ls_update_times    = cfg.get("update_times", 1)
            self._ls_reset           = cfg.get("reset_optimizer", True)
            self._start_w_ls         = cfg.get("start_with_ls", True)
            self._param_to_name      = {
                param: name for name, param in self.model.named_parameters()
            }
        else:
            self._ls = None
            self._ls_update_interval = 0
            self._ls_update_times    = 0
            self._ls_reset           = False
            self._param_to_name      = {}
            self._start_w_ls         = False

    # ------------------------------------------------------------------
    # Checkpoint I/O (via RunState)
    # ------------------------------------------------------------------

    def export_run_state(self, epoch: int) -> RunState:
        """Package the current trainer state into a RunState."""
        return RunState(
            config=self.config,
            device=self.device,
            epoch=epoch,
            best_loss=copy.deepcopy(self.best_loss),
            hist=copy.deepcopy(self.hist),
            crit=list(self.crit),
            epoch_times=list(self.epoch_times),
            converged=self.convergence_tolerance_reached,
            model=self.model,
            optimizer=self.optimizer,
            schedulers=self.schedulers,
            criteria=self.criteria,
            criteria_weights=self.criteria_weights,
            criteria_names=self.criteria_names,
            train_set=self.train_set,
            valid_set=self.valid_set,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            train_md=self.train_md,
            valid_md=self.valid_md,
        )

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> None:
        """Save checkpoint using current RunState."""
        state = self.export_run_state(epoch)
        ckpt = state.to_checkpoint()
        torch.save(ckpt, self.checkpoint_path if path is None else path)

    def save_if_best(self, local_hist: Dict) -> None:
        """Save best model if validation loss improves."""
        if local_hist["valid_total"][-1] < self.best_loss["valid_total"]:
            epoch = local_hist['epoch'][-1]
            self.best_loss = {_k : _v[-1] for _k, _v in local_hist.items()}
            self.convergence_epoch = epoch+1
            self.save_checkpoint(epoch, self.best_model_path)
            logger.info(f"New best model at epoch {epoch}, valid_loss={self.best_loss['valid_total']:.4e}")
            return True
        return False

    # ------------------------------------------------------------------
    # Per-step training API's, usually overridden by child classes
    # ------------------------------------------------------------------

    def _process_batch(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss terms for a batch.

        Return a list of losses corresponding to `self.criteria_names[:-1]`.
        """
        raise NotImplementedError

    def train_epoch(self) -> float:
        """
        Generic training loop for one epoch.

        Uses `_process_batch` + multi-loss aggregation.
        Handles

          - optimizer step,
          - scheduler step(eploss),
          - convergence flag,
          - minimum learning rate.
        """
        self.model.train()

        loss_total = 0.0
        loss_items = [0.0 for _ in self.criteria_weights]
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss_list = self._process_batch(batch)
            loss = self._aggregate_losses(loss_list)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            loss_items = [_l + b.item() for _l, b in zip(loss_items, loss_list)]

        avg_loss_epoch = loss_total / len(self.train_loader)
        avg_loss_items = [l / len(self.train_loader) for l in loss_items]

        # Step schedulers
        for scheduler in self.schedulers:
            flag, changed = scheduler.step(eploss=avg_loss_epoch)
            self.convergence_tolerance_reached = (
                self.convergence_tolerance_reached or flag
            )
            if changed:
                self.best_loss = {"valid_total" : float("inf")}
                logger.info("Resetting best loss due to scheduler change.")

        # Enforce minimum LR
        min_lr = float(self.config_phase.get("min_learning_rate", 1e-6))
        if min_lr > 0.0:
            for param_group in self.optimizer.param_groups:
                if param_group["lr"] < min_lr:
                    param_group["lr"] = min_lr

        return avg_loss_epoch, avg_loss_items

    # ------------------------------------------------------------------
    # Per-step training API's, other helpers
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader) -> float:
        """Generic evaluation loop using `_process_batch`."""
        self.model.eval()
        loss_total = 0.0
        loss_items = [0.0 for _ in self.criteria_weights]
        with torch.no_grad():
            for batch in dataloader:
                loss_list = self._process_batch(batch)
                loss = self._aggregate_losses(loss_list)
                loss_total += loss.item()
                loss_items = [_l + b.item() for _l, b in zip(loss_items, loss_list)]
        return loss_total / len(dataloader), [l / len(dataloader) for l in loss_items]

    def _aggregate_losses(self, loss_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine named loss terms using config-defined weights.

        - If 'total' in loss_dict: we directly return it.
        - Else: sum w_i * loss_i, where w_i = loss_weights.get(name, 1.0).
        """
        total = 0.0
        for _l, _w in zip(loss_list, self.criteria_weights):
            total += _l * _w
        return total

    def _additional_criteria_evaluation(self, x_hat, predictions, B) -> None:
        """
        Compute additional criteria losses beyond dynamics
        """
        loss_list = []
        if len(self.criteria_weights) < 2:
            return loss_list

        if self.criteria_names[1] == "recon":
            if x_hat is None:
                _z = self.model.encoder(B)
                x_hat = self.model.decoder(_z, B)
            recon_loss = self.criteria[1](B.x, x_hat.view(*B.x.shape))
            loss_list.append(recon_loss)
            n_eval = 2
        else:
            n_eval = 1

        preds = predictions
        if predictions is None:
            if len(self.criteria)-1 > n_eval:
                # This means there are additional criteria,
                # which we assume requires predictions, and need to compute this
                init_states = B.x[:, 0, :]  # (batch_size, n_total_state_features)
                # Use the actual time points from trajectory manager
                ts = B.t
                if ts.dim() == 3 and ts.size(0) == 1:
                    # Expect this to be the graph case with broadcasted time
                    # For now we only take the first batch entry, assuming all are identical
                    ts = ts[..., 0]
                ts = ts.to(self.device)
                preds = self.model.predict(
                    init_states,
                    B,
                    ts,
                    method=self.ode_method,
                    **self.ode_args,
                )

        for _i in range(n_eval, len(self.criteria)-1):
            loss_value = self.criteria[_i](preds, B.x)
            loss_list.append(loss_value)

        return loss_list

    # ------------------------------------------------------------------
    # Prediction criterion evaluation
    # ------------------------------------------------------------------

    def evaluate_prediction_criterion_single(self,
                    truth: DynData,
                    method: str = 'dopri5',
                    plot: bool = False) -> float:
        """
        Calculate prediction criterion between model predictions and ground truth.

        Regardless of the training criterion, this function evaluates
        a prediction-based criterion (e.g., RMSE over trajectory).

        Args:
            truth (DynData): Ground truth trajectory data
            method (str): ODE solver method (for models that use ODE solvers)
            plot (bool): Whether to plot the predicted vs ground truth trajectories

        Returns:
            float: Prediction criterion between predictions and ground truth, can be problem dependent
        """
        with torch.no_grad():
            # Extract states and controls
            x_truth = truth.x
            x0 = truth.x[:, 0, :]
            ts = truth.t

            # Make prediction
            x_pred = self.model.predict(x0, truth, ts, method=method)

            # Prediction criterion
            prediction_crit = self.criteria[-1](x_pred, x_truth)
            _crit = f" {self.criteria_names[-1]} {prediction_crit.item():.4e}"

            if plot:
                x_truth = x_truth.detach().cpu().numpy().squeeze(0)
                x_pred = x_pred.detach().cpu().numpy().squeeze(0)
                ts = ts.detach().cpu().numpy().squeeze(0)
                _us = None if truth.u is None else truth.u.detach().cpu().numpy().squeeze(0)
                plotting_config = self.config.get('plotting', {})
                plot_trajectory(np.array([x_truth, x_pred]), ts, self.model_name,
                                us=_us, labels=['Truth', 'Prediction'+_crit], prefix=self.results_prefix,
                                **plotting_config)

            return prediction_crit.item()

    def evaluate_prediction_criterion(self, split: str = 'test', plot: bool = False, evaluate_all: bool = False) -> float:
        """
        Calculate prediction criterion on trajectory(ies) from the specified split.

        Args:
            split (str): Dataset split to use ('train', 'valid')
            plot (bool): Whether to plot the results (only works when evaluate_all=False)
            evaluate_all (bool):

                - If True, evaluate all trajectories and return the mean.
                - If False, evaluate a single random trajectory.

        Returns:
            float: Criterion value
        """
        if split == "train":
            dataset = self.train_set
        elif split == "valid":
            dataset = self.valid_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if evaluate_all:
            plot = False
            # dataset = dataset
        else:
            dataset = [random.choice(dataset)]

        _method = self.config.get("training", {}).get("ode_method", "dopri5")
        criterion_values = [
            self.evaluate_prediction_criterion_single(trajectory, method=_method, plot=plot)
            for trajectory in dataset
        ]

        return sum(criterion_values) / len(criterion_values)

    def update_prediction_criterion_history(self, epoch: int) -> None:
        """
        Handy wrapper for evaluating prediction criterion on train and valid sets, log and store results.
        """
        train_crit = self.evaluate_prediction_criterion('train', plot=False)
        valid_crit = self.evaluate_prediction_criterion('valid', plot=True)
        self.crit.append([epoch, train_crit, valid_crit])

        logger.info(
            f"Prediction criterion - "
            f"Train: {train_crit:.4e}, "
            f"Valid: {valid_crit:.4e}"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> int:
        """Run full training loop."""

        n_epochs = self.config_phase['n_epochs']
        save_interval = self.config_phase['save_interval']

        self.convergence_epoch = None
        self.epoch_times = []

        _ls_count = 0

        local_hist = {"epoch": []}
        local_hist.update({"train_"+_k : [] for _k in self.criteria_names[:-1]})
        local_hist.update({"valid_"+_k : [] for _k in self.criteria_names[:-1]})
        local_hist['train_total'] = []
        local_hist['valid_total'] = []

        overall_start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            # Training and evaluation
            # Only timing the train and validation phases
            # since test loss is only for reference
            epoch_start_time = time.time()

            # Partial LS update if specified
            if self._ls is not None:
                if epoch % self._ls_update_interval == 0:
                    if epoch == self.start_epoch and not self._start_w_ls:
                        logger.info("Skipping LS update at the start epoch as per configuration.")
                    else:
                        _ls_count += 1
                        if _ls_count == self._ls_update_times+1:
                            logger.info(f"LS update performed {self._ls_update_times} times, stopping further LS updates.")
                        elif _ls_count <= self._ls_update_times:
                            logger.info(f"Performing LS update {_ls_count} of {self._ls_update_times} times at epoch {epoch+1}")
                            _, params = self._ls.update(self.model, self.train_loader)

                            # Remove optimizer state for parameters updated by LS
                            if self._ls_reset:
                                target_names = [self._param_to_name.get(p, "<unnamed>") for p in params]
                                for _p, _n in zip(params, target_names):
                                    for param_group in self.optimizer.param_groups:
                                        param_names = [self._param_to_name.get(_q, "<unnamed>") for _q in param_group['params']]
                                        if _n in param_names:
                                            self.optimizer.state.pop(_p, None)
                                            logger.info(f"Removed optimizer state for {_n} after LS update.")

            loss_train_total, loss_train_items = self.train_epoch()
            loss_valid_total, loss_valid_items = self.evaluate(self.valid_loader)

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            # Record history
            local_hist['epoch'].append(epoch)
            for _k, _t, _v in zip(self.criteria_names[:-1], loss_train_items, loss_valid_items):
                local_hist['train_'+_k].append(_t)
                local_hist['valid_'+_k].append(_v)
            local_hist['train_total'].append(loss_train_total)
            local_hist['valid_total'].append(loss_valid_total)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.start_epoch + n_epochs}, "
                f"Train Loss: {loss_train_total:.4e}, " + ", ".join([
                    f"{name}: {value:.4e}" for name, value in zip(self.criteria_names[:-1], loss_train_items)
                ]) + "; " +
                f"Valid Loss: {loss_valid_total:.4e}, " + ", ".join([
                    f"{name}: {value:.4e}" for name, value in zip(self.criteria_names[:-1], loss_valid_items)
                ])
            )

            # Save best model
            self.save_if_best(local_hist)

            # Periodic checkpoint and evaluation
            if (epoch + 1) % save_interval == 0 or self.convergence_tolerance_reached:
                self.save_checkpoint(epoch)

                # Plot loss curves
                plot_hist(
                    self.hist + [local_hist], self.crit, self.criteria_names[-1],
                    self.model_name, prefix=self.results_prefix)

                # Evaluate prediction criterion on random trajectories
                self.update_prediction_criterion_history(epoch)

                if self.convergence_tolerance_reached:
                    self.convergence_epoch = epoch+1
                    logger.info(f"Convergence reached at epoch {epoch+1} "
                                f"with validation loss {loss_valid_total:.4e}")
                    break
        self.hist += [local_hist]

        if self.crit == []:
            self.update_prediction_criterion_history(epoch)

        total_training_time = time.time() - overall_start_time
        avg_epoch_time = np.mean(self.epoch_times)

        plot_hist(
            self.hist, self.crit, self.criteria_names[-1],
            self.model_name, prefix=self.results_prefix)
        _ = self.evaluate_prediction_criterion('valid', plot=True)

        # Save summary of training
        # Here we also save the model itself - a lazy approach but more "out-of-the-box" for deployment
        tmp = np.array(self.crit).T
        crit_epoch, crits = tmp[0], tmp[1:]
        results = {
            'model_name': self.model_name,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'final_train_loss': local_hist['train_total'][-1],
            'final_valid_loss': local_hist['valid_total'][-1],
            'best_valid_loss': copy.deepcopy(self.best_loss),
            'convergence_epoch': self.convergence_epoch,
            'hist': self.hist,
            'crit_name': self.criteria_names[-1],  # Only the prediction criterion
            'crit_epoch': crit_epoch,
            'crits': crits,
        }

        file_name = f"{self.results_prefix}/{self.model_name}_summary.npz"
        np.savez_compressed(file_name, **results)
        logger.info("Training complete. Summary of training:")
        for key in [
            "model_name",
            "total_training_time",
            "avg_epoch_time",
            "final_train_loss",
            "final_valid_loss",
            "best_valid_loss",
            "convergence_epoch",
        ]:
            info = f"{results[key]:.4e}" if isinstance(results[key], float) else str(results[key])
            logger.info(f"{key}: {info}")
        logger.info(f"Summary and loss/criterion histories saved to {file_name}")

        return epoch