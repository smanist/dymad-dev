from dataclasses import dataclass, field
from itertools import product
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Iterable, List, Optional

@dataclass
class RunState:
    """
    States of a training run to be checkpointed and restored.

    The usages are threefold:

        - Data-only state (only data members): For initializing an optimization in StackedOpt
        - Augmented state (adding persistent and model): For continuing an optimization in StackedOpt
        - Full state (adding optimizer, schedulers, criteria): For resuming an interrupted optimization

    It contains interfaces to and from checkpoint dictionaries, assuming full state minus data.
    """
    # Persistent
    config: Optional[Dict[str, Any]]
    device: Optional[torch.device] = None
    epoch: int = 0
    best_loss: Dict[str, float] = field(default_factory=lambda: {"valid_total" : float("inf")})
    hist: List[Any] = field(default_factory=list)
    crit: List[Any] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    converged: bool = False

    # Model
    model: Optional[torch.nn.Module] = None

    # Optimization states
    optimizer: Optional[torch.optim.Optimizer] = None
    schedulers: List[Any] = field(default_factory=list)
    criteria: Optional[Dict[str, torch.nn.Module]] = None
    criteria_weights: Optional[Dict[str, float]] = None
    prediction_criterion: Optional[torch.nn.Module] = None
    prediction_criterion_name: Optional[str] = None
    latent: Optional[Dict[str, Any]] = None

    # Data: live objects only (not serialized)
    train_set: Optional[Dataset] = None
    valid_set: Optional[Dataset] = None
    train_loader: Optional[DataLoader] = None
    valid_loader: Optional[DataLoader] = None
    train_md: Optional[Dict[str, Any]] = None
    valid_md: Optional[Dict[str, Any]] = None

    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize the persistent subset of state."""
        return {
            "config": self.config,
            "loss_format": "dict_v1",
            "device": self.device,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "hist": self.hist,
            "crit": self.crit,
            "epoch_times": self.epoch_times,
            "converged": self.converged,
            "model_state_dict": None if self.model is None else self.model.state_dict(),
            "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler_state_dicts": [
                scheduler.state_dict() for scheduler in self.schedulers if hasattr(scheduler, "state_dict")
            ],
            "criteria_state_dicts": (
                {name: criterion.state_dict() for name, criterion in self.criteria.items()}
                if self.criteria is not None else {}
            ),
            "criteria_weights": self.criteria_weights,
            "prediction_criterion_state_dict": (
                None if self.prediction_criterion is None
                else self.prediction_criterion.state_dict()
            ),
            "prediction_criterion_name": self.prediction_criterion_name,
            "latent": self.latent,
            "train_md": self.train_md,
            "valid_md": self.valid_md,
        }

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: Dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        schedulers: List[Any],
        criteria: Optional[Dict[str, torch.nn.Module]],
        prediction_criterion: Optional[torch.nn.Module],
    ) -> "RunState":
        """
        Rebuild RunState from a checkpoint, meant for restarting an interrupted run.
        """
        if ckpt.get("loss_format") != "dict_v1":
            raise ValueError("old checkpoint format not supported after loss migration")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dicts" in ckpt:
            for s, s_state in zip(schedulers, ckpt["scheduler_state_dicts"]):
                if hasattr(s, "load_state_dict"):
                    s.load_state_dict(s_state)
        if criteria is not None and "criteria_state_dicts" in ckpt:
            c_states = ckpt["criteria_state_dicts"]
            if not isinstance(c_states, dict):
                raise ValueError("old checkpoint format not supported after loss migration")
            for name, criterion in criteria.items():
                if name in c_states:
                    criterion.load_state_dict(c_states[name])
        if prediction_criterion is not None and "prediction_criterion_state_dict" in ckpt:
            c_state = ckpt["prediction_criterion_state_dict"]
            if c_state is not None:
                prediction_criterion.load_state_dict(c_state)

        return cls(
            config=ckpt.get("config", {}),
            epoch=ckpt.get("epoch", 0),
            best_loss=ckpt.get("best_loss", {"valid_total" : float("inf")}),
            hist=ckpt.get("hist", []),
            crit=ckpt.get("crit", []),
            epoch_times=ckpt.get("epoch_times", []),
            converged=ckpt.get("converged", False),
            model=model,
            optimizer=optimizer,
            schedulers=schedulers,
            criteria=criteria,
            criteria_weights=ckpt.get("criteria_weights", {}),
            prediction_criterion=prediction_criterion,
            prediction_criterion_name=ckpt.get("prediction_criterion_name", None),
            latent=ckpt.get("latent", None),
            train_md=ckpt.get("train_md", {}),
            valid_md=ckpt.get("valid_md", {}),
        )

    def get_metric(self, metric_name: str) -> float:
        """Get the best value of a metric from history."""
        key = f"valid_{metric_name}"
        if key in self.best_loss:
            return self.best_loss[key]
        if metric_name.startswith("loss_"):
            key = f"valid_{metric_name[5:]}"
            if key in self.best_loss:
                return self.best_loss[key]
        raise KeyError(f"Metric '{metric_name}' not found in best_loss.")


@dataclass
class CVResult:
    params: Dict[str, Any]
    fold_metrics: List[float]
    mean_metric: float = 0.0
    std_metric: float = 0.0
    checkpoint_paths: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.mean_metric = float(np.mean(self.fold_metrics))
        self.std_metric = float(np.std(self.fold_metrics))

def aggregate_cv_results(results: List[Dict[str, Any]]):
    """
    The results are potentially from concurrent runs, and each is in the format of

    {'combo_idx', 'fold_idx', 'combo', 'metric_value', 'model_prefix'}

    This function aggregates them into CVResult objects by collecting fold results for each combo_idx.
    """
    tmp = [res['combo_idx'] for res in results]
    max_combo_idx, min_combo_idx = max(tmp), min(tmp)
    tmp = [[[], [], []] for _ in range(max_combo_idx - min_combo_idx + 1)]
    for res in results:
        c_idx = res['combo_idx'] - min_combo_idx
        tmp[c_idx][0].append(res['combo'])
        tmp[c_idx][1].append(res['metric_value'])
        tmp[c_idx][2].append(res['model_prefix'])

    cv_results = []
    for combos, metrics, paths in tmp:
        assert len(set(tuple(sorted(c.items())) for c in combos)) == 1, "Inconsistent combos for same combo_idx"
        cv_results.append(CVResult(
            params=combos[0],
            fold_metrics=metrics,
            checkpoint_paths=paths
        ))

    return cv_results

def iter_param_grid(param_grid: Dict[str, Iterable[Any]]):
    """
    param_grid: dict mapping dotted keys to iterables.
    Yields dicts mapping dotted keys -> single value.
    """
    keys = list(param_grid.keys())
    values_lists = []
    for k in keys:
        val = param_grid[k]
        if isinstance(val, list):
            values_lists.append(val)
        elif isinstance(val, tuple):
            if val[0] == 'linspace':
                values_lists.append(np.linspace(*val[1]).tolist())
            elif val[0] == 'logspace':
                values_lists.append(np.logspace(*val[1]).tolist())
            else:
                raise ValueError(f"Unknown param grid specifier: {val}")
        else:
            raise ValueError(f"Param grid values must be lists or tuples, got {type(val)}")
    for values in product(*values_lists):
        yield dict(zip(keys, values))

def set_by_dotted_key(d: Dict[str, Any], dotted_key: str, value: Any):
    """
    Set d['a']['b']['c'] when dotted_key is 'a.b.c'.
    Creates intermediate dicts as needed.
    """
    parts = dotted_key.split(".")
    curr = d
    for p in parts[:-1]:
        if p not in curr or not isinstance(curr[p], dict):
            curr[p] = {}
        curr = curr[p]
    curr[parts[-1]] = value
