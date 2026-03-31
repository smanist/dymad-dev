"""Phase/state runtime primitives for staged training-layer migration.

The adapters in this module are temporary compatibility seams while trainers
still consume ``RunState`` directly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from dymad.training.helper import RunState


@dataclass
class TrainerState:
    """Checkpointable trainer state, split from live data context."""

    config: Optional[Dict[str, Any]]
    device: Optional[torch.device] = None
    epoch: int = 0
    best_loss: Dict[str, float] = field(default_factory=lambda: {"valid_total": float("inf")})
    hist: List[Any] = field(default_factory=list)
    crit: List[Any] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    converged: bool = False
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    schedulers: List[Any] = field(default_factory=list)
    criteria: Optional[List[torch.nn.Module]] = None
    criteria_weights: Optional[List[float]] = None
    criteria_names: Optional[List[str]] = None


@dataclass
class PhaseContext:
    """Live phase context (datasets, loaders, metadata) for one run."""

    train_set: Optional[Dataset] = None
    valid_set: Optional[Dataset] = None
    train_loader: Optional[DataLoader] = None
    valid_loader: Optional[DataLoader] = None
    train_md: Optional[Dict[str, Any]] = None
    valid_md: Optional[Dict[str, Any]] = None


def run_state_to_trainer_state(run_state: RunState) -> TrainerState:
    """Temporary adapter: project legacy ``RunState`` into ``TrainerState``."""

    return TrainerState(
        config=copy.deepcopy(run_state.config),
        device=run_state.device,
        epoch=run_state.epoch,
        best_loss=copy.deepcopy(run_state.best_loss),
        hist=copy.deepcopy(run_state.hist),
        crit=copy.deepcopy(run_state.crit),
        epoch_times=copy.deepcopy(run_state.epoch_times),
        converged=run_state.converged,
        model=run_state.model,
        optimizer=run_state.optimizer,
        schedulers=list(run_state.schedulers),
        criteria=None if run_state.criteria is None else list(run_state.criteria),
        criteria_weights=None if run_state.criteria_weights is None else list(run_state.criteria_weights),
        criteria_names=None if run_state.criteria_names is None else list(run_state.criteria_names),
    )


def run_state_to_phase_context(run_state: RunState) -> PhaseContext:
    """Temporary adapter: extract live context from legacy ``RunState``."""

    return PhaseContext(
        train_set=run_state.train_set,
        valid_set=run_state.valid_set,
        train_loader=run_state.train_loader,
        valid_loader=run_state.valid_loader,
        train_md=run_state.train_md,
        valid_md=run_state.valid_md,
    )


def compose_run_state(trainer_state: TrainerState, phase_context: PhaseContext) -> RunState:
    """Temporary adapter: rebuild ``RunState`` for legacy trainer APIs."""

    return RunState(
        config=copy.deepcopy(trainer_state.config),
        device=trainer_state.device,
        epoch=trainer_state.epoch,
        best_loss=copy.deepcopy(trainer_state.best_loss),
        hist=copy.deepcopy(trainer_state.hist),
        crit=copy.deepcopy(trainer_state.crit),
        epoch_times=copy.deepcopy(trainer_state.epoch_times),
        converged=trainer_state.converged,
        model=trainer_state.model,
        optimizer=trainer_state.optimizer,
        schedulers=list(trainer_state.schedulers),
        criteria=None if trainer_state.criteria is None else list(trainer_state.criteria),
        criteria_weights=None if trainer_state.criteria_weights is None else list(trainer_state.criteria_weights),
        criteria_names=None if trainer_state.criteria_names is None else list(trainer_state.criteria_names),
        train_set=phase_context.train_set,
        valid_set=phase_context.valid_set,
        train_loader=phase_context.train_loader,
        valid_loader=phase_context.valid_loader,
        train_md=phase_context.train_md,
        valid_md=phase_context.valid_md,
    )
