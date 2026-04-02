"""Adapters between trainer batches and native typed runtimes."""

from __future__ import annotations

from typing import TypeAlias

from dymad.core.graph_series import GraphSeries
from dymad.core.runtime import TypedRuntime, runtime_from_series
from dymad.core.series import RegularSeries
from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch

TrainerBatch: TypeAlias = RegularTrainerBatch | GraphTrainerBatch | RegularSeries | GraphSeries
RuntimeBatch: TypeAlias = TypedRuntime


def batch_to_runtime(batch: TrainerBatch) -> RuntimeBatch:
    """Normalize trainer input batches to the native typed runtime payload."""
    if isinstance(batch, (RegularSeries, GraphSeries)):
        return runtime_from_series(batch)
    return batch.runtime.to_runtime()


def iter_runtime_batches(runtime: RuntimeBatch):
    """Iterate over one or more typed runtimes."""

    return runtime.iter_series() if not runtime.is_uniform_length else iter((runtime,))
