"""Adapters between typed trainer batches and legacy runtime payloads."""

from __future__ import annotations

from typing import TypeAlias

from dymad.core.model_context import LegacyRuntimeCollection
from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch
from dymad.io.legacy_runtime import LegacyRuntimeBatch

TrainerBatch: TypeAlias = LegacyRuntimeBatch | RegularTrainerBatch | GraphTrainerBatch
RuntimeBatch: TypeAlias = LegacyRuntimeBatch | LegacyRuntimeCollection


def batch_to_legacy_runtime(batch: TrainerBatch) -> RuntimeBatch:
    """Normalize trainer input batches to the temporary legacy runtime payload."""

    if isinstance(batch, LegacyRuntimeBatch):
        return batch
    return batch.runtime.to_legacy_runtime()


def iter_runtime_batches(runtime: RuntimeBatch):
    """Iterate over one or more legacy runtime payloads."""

    if isinstance(runtime, LegacyRuntimeCollection):
        return iter(runtime.items)
    return iter((runtime,))
