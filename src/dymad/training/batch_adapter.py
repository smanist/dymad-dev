"""Adapters between typed trainer batches and legacy runtime payloads."""

from __future__ import annotations

from typing import TypeAlias

from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch
from dymad.io.legacy_runtime import LegacyRuntimeBatch

TrainerBatch: TypeAlias = LegacyRuntimeBatch | RegularTrainerBatch | GraphTrainerBatch


def batch_to_legacy_runtime(batch: TrainerBatch) -> LegacyRuntimeBatch:
    """Normalize trainer input batches to the temporary legacy runtime payload."""

    if isinstance(batch, LegacyRuntimeBatch):
        return batch
    return batch.runtime.to_legacy_runtime()
