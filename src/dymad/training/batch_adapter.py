"""Adapters between trainer batches and native typed runtimes."""

from __future__ import annotations

from typing import TypeAlias

from dymad.core.model_context import LegacyRuntimeCollection
from dymad.core.runtime import TypedRuntime
from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch
from dymad.io.legacy_runtime import LegacyRuntimeBatch

TrainerBatch: TypeAlias = LegacyRuntimeBatch | RegularTrainerBatch | GraphTrainerBatch
RuntimeBatch: TypeAlias = TypedRuntime | LegacyRuntimeBatch


def batch_to_runtime(batch: TrainerBatch) -> RuntimeBatch:
    """Normalize trainer input batches to the native typed runtime payload."""

    if isinstance(batch, LegacyRuntimeBatch):
        return batch
    return batch.runtime.to_runtime()


def batch_to_legacy_runtime(batch: TrainerBatch) -> LegacyRuntimeBatch | LegacyRuntimeCollection:
    """Compatibility-only adapter retained for checkpoint and serializer seams."""

    if isinstance(batch, LegacyRuntimeBatch):
        return batch
    return batch.runtime.to_legacy_runtime()


def iter_runtime_batches(runtime: RuntimeBatch):
    """Iterate over one or more typed runtimes."""

    if hasattr(runtime, "is_uniform_length"):
        return runtime.iter_series() if not runtime.is_uniform_length else iter((runtime,))
    return iter((runtime,))
