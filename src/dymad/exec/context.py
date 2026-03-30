"""Composition root for the migration boundary skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from dymad.exec.workflow import CompatibilityExecutor
from dymad.facade.operations import FacadeOperations
from dymad.store.object_store import ObjectStore


@dataclass(frozen=True)
class ExecutionContext:
    store: ObjectStore
    facade: FacadeOperations
    executor: CompatibilityExecutor


def build_default_context() -> ExecutionContext:
    """Wire store -> facade -> exec for typed-handle compatibility flow."""
    store = ObjectStore()
    facade = FacadeOperations(store)
    executor = CompatibilityExecutor(facade)
    return ExecutionContext(
        store=store,
        facade=facade,
        executor=executor,
    )
