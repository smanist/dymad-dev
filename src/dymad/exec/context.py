"""Composition root for the migration boundary skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from dymad.exec.workflow import CompatibilityExecutor
from dymad.facade.operations import FacadeOperations
from dymad.store.filesystem_artifact_store import FilesystemArtifactStore
from dymad.store.object_store import ObjectStore


@dataclass(frozen=True)
class ExecutionContext:
    artifact_store: FilesystemArtifactStore
    store: ObjectStore
    facade: FacadeOperations
    executor: CompatibilityExecutor


def build_default_context(
    *,
    artifact_root: str | PathLike[str] | None = None,
) -> ExecutionContext:
    """Wire filesystem store -> active store -> facade -> exec for boundary workflows."""
    root = Path(artifact_root) if artifact_root is not None else Path(".dymad/artifacts")
    artifact_store = FilesystemArtifactStore(root)
    store = ObjectStore(artifact_store=artifact_store)
    facade = FacadeOperations(store)
    context_holder: dict[str, ExecutionContext] = {}
    executor = CompatibilityExecutor(facade, context_provider=lambda: context_holder["context"])
    context = ExecutionContext(
        artifact_store=artifact_store,
        store=store,
        facade=facade,
        executor=executor,
    )
    context_holder["context"] = context
    return context
