"""Composition root for the migration boundary skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import os
from os import PathLike
from pathlib import Path

from dymad.agent.exec.workflow import CompatibilityExecutor
from dymad.agent.facade.operations import FacadeOperations
from dymad.agent.store.filesystem_artifact_store import FilesystemArtifactStore
from dymad.agent.store.object_store import ObjectStore


@dataclass(frozen=True)
class ExecutionContext:
    artifact_store: FilesystemArtifactStore | None
    store: ObjectStore
    facade: FacadeOperations
    executor: CompatibilityExecutor


def _find_repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _normalize_anchor_path(anchor_path: str | PathLike[str] | None) -> Path | None:
    if anchor_path is None:
        return None
    candidate = Path(anchor_path).expanduser()
    if candidate.exists():
        return candidate if candidate.is_dir() else candidate.parent
    if candidate.parent != candidate:
        return candidate.parent
    return None


def resolve_artifact_root(
    *,
    artifact_root: str | PathLike[str] | None = None,
    anchor_path: str | PathLike[str] | None = None,
) -> Path:
    if artifact_root is not None:
        return Path(artifact_root).expanduser().resolve()

    candidates: list[Path] = []
    anchor = _normalize_anchor_path(anchor_path)
    if anchor is not None:
        candidates.append(anchor.resolve())

    pwd = os.environ.get("PWD")
    if pwd:
        pwd_path = Path(pwd).expanduser()
        if pwd_path.exists():
            candidates.append(pwd_path.resolve())

    cwd = Path.cwd()
    if cwd.exists():
        candidates.append(cwd.resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        repo_root = _find_repo_root(candidate)
        if repo_root is not None:
            return repo_root / ".dymad" / "artifacts"
        if candidate != Path(candidate.anchor):
            return candidate / ".dymad" / "artifacts"

    return Path.home() / ".dymad" / "artifacts"


def build_default_context(
    *,
    artifact_root: str | PathLike[str] | None = None,
) -> ExecutionContext:
    """Wire filesystem store -> active store -> facade -> exec for boundary workflows."""

    def artifact_store_factory(
        anchor_path: str | PathLike[str] | None = None,
    ) -> FilesystemArtifactStore:
        root = resolve_artifact_root(artifact_root=artifact_root, anchor_path=anchor_path)
        return FilesystemArtifactStore(root)

    store = ObjectStore(artifact_store_factory=artifact_store_factory)
    facade = FacadeOperations(store)
    context_holder: dict[str, ExecutionContext] = {}
    executor = CompatibilityExecutor(facade, context_provider=lambda: context_holder["context"])
    context = ExecutionContext(
        artifact_store=None,
        store=store,
        facade=facade,
        executor=executor,
    )
    context_holder["context"] = context
    return context
