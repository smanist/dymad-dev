"""Active object store for migration-boundary artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from dymad.sako.snapshot import SpectralSnapshot
    from dymad.store.filesystem_artifact_store import FilesystemArtifactStore
else:
    SpectralSnapshot = Any


class ObjectNotFoundError(KeyError):
    """Raised when a handle cannot be resolved from the active store."""


@dataclass(frozen=True)
class CheckpointRecord:
    handle: str
    model_ref: str
    checkpoint_path: str
    device: str


@dataclass(frozen=True)
class PredictionRequestRecord:
    handle: str
    checkpoint_handle: str
    horizon: int
    has_control: bool
    has_graph: bool


@dataclass(frozen=True)
class SpectralSnapshotRecord:
    handle: str
    checkpoint_handle: str
    snapshot: SpectralSnapshot


@dataclass(frozen=True)
class ObjectSummary:
    handle: str
    kind: str
    derived_from: str | None
    preview: str


class ObjectStore:
    """Active object store with optional filesystem-backed persistence."""

    def __init__(self, artifact_store: FilesystemArtifactStore | None = None) -> None:
        self._artifact_store = artifact_store
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._prediction_requests: dict[str, PredictionRequestRecord] = {}
        self._spectral_snapshots: dict[str, SpectralSnapshotRecord] = {}

    def put_checkpoint(self, *, model_ref: str, checkpoint_path: str, device: str) -> str:
        handle = self._new_handle("chk")
        record = CheckpointRecord(
            handle=handle,
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self._checkpoints[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_checkpoint(record)
        return handle

    def get_checkpoint(self, handle: str) -> CheckpointRecord:
        try:
            return self._checkpoints[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(f"unknown checkpoint handle: {handle}") from exc
            record = self._artifact_store.load_checkpoint(handle)
            self._checkpoints[handle] = record
            return record

    def put_prediction_request(
        self,
        *,
        checkpoint_handle: str,
        horizon: int,
        has_control: bool,
        has_graph: bool,
    ) -> str:
        # Validate derived handle exists before creating a request record.
        self.get_checkpoint(checkpoint_handle)
        handle = self._new_handle("pred")
        record = PredictionRequestRecord(
            handle=handle,
            checkpoint_handle=checkpoint_handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )
        self._prediction_requests[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_prediction_request(record)
        return handle

    def get_prediction_request(self, handle: str) -> PredictionRequestRecord:
        try:
            return self._prediction_requests[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(f"unknown prediction handle: {handle}") from exc
            record = self._artifact_store.load_prediction_request(handle)
            self._prediction_requests[handle] = record
            return record

    def put_spectral_snapshot(self, *, checkpoint_handle: str, snapshot: SpectralSnapshot) -> str:
        # Validate derived handle exists before creating a snapshot record.
        self.get_checkpoint(checkpoint_handle)
        handle = self._new_handle("specsnap")
        record = SpectralSnapshotRecord(
            handle=handle,
            checkpoint_handle=checkpoint_handle,
            snapshot=snapshot,
        )
        self._spectral_snapshots[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_spectral_snapshot(record)
        return handle

    def get_spectral_snapshot(self, handle: str) -> SpectralSnapshotRecord:
        try:
            return self._spectral_snapshots[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(f"unknown spectral snapshot handle: {handle}") from exc
            record = self._artifact_store.load_spectral_snapshot(handle)
            self._spectral_snapshots[handle] = record
            return record

    def summarize(self, handle: str) -> ObjectSummary:
        if handle.startswith("chk_"):
            checkpoint = self.get_checkpoint(handle)
            return ObjectSummary(
                handle=handle,
                kind="checkpoint",
                derived_from=None,
                preview=f"{checkpoint.model_ref} @ {checkpoint.checkpoint_path}",
            )
        if handle.startswith("pred_"):
            request = self.get_prediction_request(handle)
            return ObjectSummary(
                handle=handle,
                kind="prediction_request",
                derived_from=request.checkpoint_handle,
                preview=f"horizon={request.horizon}, control={request.has_control}, graph={request.has_graph}",
            )
        if handle.startswith("specsnap_"):
            snapshot_record = self.get_spectral_snapshot(handle)
            snapshot = snapshot_record.snapshot
            return ObjectSummary(
                handle=handle,
                kind="spectral_snapshot",
                derived_from=snapshot_record.checkpoint_handle,
                preview=f"samples={snapshot.sample_count}, obs_dim={snapshot.obs_dim}",
            )
        raise ObjectNotFoundError(f"unknown handle: {handle}")

    def list_summaries(self, *, kind: str | None = None) -> list[ObjectSummary]:
        summaries: dict[str, ObjectSummary] = {}
        if self._artifact_store is not None:
            for summary in self._artifact_store.list_object_summaries(kind=kind):
                summaries[summary.handle] = summary

        if kind in (None, "checkpoint"):
            for handle in self._checkpoints:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "prediction_request"):
            for handle in self._prediction_requests:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "spectral_snapshot"):
            for handle in self._spectral_snapshots:
                summaries[handle] = self.summarize(handle)
        return [summaries[handle] for handle in sorted(summaries)]

    @staticmethod
    def _new_handle(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:12]}"
