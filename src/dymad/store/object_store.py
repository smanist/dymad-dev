"""Minimal object store for migration-boundary skeleton flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from dymad.sako.snapshot import SpectralSnapshot
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
    """In-memory store used by the first facade/store/exec skeleton."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._prediction_requests: dict[str, PredictionRequestRecord] = {}
        self._spectral_snapshots: dict[str, SpectralSnapshotRecord] = {}

    def put_checkpoint(self, *, model_ref: str, checkpoint_path: str, device: str) -> str:
        handle = self._new_handle("chk")
        self._checkpoints[handle] = CheckpointRecord(
            handle=handle,
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        return handle

    def get_checkpoint(self, handle: str) -> CheckpointRecord:
        try:
            return self._checkpoints[handle]
        except KeyError as exc:
            raise ObjectNotFoundError(f"unknown checkpoint handle: {handle}") from exc

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
        self._prediction_requests[handle] = PredictionRequestRecord(
            handle=handle,
            checkpoint_handle=checkpoint_handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )
        return handle

    def get_prediction_request(self, handle: str) -> PredictionRequestRecord:
        try:
            return self._prediction_requests[handle]
        except KeyError as exc:
            raise ObjectNotFoundError(f"unknown prediction handle: {handle}") from exc

    def put_spectral_snapshot(self, *, checkpoint_handle: str, snapshot: SpectralSnapshot) -> str:
        # Validate derived handle exists before creating a snapshot record.
        self.get_checkpoint(checkpoint_handle)
        handle = self._new_handle("specsnap")
        self._spectral_snapshots[handle] = SpectralSnapshotRecord(
            handle=handle,
            checkpoint_handle=checkpoint_handle,
            snapshot=snapshot,
        )
        return handle

    def get_spectral_snapshot(self, handle: str) -> SpectralSnapshotRecord:
        try:
            return self._spectral_snapshots[handle]
        except KeyError as exc:
            raise ObjectNotFoundError(f"unknown spectral snapshot handle: {handle}") from exc

    def summarize(self, handle: str) -> ObjectSummary:
        if handle in self._checkpoints:
            checkpoint = self._checkpoints[handle]
            return ObjectSummary(
                handle=handle,
                kind="checkpoint",
                derived_from=None,
                preview=f"{checkpoint.model_ref} @ {checkpoint.checkpoint_path}",
            )
        if handle in self._prediction_requests:
            request = self._prediction_requests[handle]
            return ObjectSummary(
                handle=handle,
                kind="prediction_request",
                derived_from=request.checkpoint_handle,
                preview=f"horizon={request.horizon}, control={request.has_control}, graph={request.has_graph}",
            )
        if handle in self._spectral_snapshots:
            snapshot_record = self._spectral_snapshots[handle]
            snapshot = snapshot_record.snapshot
            return ObjectSummary(
                handle=handle,
                kind="spectral_snapshot",
                derived_from=snapshot_record.checkpoint_handle,
                preview=f"samples={snapshot.sample_count}, obs_dim={snapshot.obs_dim}",
            )
        raise ObjectNotFoundError(f"unknown handle: {handle}")

    @staticmethod
    def _new_handle(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:12]}"
