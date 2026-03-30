"""Facade boundary operations for checkpoint-compatible prediction setup."""

from __future__ import annotations

from dymad.facade.handles import CheckpointHandle, PredictionHandle
from dymad.store.object_store import (
    CheckpointRecord,
    ObjectStore,
    ObjectSummary,
    PredictionRequestRecord,
)


class FacadeOperations:
    """Stable typed boundary over the skeleton object store."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def register_checkpoint(self, *, model_ref: str, checkpoint_path: str, device: str = "cpu") -> ObjectSummary:
        if not model_ref.strip():
            raise ValueError("model_ref cannot be empty")
        if not checkpoint_path.strip():
            raise ValueError("checkpoint_path cannot be empty")
        handle = self._store.put_checkpoint(
            model_ref=model_ref.strip(),
            checkpoint_path=checkpoint_path.strip(),
            device=device.strip() or "cpu",
        )
        return self._store.summarize(handle)

    def prepare_prediction_request(
        self,
        *,
        checkpoint_handle: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> ObjectSummary:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        checkpoint = CheckpointHandle.parse(checkpoint_handle)
        handle = self._store.put_prediction_request(
            checkpoint_handle=checkpoint.value,
            horizon=int(horizon),
            has_control=bool(has_control),
            has_graph=bool(has_graph),
        )
        return self._store.summarize(handle)

    def get_prediction_request(self, handle: str) -> PredictionRequestRecord:
        request = PredictionHandle.parse(handle)
        return self._store.get_prediction_request(request.value)

    def get_checkpoint(self, handle: str) -> CheckpointRecord:
        checkpoint = CheckpointHandle.parse(handle)
        return self._store.get_checkpoint(checkpoint.value)

    def describe_object(self, handle: str) -> ObjectSummary:
        return self._store.summarize(handle)
