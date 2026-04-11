"""Facade boundary operations for checkpoint-compatible prediction setup."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dymad.agent.facade.handles import (
    CheckpointHandle,
    CompiledTrainingRequestHandle,
    DatasetHandle,
    EvaluationHandle,
    PredictionHandle,
    SpectralSnapshotHandle,
    TrainingRunHandle,
)
from dymad.agent.store.object_store import (
    CheckpointRecord,
    CompiledTrainingRequestRecord,
    DatasetRecord,
    EvaluationRecord,
    ObjectStore,
    ObjectSummary,
    PredictionRequestRecord,
    SpectralSnapshotRecord,
    TrainingRunRecord,
)

if TYPE_CHECKING:
    from dymad.agent.compiler import CompiledTrainingRequest
    from dymad.sako.snapshot import SpectralSnapshot
else:
    CompiledTrainingRequest = Any
    SpectralSnapshot = Any


class FacadeOperations:
    """Stable typed boundary over the skeleton object store."""

    _DATASET_FORMATS = {"npz"}
    _DATASET_KINDS = {"regular", "graph"}
    _LISTABLE_KINDS = {
        "dataset",
        "checkpoint",
        "training_run",
        "compiled_training_request",
        "evaluation",
        "prediction_request",
        "spectral_snapshot",
    }

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def register_dataset_file(
        self,
        *,
        path: str,
        format: str = "npz",
        kind: str = "regular",
    ) -> ObjectSummary:
        normalized_path = str(Path(path).expanduser().resolve())
        if not Path(normalized_path).is_file():
            raise FileNotFoundError(f"dataset file does not exist: {normalized_path}")
        if format not in self._DATASET_FORMATS:
            raise ValueError(f"unsupported dataset format: {format}")
        if kind not in self._DATASET_KINDS:
            raise ValueError(f"unsupported dataset kind: {kind}")
        handle = self._store.put_dataset(
            path=normalized_path,
            format=format,
            kind=kind,
        )
        return self._store.summarize(handle)

    def register_checkpoint(
        self, *, model_ref: str, checkpoint_path: str, device: str = "cpu"
    ) -> ObjectSummary:
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

    def register_training_run(
        self,
        *,
        model_ref: str,
        train_dataset_handle: str,
        valid_dataset_handle: str | None,
        reference_profile: str | None,
        checkpoint_handle: str,
        artifact_root: str,
        run_name: str,
    ) -> ObjectSummary:
        if not model_ref.strip():
            raise ValueError("model_ref cannot be empty")
        if not artifact_root.strip():
            raise ValueError("artifact_root cannot be empty")
        if not run_name.strip():
            raise ValueError("run_name cannot be empty")
        train_dataset = DatasetHandle.parse(train_dataset_handle)
        valid_dataset = (
            None if valid_dataset_handle is None else DatasetHandle.parse(valid_dataset_handle)
        )
        checkpoint = CheckpointHandle.parse(checkpoint_handle)
        handle = self._store.put_training_run(
            model_ref=model_ref.strip(),
            train_dataset_handle=train_dataset.value,
            valid_dataset_handle=None if valid_dataset is None else valid_dataset.value,
            reference_profile=None
            if reference_profile is None
            else reference_profile.strip() or None,
            checkpoint_handle=checkpoint.value,
            artifact_root=artifact_root.strip(),
            run_name=run_name.strip(),
        )
        return self._store.summarize(handle)

    def register_compiled_training_request(
        self,
        *,
        compiled_request: CompiledTrainingRequest,
    ) -> ObjectSummary:
        train_dataset = DatasetHandle.parse(compiled_request.request.train_dataset_handle)
        valid_dataset = (
            None
            if compiled_request.request.valid_dataset_handle is None
            else DatasetHandle.parse(compiled_request.request.valid_dataset_handle)
        )
        handle = self._store.put_compiled_training_request(
            train_dataset_handle=train_dataset.value,
            valid_dataset_handle=None if valid_dataset is None else valid_dataset.value,
            model_key=compiled_request.model.key,
            model_ref=compiled_request.model_ref,
            reference_profile=compiled_request.profile.key,
            train_dataset_kind=compiled_request.train_dataset_kind,
            valid_dataset_kind=compiled_request.valid_dataset_kind,
            effective_run_name=compiled_request.effective_run_name,
            effective_config=compiled_request.effective_config,
            trainer_kind=compiled_request.trainer_kind,
            seed=compiled_request.request.seed,
            device=compiled_request.request.device,
            max_workers=compiled_request.request.max_workers,
            warnings=[asdict(warning) for warning in compiled_request.warnings],
        )
        return self._store.summarize(handle)

    def register_evaluation(
        self,
        *,
        checkpoint_handle: str,
        test_dataset_handle: str,
        metric: str,
        metrics_path: str,
        plot_paths: list[str],
    ) -> ObjectSummary:
        checkpoint = CheckpointHandle.parse(checkpoint_handle)
        dataset = DatasetHandle.parse(test_dataset_handle)
        if not metric.strip():
            raise ValueError("metric cannot be empty")
        if not metrics_path.strip():
            raise ValueError("metrics_path cannot be empty")
        handle = self._store.put_evaluation(
            checkpoint_handle=checkpoint.value,
            test_dataset_handle=dataset.value,
            metric=metric.strip(),
            metrics_path=metrics_path.strip(),
            plot_paths=list(plot_paths),
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

    def get_dataset(self, handle: str) -> DatasetRecord:
        dataset = DatasetHandle.parse(handle)
        return self._store.get_dataset(dataset.value)

    def get_checkpoint(self, handle: str) -> CheckpointRecord:
        checkpoint = CheckpointHandle.parse(handle)
        return self._store.get_checkpoint(checkpoint.value)

    def get_training_run(self, handle: str) -> TrainingRunRecord:
        run = TrainingRunHandle.parse(handle)
        return self._store.get_training_run(run.value)

    def get_compiled_training_request(self, handle: str) -> CompiledTrainingRequestRecord:
        request = CompiledTrainingRequestHandle.parse(handle)
        return self._store.get_compiled_training_request(request.value)

    def get_evaluation(self, handle: str) -> EvaluationRecord:
        evaluation = EvaluationHandle.parse(handle)
        return self._store.get_evaluation(evaluation.value)

    def register_spectral_snapshot(
        self,
        *,
        checkpoint_handle: str,
        snapshot: SpectralSnapshot,
    ) -> ObjectSummary:
        checkpoint = CheckpointHandle.parse(checkpoint_handle)
        handle = self._store.put_spectral_snapshot(
            checkpoint_handle=checkpoint.value,
            snapshot=snapshot,
        )
        return self._store.summarize(handle)

    def get_spectral_snapshot(self, handle: str) -> SpectralSnapshotRecord:
        snapshot = SpectralSnapshotHandle.parse(handle)
        return self._store.get_spectral_snapshot(snapshot.value)

    def describe_object(self, handle: str) -> ObjectSummary:
        return self._store.summarize(handle)

    def list_objects(self, *, kind: str | None = None) -> list[ObjectSummary]:
        if kind is not None and kind not in self._LISTABLE_KINDS:
            raise ValueError(f"unsupported object kind: {kind}")
        return self._store.list_summaries(kind=kind)
