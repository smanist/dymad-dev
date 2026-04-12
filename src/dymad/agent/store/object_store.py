"""Active object store for migration-boundary artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from dymad.agent.store.filesystem_artifact_store import FilesystemArtifactStore
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
class DatasetRecord:
    handle: str
    path: str
    format: str
    kind: str


@dataclass(frozen=True)
class TrainingRunRecord:
    handle: str
    model_ref: str
    train_dataset_handle: str
    valid_dataset_handle: str | None
    reference_profile: str | None
    checkpoint_handle: str
    artifact_root: str
    run_name: str


@dataclass(frozen=True)
class CompiledTrainingRequestRecord:
    handle: str
    train_dataset_handle: str
    valid_dataset_handle: str | None
    model_key: str
    model_ref: str
    reference_profile: str
    train_dataset_kind: str
    valid_dataset_kind: str | None
    effective_run_name: str
    effective_config: dict[str, Any]
    trainer_kind: str
    seed: int | None
    device: str
    max_workers: int
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class CompiledAnalysisRequestRecord:
    handle: str
    workflow_key: str
    checkpoint_handle: str | None
    dataset_handles: dict[str, str]
    parameters: dict[str, Any]
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class EvaluationRecord:
    handle: str
    checkpoint_handle: str
    test_dataset_handle: str
    metric: str
    metrics_path: str
    plot_paths: list[str]


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
        self._datasets: dict[str, DatasetRecord] = {}
        self._training_runs: dict[str, TrainingRunRecord] = {}
        self._compiled_training_requests: dict[str, CompiledTrainingRequestRecord] = {}
        self._compiled_analysis_requests: dict[str, CompiledAnalysisRequestRecord] = {}
        self._evaluations: dict[str, EvaluationRecord] = {}
        self._prediction_requests: dict[str, PredictionRequestRecord] = {}
        self._spectral_snapshots: dict[str, SpectralSnapshotRecord] = {}

    def put_dataset(self, *, path: str, format: str, kind: str) -> str:
        handle = self._new_handle("ds")
        record = DatasetRecord(
            handle=handle,
            path=path,
            format=format,
            kind=kind,
        )
        self._datasets[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_dataset(record)
        return handle

    def get_dataset(self, handle: str) -> DatasetRecord:
        try:
            return self._datasets[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(f"unknown dataset handle: {handle}") from exc
            record = self._artifact_store.load_dataset(handle)
            self._datasets[handle] = record
            return record

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

    def put_training_run(
        self,
        *,
        model_ref: str,
        train_dataset_handle: str,
        valid_dataset_handle: str | None,
        reference_profile: str | None,
        checkpoint_handle: str,
        artifact_root: str,
        run_name: str,
    ) -> str:
        self.get_dataset(train_dataset_handle)
        if valid_dataset_handle is not None:
            self.get_dataset(valid_dataset_handle)
        self.get_checkpoint(checkpoint_handle)
        handle = self._new_handle("run")
        record = TrainingRunRecord(
            handle=handle,
            model_ref=model_ref,
            train_dataset_handle=train_dataset_handle,
            valid_dataset_handle=valid_dataset_handle,
            reference_profile=reference_profile,
            checkpoint_handle=checkpoint_handle,
            artifact_root=artifact_root,
            run_name=run_name,
        )
        self._training_runs[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_training_run(record)
        return handle

    def get_training_run(self, handle: str) -> TrainingRunRecord:
        try:
            return self._training_runs[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(f"unknown training run handle: {handle}") from exc
            record = self._artifact_store.load_training_run(handle)
            self._training_runs[handle] = record
            return record

    def put_compiled_training_request(
        self,
        *,
        train_dataset_handle: str,
        valid_dataset_handle: str | None,
        model_key: str,
        model_ref: str,
        reference_profile: str,
        train_dataset_kind: str,
        valid_dataset_kind: str | None,
        effective_run_name: str,
        effective_config: dict[str, Any],
        trainer_kind: str,
        seed: int | None,
        device: str,
        max_workers: int,
        warnings: list[dict[str, Any]],
    ) -> str:
        self.get_dataset(train_dataset_handle)
        if valid_dataset_handle is not None:
            self.get_dataset(valid_dataset_handle)
        handle = self._new_handle("trainreq")
        record = CompiledTrainingRequestRecord(
            handle=handle,
            train_dataset_handle=train_dataset_handle,
            valid_dataset_handle=valid_dataset_handle,
            model_key=model_key,
            model_ref=model_ref,
            reference_profile=reference_profile,
            train_dataset_kind=train_dataset_kind,
            valid_dataset_kind=valid_dataset_kind,
            effective_run_name=effective_run_name,
            effective_config=effective_config,
            trainer_kind=trainer_kind,
            seed=seed,
            device=device,
            max_workers=max_workers,
            warnings=list(warnings),
        )
        self._compiled_training_requests[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_compiled_training_request(record)
        return handle

    def get_compiled_training_request(self, handle: str) -> CompiledTrainingRequestRecord:
        try:
            return self._compiled_training_requests[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(
                    f"unknown compiled training request handle: {handle}"
                ) from exc
            record = self._artifact_store.load_compiled_training_request(handle)
            self._compiled_training_requests[handle] = record
            return record

    def put_compiled_analysis_request(
        self,
        *,
        workflow_key: str,
        checkpoint_handle: str | None,
        dataset_handles: dict[str, str],
        parameters: dict[str, Any],
        warnings: list[dict[str, Any]],
    ) -> str:
        if checkpoint_handle is not None:
            self.get_checkpoint(checkpoint_handle)
        for handle in dataset_handles.values():
            self.get_dataset(handle)
        handle = self._new_handle("analysisreq")
        record = CompiledAnalysisRequestRecord(
            handle=handle,
            workflow_key=workflow_key,
            checkpoint_handle=checkpoint_handle,
            dataset_handles=dict(dataset_handles),
            parameters=dict(parameters),
            warnings=list(warnings),
        )
        self._compiled_analysis_requests[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_compiled_analysis_request(record)
        return handle

    def get_compiled_analysis_request(self, handle: str) -> CompiledAnalysisRequestRecord:
        try:
            return self._compiled_analysis_requests[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(
                    f"unknown compiled analysis request handle: {handle}"
                ) from exc
            record = self._artifact_store.load_compiled_analysis_request(handle)
            self._compiled_analysis_requests[handle] = record
            return record

    def put_evaluation(
        self,
        *,
        checkpoint_handle: str,
        test_dataset_handle: str,
        metric: str,
        metrics_path: str,
        plot_paths: list[str],
    ) -> str:
        self.get_checkpoint(checkpoint_handle)
        self.get_dataset(test_dataset_handle)
        handle = self._new_handle("eval")
        record = EvaluationRecord(
            handle=handle,
            checkpoint_handle=checkpoint_handle,
            test_dataset_handle=test_dataset_handle,
            metric=metric,
            metrics_path=metrics_path,
            plot_paths=list(plot_paths),
        )
        self._evaluations[handle] = record
        if self._artifact_store is not None:
            self._artifact_store.persist_evaluation(record)
        return handle

    def get_evaluation(self, handle: str) -> EvaluationRecord:
        try:
            return self._evaluations[handle]
        except KeyError as exc:
            if self._artifact_store is None:
                raise ObjectNotFoundError(f"unknown evaluation handle: {handle}") from exc
            record = self._artifact_store.load_evaluation(handle)
            self._evaluations[handle] = record
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
        if handle.startswith("ds_"):
            dataset = self.get_dataset(handle)
            return ObjectSummary(
                handle=handle,
                kind="dataset",
                derived_from=None,
                preview=f"{dataset.format} {dataset.kind} @ {dataset.path}",
            )
        if handle.startswith("chk_"):
            checkpoint = self.get_checkpoint(handle)
            return ObjectSummary(
                handle=handle,
                kind="checkpoint",
                derived_from=None,
                preview=f"{checkpoint.model_ref} @ {checkpoint.checkpoint_path}",
            )
        if handle.startswith("run_"):
            run = self.get_training_run(handle)
            return ObjectSummary(
                handle=handle,
                kind="training_run",
                derived_from=run.checkpoint_handle,
                preview=f"{run.run_name} ({run.model_ref})",
            )
        if handle.startswith("trainreq_"):
            request = self.get_compiled_training_request(handle)
            return ObjectSummary(
                handle=handle,
                kind="compiled_training_request",
                derived_from=request.train_dataset_handle,
                preview=(
                    f"{request.model_key}/{request.train_dataset_kind} -> "
                    f"{request.reference_profile} ({request.trainer_kind})"
                ),
            )
        if handle.startswith("analysisreq_"):
            request = self.get_compiled_analysis_request(handle)
            return ObjectSummary(
                handle=handle,
                kind="compiled_analysis_request",
                derived_from=request.checkpoint_handle
                or next(iter(request.dataset_handles.values())),
                preview=request.workflow_key,
            )
        if handle.startswith("eval_"):
            evaluation = self.get_evaluation(handle)
            return ObjectSummary(
                handle=handle,
                kind="evaluation",
                derived_from=evaluation.checkpoint_handle,
                preview=f"{evaluation.metric} @ {evaluation.metrics_path}",
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

        if kind in (None, "dataset"):
            for handle in self._datasets:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "checkpoint"):
            for handle in self._checkpoints:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "training_run"):
            for handle in self._training_runs:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "compiled_training_request"):
            for handle in self._compiled_training_requests:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "compiled_analysis_request"):
            for handle in self._compiled_analysis_requests:
                summaries[handle] = self.summarize(handle)
        if kind in (None, "evaluation"):
            for handle in self._evaluations:
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
