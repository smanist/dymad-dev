"""Thin MCP-safe adapter over the execution context."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.registry import (
    DatasetKind,
    describe_training_capability,
    list_evaluation_capabilities,
    list_model_capabilities,
    list_profile_capabilities,
    list_training_capabilities,
    resolve_model_capability,
)
from dymad.agent.store.object_store import ObjectSummary


class DemoTools:
    """Wrap facade/exec workflows in JSON-safe success/error envelopes."""

    def __init__(self, *, context: ExecutionContext | None = None) -> None:
        self._context = context or build_default_context()

    @property
    def context(self) -> ExecutionContext:
        return self._context

    def register_checkpoint(
        self,
        *,
        model_ref: str,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: self._summary_data(
                self._context.facade.register_checkpoint(
                    model_ref=model_ref,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
            )
        )

    def register_dataset_file(
        self,
        *,
        path: str,
        format: str = "npz",
        kind: str = "regular",
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: self._dataset_data(
                self._context.facade.register_dataset_file(
                    path=path,
                    format=format,
                    kind=kind,
                ).handle
            )
        )

    def inspect_dataset(self, *, dataset_handle: str) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "inspection": asdict(
                    self._context.executor.inspect_dataset(dataset_handle=dataset_handle)
                )
            }
        )

    def prepare_prediction_request(
        self,
        *,
        checkpoint_handle: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: self._summary_data(
                self._context.facade.prepare_prediction_request(
                    checkpoint_handle=checkpoint_handle,
                    horizon=horizon,
                    has_control=has_control,
                    has_graph=has_graph,
                )
            )
        )

    def plan_checkpoint_prediction(
        self,
        *,
        model_ref: str,
        checkpoint_path: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "plan": asdict(
                    self._context.executor.plan_checkpoint_prediction(
                        model_ref=model_ref,
                        checkpoint_path=checkpoint_path,
                        horizon=horizon,
                        has_control=has_control,
                        has_graph=has_graph,
                    )
                )
            }
        )

    def train_model(
        self,
        *,
        train_dataset_handle: str,
        valid_dataset_handle: str | None = None,
        model_ref: str,
        reference_profile: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        artifact_root: str,
        seed: int | None = None,
        device: str = "auto",
        max_workers: int = 1,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "result": asdict(
                    self._context.executor.train_model(
                        train_dataset_handle=train_dataset_handle,
                        valid_dataset_handle=valid_dataset_handle,
                        model_ref=model_ref,
                        reference_profile=reference_profile,
                        config=config,
                        run_name=run_name,
                        artifact_root=artifact_root,
                        seed=seed,
                        device=device,
                        max_workers=max_workers,
                    )
                )
            }
        )

    def evaluate_model(
        self,
        *,
        checkpoint_handle: str,
        test_dataset_handle: str,
        metric: str,
        artifact_root: str,
        plot_selection: str = "median",
        max_plots: int = 1,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "result": asdict(
                    self._context.executor.evaluate_model(
                        checkpoint_handle=checkpoint_handle,
                        test_dataset_handle=test_dataset_handle,
                        metric=metric,
                        artifact_root=artifact_root,
                        plot_selection=plot_selection,
                        max_plots=max_plots,
                        predict_kwargs=predict_kwargs,
                    )
                )
            }
        )

    def list_model_capabilities(self) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "capabilities": [asdict(capability) for capability in list_model_capabilities()]
            }
        )

    def resolve_model_capability(self, *, key_or_alias: str) -> dict[str, Any]:
        return self._wrap(lambda: {"capability": asdict(resolve_model_capability(key_or_alias))})

    def list_profile_capabilities(self) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "capabilities": [asdict(capability) for capability in list_profile_capabilities()]
            }
        )

    def list_training_capabilities(self, *, dataset_handle: str | None = None) -> dict[str, Any]:
        dataset_kind: DatasetKind | None = None
        if dataset_handle is not None:
            dataset_kind = cast(DatasetKind, self._context.facade.get_dataset(dataset_handle).kind)
        return self._wrap(
            lambda: {
                "dataset_kind": dataset_kind,
                "capabilities": [
                    asdict(capability)
                    for capability in list_training_capabilities(dataset_kind=dataset_kind)
                ],
            }
        )

    def list_evaluation_capabilities(self, *, dataset_handle: str | None = None) -> dict[str, Any]:
        dataset_kind: DatasetKind | None = None
        if dataset_handle is not None:
            dataset_kind = cast(DatasetKind, self._context.facade.get_dataset(dataset_handle).kind)
        return self._wrap(
            lambda: {
                "dataset_kind": dataset_kind,
                "capabilities": [
                    asdict(capability)
                    for capability in list_evaluation_capabilities(dataset_kind=dataset_kind)
                ],
            }
        )

    def describe_training_capability(
        self,
        *,
        model_key: str,
        dataset_handle: str | None = None,
        dataset_kind: DatasetKind | None = None,
    ) -> dict[str, Any]:
        if dataset_handle is not None:
            resolved_dataset_kind = cast(
                DatasetKind, self._context.facade.get_dataset(dataset_handle).kind
            )
        elif dataset_kind is not None:
            resolved_dataset_kind = dataset_kind
        else:
            raise ValueError("describe_training_capability requires dataset_handle or dataset_kind")
        return self._wrap(
            lambda: {
                "dataset_kind": resolved_dataset_kind,
                "detail": asdict(
                    describe_training_capability(
                        model_key=model_key,
                        dataset_kind=resolved_dataset_kind,
                    )
                ),
            }
        )

    def describe_object(self, *, handle: str) -> dict[str, Any]:
        return self._wrap(lambda: self._summary_data(self._context.facade.describe_object(handle)))

    def list_objects(self, *, kind: str | None = None) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "objects": [
                    asdict(summary) for summary in self._context.facade.list_objects(kind=kind)
                ]
            }
        )

    def _wrap(self, fn) -> dict[str, Any]:
        try:
            return {
                "ok": True,
                "data": self._json_safe(fn()),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: DemoTools._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DemoTools._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [DemoTools._json_safe(item) for item in value]
        return value

    @staticmethod
    def _summary_data(summary: ObjectSummary) -> dict[str, Any]:
        return {"summary": asdict(summary)}

    def _dataset_data(self, handle: str) -> dict[str, Any]:
        summary = self._context.facade.describe_object(handle)
        dataset = self._context.facade.get_dataset(handle)
        return {
            "summary": asdict(summary),
            "dataset": asdict(dataset),
        }
