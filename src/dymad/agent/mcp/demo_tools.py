"""Thin MCP-safe adapter over the execution context."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dymad.agent.exec.context import ExecutionContext, build_default_context
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

    def validate_dataset_compatibility(
        self,
        *,
        dataset_handle: str,
        model_ref: str,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "compatibility": asdict(
                    self._context.executor.validate_dataset_compatibility(
                        dataset_handle=dataset_handle,
                        model_ref=model_ref,
                    )
                )
            }
        )

    def list_model_families(self) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "model_families": [
                    asdict(item) for item in self._context.executor.list_model_families()
                ]
            }
        )

    def describe_model_family(self, *, model_ref: str) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "model_family": asdict(
                    self._context.executor.describe_model_family(model_ref=model_ref)
                )
            }
        )

    def list_reference_profiles(
        self,
        *,
        model_ref: str | None = None,
        dataset_kind: str | None = None,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "reference_profiles": [
                    asdict(item)
                    for item in self._context.executor.list_reference_profiles(
                        model_ref=model_ref,
                        dataset_kind=dataset_kind,
                    )
                ]
            }
        )

    def describe_reference_profile(self, *, profile_name: str) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "reference_profile": asdict(
                    self._context.executor.describe_reference_profile(profile_name=profile_name)
                )
            }
        )

    def validate_training_config(
        self,
        *,
        train_dataset_handle: str,
        model_ref: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "validation": asdict(
                    self._context.executor.validate_training_config(
                        train_dataset_handle=train_dataset_handle,
                        valid_dataset_handle=valid_dataset_handle,
                        model_ref=model_ref,
                        reference_profile=reference_profile,
                        config=config,
                        run_name=run_name,
                    )
                )
            }
        )

    def materialize_training_config(
        self,
        *,
        train_dataset_handle: str,
        artifact_root: str,
        model_ref: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "result": asdict(
                    self._context.executor.materialize_training_config(
                        train_dataset_handle=train_dataset_handle,
                        valid_dataset_handle=valid_dataset_handle,
                        model_ref=model_ref,
                        reference_profile=reference_profile,
                        config=config,
                        run_name=run_name,
                        artifact_root=artifact_root,
                    )
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

    def inspect_training_run(self, *, run_handle: str) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "inspection": asdict(
                    self._context.executor.inspect_training_run(run_handle=run_handle)
                )
            }
        )

    def list_training_artifacts(self, *, run_handle: str) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "artifacts": asdict(
                    self._context.executor.list_training_artifacts(run_handle=run_handle)
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
                "data": fn(),
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
    def _summary_data(summary: ObjectSummary) -> dict[str, Any]:
        return {"summary": asdict(summary)}

    def _dataset_data(self, handle: str) -> dict[str, Any]:
        summary = self._context.facade.describe_object(handle)
        dataset = self._context.facade.get_dataset(handle)
        return {
            "summary": asdict(summary),
            "dataset": asdict(dataset),
        }
