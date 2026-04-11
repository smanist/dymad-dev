"""User-mode MCP adapter over registry/compiler-backed training workflows."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from dymad.agent.compiler import TrainingRequest, compile_training_request
from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.registry import DatasetKind, list_training_capabilities


class UserTools:
    """Wrap registry/compiler-backed user workflows in JSON-safe envelopes."""

    def __init__(self, *, context: ExecutionContext | None = None) -> None:
        self._context = context or build_default_context()

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

    def compile_training_request(
        self,
        *,
        train_dataset_handle: str,
        model_key: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        overrides: dict[str, Any] | None = None,
        run_name: str | None = None,
        seed: int | None = None,
        device: str = "auto",
        max_workers: int = 1,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: self._compiled_request_data(
                self._context.facade.register_compiled_training_request(
                    compiled_request=compile_training_request(
                        facade=self._context.facade,
                        request=TrainingRequest(
                            train_dataset_handle=train_dataset_handle,
                            model_key=model_key,
                            valid_dataset_handle=valid_dataset_handle,
                            reference_profile=reference_profile,
                            overrides=overrides,
                            run_name=run_name,
                            seed=seed,
                            device=device,
                            max_workers=max_workers,
                        ),
                    )
                ).handle
            )
        )

    def train_compiled_request(
        self,
        *,
        compiled_request_handle: str,
        artifact_root: str,
    ) -> dict[str, Any]:
        return self._wrap(
            lambda: {
                "result": asdict(
                    self._context.executor.train_compiled_request(
                        compiled_request_handle=compiled_request_handle,
                        artifact_root=artifact_root,
                    )
                )
            }
        )

    def evaluate_checkpoint(
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

    def _compiled_request_data(self, handle: str) -> dict[str, Any]:
        summary = self._context.facade.describe_object(handle)
        compiled_request = self._context.facade.get_compiled_training_request(handle)
        return {
            "summary": asdict(summary),
            "compiled_request": asdict(compiled_request),
        }
