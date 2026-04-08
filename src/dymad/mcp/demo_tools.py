"""Thin MCP-safe adapter over the execution context."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dymad.exec.context import ExecutionContext, build_default_context
from dymad.store.object_store import ObjectSummary


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
