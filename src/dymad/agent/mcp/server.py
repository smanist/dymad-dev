"""FastMCP server assembly for the persisted facade/exec boundary."""

from __future__ import annotations

from typing import Any

from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.mcp.demo_tools import DemoTools


def build_server(
    *,
    context: ExecutionContext | None = None,
    name: str = "DyMAD Demo",
) -> Any:
    """Build one FastMCP server around the DemoTools adapter."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "fastmcp is required to build the MCP server. Install the 'fastmcp' package."
        ) from exc

    active_context = context or build_default_context()
    tools = DemoTools(context=active_context)
    server = FastMCP(name)

    @server.tool
    def register_dataset_file(
        path: str,
        format: str = "npz",
        kind: str = "regular",
    ) -> dict[str, Any]:
        """Persist one dataset file reference and return its summary."""
        return tools.register_dataset_file(
            path=path,
            format=format,
            kind=kind,
        )

    @server.tool
    def inspect_dataset(dataset_handle: str) -> dict[str, Any]:
        """Inspect one persisted dataset and return a lightweight schema summary."""
        return tools.inspect_dataset(dataset_handle=dataset_handle)

    @server.tool
    def register_checkpoint(
        model_ref: str,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Persist a checkpoint reference and return its summary."""
        return tools.register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    @server.tool
    def prepare_prediction_request(
        checkpoint_handle: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> dict[str, Any]:
        """Persist one prediction request tied to an existing checkpoint handle."""
        return tools.prepare_prediction_request(
            checkpoint_handle=checkpoint_handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )

    @server.tool
    def plan_checkpoint_prediction(
        model_ref: str,
        checkpoint_path: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> dict[str, Any]:
        """Plan one checkpoint-backed prediction workflow."""
        return tools.plan_checkpoint_prediction(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )

    @server.tool
    def train_model(
        train_dataset_handle: str,
        artifact_root: str,
        model_ref: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        seed: int | None = None,
        device: str = "auto",
        max_workers: int = 1,
    ) -> dict[str, Any]:
        """Train one DyMAD model from registered datasets and structured config."""
        return tools.train_model(
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

    @server.tool
    def evaluate_model(
        checkpoint_handle: str,
        test_dataset_handle: str,
        metric: str,
        artifact_root: str,
        plot_selection: str = "median",
        max_plots: int = 1,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one registered checkpoint against one registered test dataset."""
        return tools.evaluate_model(
            checkpoint_handle=checkpoint_handle,
            test_dataset_handle=test_dataset_handle,
            metric=metric,
            artifact_root=artifact_root,
            plot_selection=plot_selection,
            max_plots=max_plots,
            predict_kwargs=predict_kwargs,
        )

    @server.tool
    def describe_object(handle: str) -> dict[str, Any]:
        """Return the stored summary for one known handle."""
        return tools.describe_object(handle=handle)

    @server.tool
    def list_objects(kind: str | None = None) -> dict[str, Any]:
        """List persisted object summaries, optionally filtered by kind."""
        return tools.list_objects(kind=kind)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
