# ruff: noqa: E402

"""FastMCP server assembly for the persisted facade/exec boundary."""

from __future__ import annotations

from typing import Any

from dymad.agent.mcp._bootstrap import configure_headless_matplotlib_backend

configure_headless_matplotlib_backend()

from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.mcp.developer_tools import DeveloperTools
from dymad.agent.mcp.user_tools import UserTools
from dymad.agent.registry import DatasetKind


def build_server(
    *,
    context: ExecutionContext | None = None,
    name: str = "DyMAD Demo",
    mode: str = "both",
) -> Any:
    """Build one FastMCP server around the DemoTools adapter."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "fastmcp is required to build the MCP server. Install the 'fastmcp' package."
        ) from exc

    active_context = context or build_default_context()
    tools = DeveloperTools(context=active_context)
    user_tools = UserTools(context=active_context)
    server = FastMCP(name)
    include_user = mode in {"user", "both"}
    include_developer = mode in {"developer", "both"}
    if not include_user and not include_developer:
        raise ValueError("mode must be one of 'user', 'developer', or 'both'")

    if include_developer:

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
        def list_model_capabilities() -> dict[str, Any]:
            """List supported model families and implementation variants."""
            return tools.list_model_capabilities()

        @server.tool
        def resolve_model_capability(key_or_alias: str) -> dict[str, Any]:
            """Resolve one model family by canonical key, alias, or current model_ref."""
            return tools.resolve_model_capability(key_or_alias=key_or_alias)

        @server.tool
        def list_profile_capabilities() -> dict[str, Any]:
            """List training profiles and their current model/dataset mappings."""
            return tools.list_profile_capabilities()

        @server.tool
        def describe_training_capability(
            model_key: str,
            dataset_handle: str | None = None,
            dataset_kind: DatasetKind | None = None,
        ) -> dict[str, Any]:
            """Describe one training capability, including accepted override and phase schema."""
            return tools.describe_training_capability(
                model_key=model_key,
                dataset_handle=dataset_handle,
                dataset_kind=dataset_kind,
            )

    if include_user:

        @server.tool
        def list_training_capabilities(dataset_handle: str | None = None) -> dict[str, Any]:
            """List supported training workflows, optionally filtered by one dataset handle."""
            return user_tools.list_training_capabilities(dataset_handle=dataset_handle)

        @server.tool
        def list_analysis_capabilities() -> dict[str, Any]:
            """List supported analysis workflows."""
            return user_tools.list_analysis_capabilities()

        @server.tool
        def list_evaluation_capabilities(dataset_handle: str | None = None) -> dict[str, Any]:
            """List supported evaluation workflows and accepted metric/plot parameters."""
            return user_tools.list_evaluation_capabilities(dataset_handle=dataset_handle)

        @server.tool
        def describe_training_capability(
            model_key: str,
            dataset_handle: str | None = None,
            dataset_kind: DatasetKind | None = None,
        ) -> dict[str, Any]:
            """Describe one training capability, including accepted override and phase schema."""
            return user_tools.describe_training_capability(
                model_key=model_key,
                dataset_handle=dataset_handle,
                dataset_kind=dataset_kind,
            )

        @server.tool
        def compile_training_request(
            train_dataset_handle: str,
            model_key: str,
            valid_dataset_handle: str | None = None,
            reference_profile: str | None = None,
            overrides: dict[str, Any] | str | None = None,
            run_name: str | None = None,
            seed: int | None = None,
            device: str = "auto",
            max_workers: int = 1,
        ) -> dict[str, Any]:
            """Compile and persist one user-mode training request."""
            return user_tools.compile_training_request(
                train_dataset_handle=train_dataset_handle,
                model_key=model_key,
                valid_dataset_handle=valid_dataset_handle,
                reference_profile=reference_profile,
                overrides=overrides,
                run_name=run_name,
                seed=seed,
                device=device,
                max_workers=max_workers,
            )

        @server.tool
        def train_compiled_request(
            compiled_request_handle: str,
            artifact_root: str,
        ) -> dict[str, Any]:
            """Execute one persisted compiled training request."""
            return user_tools.train_compiled_request(
                compiled_request_handle=compiled_request_handle,
                artifact_root=artifact_root,
            )

        @server.tool
        def evaluate_checkpoint(
            checkpoint_handle: str,
            test_dataset_handle: str,
            metric: str,
            artifact_root: str,
            plot_selection: str = "median",
            max_plots: int = 1,
            predict_kwargs: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Evaluate one registered checkpoint against one registered test dataset."""
            return user_tools.evaluate_checkpoint(
                checkpoint_handle=checkpoint_handle,
                test_dataset_handle=test_dataset_handle,
                metric=metric,
                artifact_root=artifact_root,
                plot_selection=plot_selection,
                max_plots=max_plots,
                predict_kwargs=predict_kwargs,
            )

        @server.tool
        def compile_analysis_request(
            workflow_key: str,
            checkpoint_handle: str | None = None,
            dataset_handles: dict[str, str] | None = None,
            parameters: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Compile and persist one user-mode analysis request."""
            return user_tools.compile_analysis_request(
                workflow_key=workflow_key,
                checkpoint_handle=checkpoint_handle,
                dataset_handles=dataset_handles,
                parameters=parameters,
            )

        @server.tool
        def run_analysis_request(
            compiled_request_handle: str,
            artifact_root: str,
        ) -> dict[str, Any]:
            """Execute one persisted compiled analysis request."""
            return user_tools.run_analysis_request(
                compiled_request_handle=compiled_request_handle,
                artifact_root=artifact_root,
            )

    if include_developer:

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
