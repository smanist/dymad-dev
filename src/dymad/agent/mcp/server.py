"""`mcp.server.fastmcp` server assembly for the persisted facade/exec boundary."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.mcp.demo_tools import DemoTools


def build_server(
    *,
    context: ExecutionContext | None = None,
    name: str = "DyMAD Demo",
) -> FastMCP:
    """Build one `mcp.server.fastmcp.FastMCP` server around the DemoTools adapter."""
    active_context = context or build_default_context()
    tools = DemoTools(context=active_context)
    server = FastMCP(name, json_response=True, log_level="ERROR")

    @server.tool()
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

    @server.tool()
    def inspect_dataset(dataset_handle: str) -> dict[str, Any]:
        """Inspect one persisted dataset and return a lightweight schema summary."""
        return tools.inspect_dataset(dataset_handle=dataset_handle)

    @server.tool()
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

    @server.tool()
    def validate_dataset_compatibility(
        dataset_handle: str,
        model_ref: str,
    ) -> dict[str, Any]:
        """Validate whether one registered dataset is compatible with one model family."""
        return tools.validate_dataset_compatibility(
            dataset_handle=dataset_handle,
            model_ref=model_ref,
        )

    @server.tool()
    def list_model_families() -> dict[str, Any]:
        """List available predefined DyMAD model families."""
        return tools.list_model_families()

    @server.tool()
    def describe_model_family(model_ref: str) -> dict[str, Any]:
        """Describe one predefined DyMAD model family."""
        return tools.describe_model_family(model_ref=model_ref)

    @server.tool()
    def list_reference_profiles(
        model_ref: str | None = None,
        dataset_kind: str | None = None,
    ) -> dict[str, Any]:
        """List available training reference profiles, optionally filtered."""
        return tools.list_reference_profiles(
            model_ref=model_ref,
            dataset_kind=dataset_kind,
        )

    @server.tool()
    def describe_reference_profile(profile_name: str) -> dict[str, Any]:
        """Describe one training reference profile."""
        return tools.describe_reference_profile(profile_name=profile_name)

    @server.tool()
    def validate_training_config(
        train_dataset_handle: str,
        model_ref: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        config: dict[str, Any] | str | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Validate one structured training request without executing training."""
        return tools.validate_training_config(
            train_dataset_handle=train_dataset_handle,
            valid_dataset_handle=valid_dataset_handle,
            model_ref=model_ref,
            reference_profile=reference_profile,
            config=config,
            run_name=run_name,
        )

    @server.tool()
    def materialize_training_config(
        train_dataset_handle: str,
        artifact_root: str,
        model_ref: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        config: dict[str, Any] | str | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Write one normalized training config without executing training."""
        return tools.materialize_training_config(
            train_dataset_handle=train_dataset_handle,
            valid_dataset_handle=valid_dataset_handle,
            model_ref=model_ref,
            reference_profile=reference_profile,
            config=config,
            run_name=run_name,
            artifact_root=artifact_root,
        )

    @server.tool()
    def train_model(
        train_dataset_handle: str,
        artifact_root: str,
        model_ref: str,
        valid_dataset_handle: str | None = None,
        reference_profile: str | None = None,
        config: dict[str, Any] | str | None = None,
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

    @server.tool()
    def inspect_training_run(run_handle: str) -> dict[str, Any]:
        """Inspect one persisted training run."""
        return tools.inspect_training_run(run_handle=run_handle)

    @server.tool()
    def list_training_artifacts(run_handle: str) -> dict[str, Any]:
        """List the standard artifact paths for one persisted training run."""
        return tools.list_training_artifacts(run_handle=run_handle)

    @server.tool()
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

    @server.tool()
    def describe_object(handle: str) -> dict[str, Any]:
        """Return the stored summary for one known handle."""
        return tools.describe_object(handle=handle)

    @server.tool()
    def list_objects(kind: str | None = None) -> dict[str, Any]:
        """List persisted object summaries, optionally filtered by kind."""
        return tools.list_objects(kind=kind)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
