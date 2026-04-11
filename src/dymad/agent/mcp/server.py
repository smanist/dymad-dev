"""`mcp.server.fastmcp` server assembly for the persisted facade/exec boundary."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.mcp.demo_tools import DemoTools
from dymad.agent.mcp.trace import JSONLTraceRecorder, TraceRecorder


def build_server(
    *,
    context: ExecutionContext | None = None,
    name: str = "DyMAD Demo",
    trace_path: str | PathLike[str] | None = None,
    replay_script_path: str | PathLike[str] | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> FastMCP:
    """Build one `mcp.server.fastmcp.FastMCP` server around the DemoTools adapter."""
    if trace_path is not None and trace_recorder is not None:
        raise ValueError("build_server accepts trace_path or trace_recorder, not both")
    if replay_script_path is not None and trace_recorder is not None:
        raise ValueError("build_server accepts replay_script_path only with trace_path")
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
    def resolve_training_intent(
        request_text: str,
        cwd: str | None = None,
        candidate_dataset_paths: list[str] | None = None,
        train_dataset_handle: str | None = None,
        valid_dataset_handle: str | None = None,
        overrides: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """Resolve a concise training request into a sparse structured training intent."""
        return tools.resolve_training_intent(
            request_text=request_text,
            cwd=cwd,
            candidate_dataset_paths=candidate_dataset_paths,
            train_dataset_handle=train_dataset_handle,
            valid_dataset_handle=valid_dataset_handle,
            overrides=overrides,
        )

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
    def prepare_prediction_request(
        checkpoint_handle: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> dict[str, Any]:
        """Persist one prediction request for downstream reuse."""
        return tools.prepare_prediction_request(
            checkpoint_handle=checkpoint_handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )

    @server.tool()
    def plan_checkpoint_prediction(
        model_ref: str,
        checkpoint_path: str,
        horizon: int,
        has_control: bool = False,
        has_graph: bool = False,
    ) -> dict[str, Any]:
        """Register checkpoint and prediction-request handles without materializing predictions."""
        return tools.plan_checkpoint_prediction(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )

    @server.tool()
    def predict_checkpoint(
        checkpoint_handle: str,
        dataset_handle: str | None = None,
        prediction_request_handle: str | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        selection: int | list[int] | None = None,
        artifact_root: str | None = None,
    ) -> dict[str, Any]:
        """Materialize rollout predictions and persist a prediction-result handle."""
        return tools.predict_checkpoint(
            checkpoint_handle=checkpoint_handle,
            dataset_handle=dataset_handle,
            prediction_request_handle=prediction_request_handle,
            predict_kwargs=predict_kwargs,
            selection=selection,
            artifact_root=artifact_root,
        )

    @server.tool()
    def compute_rollout_metrics(
        prediction_handle: str,
        metric_specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute one or more structured rollout metrics from a prediction result."""
        return tools.compute_rollout_metrics(
            prediction_handle=prediction_handle,
            metric_specs=metric_specs,
        )

    @server.tool()
    def plot_rollouts(
        prediction_handle: str,
        selection: str = "median",
        max_plots: int = 1,
    ) -> dict[str, Any]:
        """Render representative rollout plots from a prediction result."""
        return tools.plot_rollouts(
            prediction_handle=prediction_handle,
            selection=selection,
            max_plots=max_plots,
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

    recorder = trace_recorder
    if recorder is None:
        active_trace_path = (
            Path(trace_path).expanduser().resolve()
            if trace_path is not None
            else _default_trace_path(active_context)
        )
        active_replay_script_path = (
            Path(replay_script_path).expanduser().resolve()
            if replay_script_path is not None
            else _default_replay_script_path(active_context, active_trace_path)
        )
        recorder = JSONLTraceRecorder(
            active_trace_path,
            replay_script_path=active_replay_script_path,
        )
    if recorder is not None:
        _instrument_server(server, recorder)

    return server


def _instrument_server(server: FastMCP, recorder: TraceRecorder) -> None:
    for tool_name in server._tool_manager._tools:
        tool = server._tool_manager.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(f"registered MCP tool disappeared during tracing: {tool_name}")
        tool.fn = recorder.wrap_tool(tool_name, tool.fn)


def _default_trace_path(context: ExecutionContext) -> Path:
    return Path(context.artifact_store.root).resolve() / "mcp_trace.jsonl"


def _default_replay_script_path(context: ExecutionContext, trace_path: Path) -> Path:
    replay_dir = Path(context.artifact_store.root).resolve() / "replay_scripts"
    return replay_dir / f"{trace_path.stem}.py"


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
