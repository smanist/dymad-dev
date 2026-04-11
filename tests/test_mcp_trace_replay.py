from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import build_server, generate_replay_script, load_trace_events

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_regular_dataset(path: Path) -> None:
    np.savez_compressed(
        path,
        t=np.linspace(0.0, 1.0, 5),
        x=np.ones((2, 5, 2)),
        u=np.zeros((2, 5, 1)),
    )


def test_mcp_trace_records_success_and_failure_events(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    trace_path = tmp_path / "trace.jsonl"
    artifact_root = tmp_path / "artifacts"
    replay_path = artifact_root / "replay_scripts" / "trace.py"
    server = build_server(
        context=build_default_context(artifact_root=artifact_root),
        trace_path=trace_path,
        name="DyMAD Trace Test",
    )

    registered = server._tool_manager.get_tool("register_dataset_file").fn(path=str(dataset_path))
    missing = server._tool_manager.get_tool("describe_object").fn(handle="bad")
    events = load_trace_events(trace_path)

    assert registered["ok"] is True
    assert missing["ok"] is False
    assert len(events) == 2
    assert replay_path.is_file()
    assert replay_path.stat().st_mode & 0o111

    success = events[0]
    failure = events[1]
    assert success.tool_name == "register_dataset_file"
    assert success.ok is True
    assert success.args == {"path": str(dataset_path)}
    assert success.result_summary.handles[0].kind == "dataset"
    assert success.result_summary.handles[0].variable_name == "dataset_handle_1"
    assert success.result_summary.artifact_paths[0].value == str(dataset_path)
    assert success.error is None
    assert success.started_at <= success.finished_at

    assert failure.tool_name == "describe_object"
    assert failure.ok is False
    assert failure.error is not None
    assert failure.error["type"] == "ObjectNotFoundError"
    assert "unknown handle: bad" in failure.error["message"]
    assert failure.result_summary.handles == []


def test_mcp_trace_tracks_handle_bindings_across_calls(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    trace_path = tmp_path / "trace.jsonl"
    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        trace_path=trace_path,
    )

    registered = server._tool_manager.get_tool("register_dataset_file").fn(path=str(dataset_path))
    dataset_handle = registered["data"]["summary"]["handle"]
    inspected = server._tool_manager.get_tool("inspect_dataset").fn(dataset_handle=dataset_handle)
    events = load_trace_events(trace_path)

    assert inspected["ok"] is True
    assert len(events) == 2
    assert events[1].args["dataset_handle"] == dataset_handle
    assert events[1].arg_bindings == [
        events[1]
        .arg_bindings[0]
        .__class__(
            path=["dataset_handle"],
            source_handle=dataset_handle,
            variable_name="dataset_handle_1",
        )
    ]


def test_mcp_trace_writes_replay_script_to_custom_path(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    trace_path = tmp_path / "trace.jsonl"
    replay_path = tmp_path / "scripts" / "custom_replay.py"
    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        trace_path=trace_path,
        replay_script_path=replay_path,
    )

    server._tool_manager.get_tool("register_dataset_file").fn(path=str(dataset_path))

    assert replay_path.is_file()
    script = replay_path.read_text(encoding="utf-8")
    assert "tools.register_dataset_file(**kwargs_1)" in script


def test_generate_replay_script_uses_bound_handle_variables(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    trace_path = tmp_path / "trace.jsonl"
    script_path = tmp_path / "replay_trace.py"
    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        trace_path=trace_path,
    )

    registered = server._tool_manager.get_tool("register_dataset_file").fn(path=str(dataset_path))
    dataset_handle = registered["data"]["summary"]["handle"]
    server._tool_manager.get_tool("inspect_dataset").fn(dataset_handle=dataset_handle)

    generate_replay_script(trace_path=trace_path, output_path=script_path)
    script = script_path.read_text(encoding="utf-8")

    assert "dataset_handle_1 = response_1['data']['summary']['handle']" in script
    assert "kwargs_2 = {'dataset_handle': dataset_handle_1}" in script
    assert "response_2 = tools.inspect_dataset(**kwargs_2)" in script


def test_generated_replay_script_reproduces_traced_execution(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    trace_path = tmp_path / "trace.jsonl"
    script_path = tmp_path / "replay_trace.py"
    replay_artifact_root = tmp_path / "replay_artifacts"
    server = build_server(
        context=build_default_context(artifact_root=tmp_path / "artifacts"),
        trace_path=trace_path,
    )

    registered = server._tool_manager.get_tool("register_dataset_file").fn(path=str(dataset_path))
    server._tool_manager.get_tool("inspect_dataset").fn(
        dataset_handle=registered["data"]["summary"]["handle"]
    )
    server._tool_manager.get_tool("describe_object").fn(handle="bad")
    generate_replay_script(trace_path=trace_path, output_path=script_path)

    env = dict(os.environ)
    pythonpath = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        pythonpath if not env.get("PYTHONPATH") else f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        [sys.executable, str(script_path), "--artifact-root", str(replay_artifact_root)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    persisted_datasets = list((replay_artifact_root / "datasets").glob("*.json"))
    assert len(persisted_datasets) == 1
