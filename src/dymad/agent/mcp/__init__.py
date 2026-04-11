"""MCP-facing adapters for the persisted agent boundary."""

from dymad.agent.mcp.demo_tools import DemoTools
from dymad.agent.mcp.replay import generate_replay_script
from dymad.agent.mcp.trace import JSONLTraceRecorder, load_trace_events


def build_server(*args, **kwargs):
    """Lazily import the server entrypoint to avoid `python -m` double-import warnings."""
    from dymad.agent.mcp.server import build_server as _build_server

    return _build_server(*args, **kwargs)


__all__ = [
    "DemoTools",
    "JSONLTraceRecorder",
    "build_server",
    "generate_replay_script",
    "load_trace_events",
]
