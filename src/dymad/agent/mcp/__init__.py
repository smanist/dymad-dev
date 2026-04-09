"""MCP-facing adapters for the persisted agent boundary."""

from dymad.agent.mcp.demo_tools import DemoTools


def build_server(*args, **kwargs):
    """Lazily import the server entrypoint to avoid `python -m` double-import warnings."""
    from dymad.agent.mcp.server import build_server as _build_server

    return _build_server(*args, **kwargs)


__all__ = [
    "DemoTools",
    "build_server",
]
