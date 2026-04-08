"""MCP-facing adapters for the persisted facade/exec boundary."""

from dymad.mcp.demo_tools import DemoTools
from dymad.mcp.server import build_server

__all__ = [
    "DemoTools",
    "build_server",
]
