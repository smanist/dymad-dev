# ruff: noqa: E402

"""MCP-facing adapters for the persisted agent boundary."""

from dymad.agent.mcp._bootstrap import configure_headless_matplotlib_backend

configure_headless_matplotlib_backend()

from dymad.agent.mcp.demo_tools import DemoTools
from dymad.agent.mcp.developer_tools import DeveloperTools
from dymad.agent.mcp.server import build_server
from dymad.agent.mcp.user_tools import UserTools

__all__ = [
    "DeveloperTools",
    "DemoTools",
    "UserTools",
    "build_server",
]
