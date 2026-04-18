"""Bootstrap helpers for MCP-facing modules."""

from __future__ import annotations

import os
import sys


def configure_headless_matplotlib_backend() -> None:
    """Default MCP usage to a headless matplotlib backend unless the caller overrides it."""
    if os.environ.get("MPLBACKEND"):
        return

    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is not None and "matplotlib.pyplot" not in sys.modules:
        use = getattr(matplotlib, "use", None)
        if callable(use):
            try:
                use("Agg")
            except Exception:
                pass

    os.environ["MPLBACKEND"] = "Agg"
