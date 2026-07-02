"""Shared runtime environment setup for runnable scripts."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def find_repo_root(script_file: str | Path) -> Path:
    """Find the repository root for a script located under the checkout."""
    path = Path(script_file).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "src" / "dymad").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not find repository root for {script_file!s}.")


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def ensure_script_paths(script_file: str | Path) -> Path:
    """Ensure repo-local script and source imports work for runnable scripts."""
    repo_root = find_repo_root(script_file)
    _prepend_sys_path(repo_root / "scripts")
    _prepend_sys_path(repo_root / "src")
    return repo_root


def ensure_matplotlib_config_dir(dirname: str = "dymad_matplotlib") -> Path:
    """Ensure Matplotlib has a writable config/cache directory before import."""
    default_dir = Path(tempfile.gettempdir()) / dirname
    config_dir = Path(os.environ.get("MPLCONFIGDIR", default_dir))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    return config_dir


def ensure_keops_cache_dir(dirname: str = "dymad_keops_cache") -> Path:
    """Ensure PyKeOps has a writable cache directory before use."""
    default_dir = Path(tempfile.gettempdir()) / dirname
    cache_dir = Path(
        os.environ.get("KEOPS_CACHE_FOLDER", os.environ.get("PYKEOPS_CACHE_FOLDER", default_dir))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KEOPS_CACHE_FOLDER", str(cache_dir))
    os.environ.setdefault("PYKEOPS_CACHE_FOLDER", str(cache_dir))
    return cache_dir


def configure_script_runtime(
    script_file: str | Path,
    *,
    matplotlib: bool = False,
    keops: bool = False,
) -> Path:
    """Apply common runtime setup for repo scripts."""
    repo_root = ensure_script_paths(script_file)
    if matplotlib:
        ensure_matplotlib_config_dir()
    if keops:
        ensure_keops_cache_dir()
    return repo_root
