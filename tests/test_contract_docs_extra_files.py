from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_extra_file_list() -> list[str]:
    conf_path = REPO_ROOT / "docs" / "conf.py"
    tree = ast.parse(conf_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "extra_file_list"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, list)
        assert all(isinstance(item, str) for item in value)
        return value
    raise AssertionError("docs/conf.py does not define extra_file_list")


def test_docs_extra_files_are_repo_sources() -> None:
    for relative_path in _read_extra_file_list():
        source_path = REPO_ROOT / relative_path
        assert source_path.exists(), f"{relative_path} does not exist"
        result = subprocess.run(
            ["git", "check-ignore", relative_path],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, f"{relative_path} is ignored by git"
