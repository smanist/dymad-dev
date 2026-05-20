"""Install the repo-staged DyMAD train/eval Codex skill."""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "dymad-train-eval-workflow"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "skills" / SKILL_NAME


@dataclass(frozen=True)
class InstallResult:
    source_dir: Path
    destination_dir: Path
    changed: bool


def _default_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


def _validate_source(source_dir: Path) -> None:
    skill_file = source_dir / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"source skill is missing {skill_file}")
    if f"name: {SKILL_NAME}" not in skill_file.read_text(encoding="utf-8"):
        raise ValueError(f"{skill_file} does not declare skill name {SKILL_NAME!r}")


def _relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def skill_trees_match(source_dir: Path, destination_dir: Path) -> bool:
    if not destination_dir.is_dir():
        return False
    source_files = _relative_files(source_dir)
    destination_files = _relative_files(destination_dir)
    if source_files != destination_files:
        return False
    return all(
        filecmp.cmp(source_dir / rel_path, destination_dir / rel_path, shallow=False)
        for rel_path in source_files
    )


def install_skill(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    codex_home: Path | None = None,
    dry_run: bool = False,
    check: bool = False,
) -> InstallResult:
    source_dir = source_dir.expanduser().resolve()
    codex_home = _default_codex_home() if codex_home is None else codex_home.expanduser()
    destination_dir = codex_home / "skills" / SKILL_NAME
    _validate_source(source_dir)
    changed = not skill_trees_match(source_dir, destination_dir)
    result = InstallResult(
        source_dir=source_dir,
        destination_dir=destination_dir,
        changed=changed,
    )
    if check or dry_run or not changed:
        return result

    destination_parent = destination_dir.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = destination_parent / f".{SKILL_NAME}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.copytree(source_dir, tmp_dir)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    tmp_dir.replace(destination_dir)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install skills/dymad-train-eval-workflow into CODEX_HOME/skills."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Repo skill directory to install.",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=None,
        help="Codex home directory. Defaults to CODEX_HOME or ~/.codex.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return nonzero if the installed skill differs from the repo copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report whether installation would change files without copying.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = install_skill(
        source_dir=args.source,
        codex_home=args.codex_home,
        dry_run=args.dry_run,
        check=args.check,
    )
    if args.check:
        if result.changed:
            print(f"Installed skill differs: {result.destination_dir}")
            return 1
        print(f"Installed skill is up to date: {result.destination_dir}")
        return 0
    if args.dry_run:
        verb = "would update" if result.changed else "is already current"
        print(f"Dry run: {result.destination_dir} {verb}")
        return 0
    verb = "Installed" if result.changed else "Already up to date"
    print(f"{verb}: {result.destination_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
