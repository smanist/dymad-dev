"""Install repo-staged DyMAD Codex skills."""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "dymad-train-eval-workflow"
REPO_SKILL_NAMES = (
    "dymad-train-eval-workflow",
    "dymad-tuning-convergence-study",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "skills" / SKILL_NAME


@dataclass(frozen=True)
class InstallResult:
    source_dir: Path
    destination_dir: Path
    changed: bool


def _default_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


def _validate_source(source_dir: Path, skill_name: str) -> None:
    skill_file = source_dir / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"source skill is missing {skill_file}")
    if f"name: {skill_name}" not in skill_file.read_text(encoding="utf-8"):
        raise ValueError(f"{skill_file} does not declare skill name {skill_name!r}")


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
    skill_name: str = SKILL_NAME,
    source_dir: Path | None = None,
    codex_home: Path | None = None,
    dry_run: bool = False,
    check: bool = False,
) -> InstallResult:
    if source_dir is None:
        source_dir = REPO_ROOT / "skills" / skill_name
    source_dir = source_dir.expanduser().resolve()
    codex_home = _default_codex_home() if codex_home is None else codex_home.expanduser()
    destination_dir = codex_home / "skills" / skill_name
    _validate_source(source_dir, skill_name)
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
    tmp_dir = destination_parent / f".{skill_name}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.copytree(source_dir, tmp_dir)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    tmp_dir.replace(destination_dir)
    return result


def install_repo_skills(
    *,
    skill_names: tuple[str, ...] = REPO_SKILL_NAMES,
    codex_home: Path | None = None,
    dry_run: bool = False,
    check: bool = False,
) -> list[InstallResult]:
    return [
        install_skill(
            skill_name=skill_name,
            codex_home=codex_home,
            dry_run=dry_run,
            check=check,
        )
        for skill_name in skill_names
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install repo-staged DyMAD skills into CODEX_HOME/skills."
    )
    parser.add_argument(
        "--skill",
        choices=("all", *REPO_SKILL_NAMES),
        default="all",
        help="Repo skill to install, or all repo-staged DyMAD skills.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Explicit repo skill directory to install. Only valid with a single --skill value.",
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
    if args.source is not None and args.skill == "all":
        raise ValueError("--source requires --skill to name one skill")
    results = (
        install_repo_skills(
            codex_home=args.codex_home,
            dry_run=args.dry_run,
            check=args.check,
        )
        if args.skill == "all"
        else [
            install_skill(
                skill_name=args.skill,
                source_dir=args.source,
                codex_home=args.codex_home,
                dry_run=args.dry_run,
                check=args.check,
            )
        ]
    )
    if args.check:
        changed = [result for result in results if result.changed]
        if changed:
            for result in changed:
                print(f"Installed skill differs: {result.destination_dir}")
            return 1
        for result in results:
            print(f"Installed skill is up to date: {result.destination_dir}")
        return 0
    for result in results:
        if args.dry_run:
            verb = "would update" if result.changed else "is already current"
            print(f"Dry run: {result.destination_dir} {verb}")
        else:
            verb = "Installed" if result.changed else "Already up to date"
            print(f"{verb}: {result.destination_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
