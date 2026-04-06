from __future__ import annotations

import argparse
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_cli_args(
    parser: argparse.ArgumentParser,
    *,
    include_data: bool = True,
    plot_help: str = "Skip summary plotting.",
    predict_help: str = "Skip prediction plotting.",
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--case", nargs="+", type=int, help="Case indices to run, for example '--case 0 1 2'."
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="Print available case indices and exit."
    )
    if include_data:
        parser.add_argument(
            "--data", action="store_true", help="Generate or stage data before other actions."
        )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Run in a separate working directory and stage the needed files there.",
    )
    parser.add_argument("--seed", type=int, help="Set random seeds for reproducible runs.")
    parser.add_argument("--no-train", action="store_true", help="Skip training.")
    parser.add_argument("--no-plot", action="store_true", help=plot_help)
    parser.add_argument("--no-predict", action="store_true", help=predict_help)
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show().")
    return parser


def resolve_case_indices(
    values: list[int] | None, n_cases: int, default_indices: Iterable[int]
) -> list[int]:
    indices = list(default_indices) if values is None else values
    invalid = [idx for idx in indices if idx < 0 or idx >= n_cases]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_case_table(cases: list[dict]) -> None:
    for idx, case in enumerate(cases):
        suffix = f" [{case['config']}]" if "config" in case else ""
        print(f"{idx}: {case['name']}{suffix}")


def stage_workdir(
    root: Path, base_dir: Path, relative_paths: Iterable[str | Path], *, data_dir: bool = True
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if data_dir:
        (root / "data").mkdir(exist_ok=True)
    for rel in relative_paths:
        rel_path = Path(rel)
        src = base_dir / rel_path
        dst = root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
