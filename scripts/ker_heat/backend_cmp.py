"""Visual KeOps-vs-dense checks for DyMAD diffusion kernels."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import cast

# ruff: noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runtime_env import configure_script_runtime  # noqa: E402

configure_script_runtime(__file__, matplotlib=True)

import matplotlib
import numpy as np
import torch
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dymad.kernel_analysis import DiffusionHeatSections  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "keops_dense_verify"
CSV_PATH = OUT / "keops_dense_errors.csv"

EPSILON = 0.01
STEPS = 2
REF_COUNT = 192
LOC_COUNT = 96
SEED = 2026070101
CASES = ("circle", "disk")

ifrun = 1
ifplt = 1


def circle_points(n: int, seed: int | None = None) -> np.ndarray:
    if seed is None:
        theta = 2.0 * math.pi * (np.arange(n, dtype=float) + 0.5) / n
    else:
        unit = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(math.ceil(math.log2(n)))[:n]
        theta = 2.0 * math.pi * unit[:, 0]
    return np.column_stack((np.cos(theta), np.sin(theta)))


def disk_points(n: int, seed: int) -> np.ndarray:
    unit = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(math.ceil(math.log2(n)))[:n]
    radius, theta = np.sqrt(unit[:, 0]), 2.0 * math.pi * unit[:, 1]
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def case_points(case: str, n_ref: int, n_loc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case == "circle":
        sources = circle_points(4)
        return circle_points(n_ref, SEED), circle_points(n_loc), sources
    if case == "disk":
        sources = np.asarray([[0.0, 0.0], [0.35, 0.0], [0.0, 0.55], [0.75, 0.0]])
        return disk_points(n_ref, SEED + 11), disk_points(n_loc, SEED + 17), sources
    raise ValueError(f"unknown case {case!r}")


def kernel(ref: np.ndarray, *, backend: str) -> DiffusionHeatSections:
    model = DiffusionHeatSections(
        in_dim=2,
        eps_init=EPSILON,
        alpha_init=1.0,
        dtype=torch.float64,
        backend=backend,
    )
    model.set_reference_data(torch.as_tensor(ref, dtype=torch.float64))
    return model


def gram(model: DiffusionHeatSections, ref: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(ref, dtype=torch.float64)
    return model.kernel.materialize(tensor, tensor).detach().cpu().numpy()


def sections(
    model: DiffusionHeatSections, locations: np.ndarray, sources: np.ndarray
) -> np.ndarray:
    values = cast(
        torch.Tensor,
        model.heat_kernel(
            torch.as_tensor(locations, dtype=torch.float64),
            torch.as_tensor(sources, dtype=torch.float64),
            mode="uniform",
            steps=STEPS,
            mass_normalization="none",
        ),
    )
    return values.detach().cpu().numpy().T


def rel_fro(diff: np.ndarray, base: np.ndarray) -> float:
    return float(np.linalg.norm(diff) / max(np.linalg.norm(base), np.finfo(float).tiny))


def compare_case(
    case: str,
    n_ref: int = REF_COUNT,
    n_loc: int = LOC_COUNT,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    ref, loc, src = case_points(case, n_ref, n_loc)
    dense = kernel(ref, backend="torch")
    keops = kernel(ref, backend="keops")
    gram_dense = gram(dense, ref)
    section_dense = sections(dense, loc, src)
    gram_diff = gram(keops, ref) - gram_dense
    section_diff = sections(keops, loc, src) - section_dense
    row = {
        "case": case,
        "backend": "keops",
        "gram_max_abs": float(np.max(np.abs(gram_diff))),
        "gram_rel_fro": rel_fro(gram_diff, gram_dense),
        "section_max_abs": float(np.max(np.abs(section_diff))),
        "section_rel_fro": rel_fro(section_diff, section_dense),
    }
    return row, gram_diff, section_diff


def write_rows(rows: list[dict[str, float | str]], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "backend",
        "gram_max_abs",
        "gram_rel_fro",
        "section_max_abs",
        "section_rel_fro",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_case(
    case: str,
    gram_diff: np.ndarray,
    section_diff: np.ndarray,
    output_dir: Path = OUT,
) -> None:
    gram_image = np.log10(np.abs(gram_diff) + 1e-16)
    section_image = np.log10(np.abs(section_diff) + 1e-16)
    vmin = min(float(gram_image.min()), float(section_image.min()))
    vmax = max(float(gram_image.max()), float(section_image.max()))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), constrained_layout=True)
    image = None
    for ax, values, label in zip(
        axes[:2],
        (gram_image, section_image),
        ("Gram", "sections"),
        strict=True,
    ):
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes[:2], location="right", shrink=0.88, label="log10 abs diff")
    axes[2].bar(
        ["Gram", "sections"],
        [float(np.max(np.abs(gram_diff))), float(np.max(np.abs(section_diff)))],
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("max abs difference")
    axes[2].grid(True, which="both", axis="y", alpha=0.3)
    fig.suptitle(f"{case}: DiffusionHeatSections KeOps minus dense")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case}_keops_dense_verify.png", dpi=170)
    plt.close(fig)


_computed: dict[str, tuple[dict[str, float | str], np.ndarray, np.ndarray]] = {}

if __name__ == "__main__" and ifrun:
    all_rows: list[dict[str, float | str]] = []
    for case_name in CASES:
        _computed[case_name] = compare_case(case_name)
        all_rows.append(_computed[case_name][0])
        print(f"done {case_name}", flush=True)
    write_rows(all_rows)
    print(f"Wrote {CSV_PATH}")

if __name__ == "__main__" and ifplt:
    if not _computed:
        _computed = {case_name: compare_case(case_name) for case_name in CASES}
    for case_name, (_row, gram_diff, section_diff) in _computed.items():
        plot_case(case_name, gram_diff, section_diff)
        print(f"Wrote {OUT / f'{case_name}_keops_dense_verify.png'}")
