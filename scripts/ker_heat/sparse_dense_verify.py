"""Visual sparse-vs-dense checks for DyMAD diffusion kernels."""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import cast

# ruff: noqa: E402

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "dymad_matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import torch
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dymad.modules import KernelScDM, KernelSparseScDM  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "sparse_dense_verify"
CSV_PATH = OUT / "sparse_dense_errors.csv"

EPSILON = 0.01
STEPS = 2
REF_COUNT = 192
LOC_COUNT = 96
SEED = 2026070101
TOLERANCES = (1e-3, 1e-5, 1e-7, 1e-9, 1e-11)
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


def dense_kernel(ref: np.ndarray) -> KernelScDM:
    kernel = KernelScDM(in_dim=2, eps_init=EPSILON, t_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(torch.as_tensor(ref, dtype=torch.float64))
    return kernel


def sparse_kernel(ref: np.ndarray, tol: float) -> KernelSparseScDM:
    kernel = KernelSparseScDM(
        in_dim=2, eps_init=EPSILON, t_init=1.0, dtype=torch.float64, kernel_tol=tol
    )
    kernel.set_reference_data(torch.as_tensor(ref, dtype=torch.float64))
    return kernel


def gram(kernel: KernelScDM, ref: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(ref, dtype=torch.float64)
    return kernel(tensor, tensor).detach().cpu().numpy()


def sections(kernel: KernelScDM, locations: np.ndarray, sources: np.ndarray) -> np.ndarray:
    values = cast(
        torch.Tensor,
        kernel.heat_kernel(
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
    tolerances: tuple[float, ...] = TOLERANCES,
    n_ref: int = REF_COUNT,
    n_loc: int = LOC_COUNT,
) -> tuple[list[dict[str, float | str]], list[np.ndarray], list[np.ndarray]]:
    ref, loc, src = case_points(case, n_ref, n_loc)
    dense = dense_kernel(ref)
    gram_dense = gram(dense, ref)
    section_dense = sections(dense, loc, src)
    rows, gram_diffs, section_diffs = [], [], []
    for tol in tolerances:
        sparse = sparse_kernel(ref, tol)
        gram_diff = gram(sparse, ref) - gram_dense
        section_diff = sections(sparse, loc, src) - section_dense
        rows.append(
            {
                "case": case,
                "kernel_tol": tol,
                "gram_max_abs": float(np.max(np.abs(gram_diff))),
                "gram_rel_fro": rel_fro(gram_diff, gram_dense),
                "section_max_abs": float(np.max(np.abs(section_diff))),
                "section_rel_fro": rel_fro(section_diff, section_dense),
            }
        )
        gram_diffs.append(gram_diff)
        section_diffs.append(section_diff)
    return rows, gram_diffs, section_diffs


def write_rows(rows: list[dict[str, float | str]], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "kernel_tol",
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
    rows: list[dict[str, float | str]],
    gram_diffs: list[np.ndarray],
    section_diffs: list[np.ndarray],
    output_dir: Path = OUT,
) -> None:
    tols = np.asarray([float(row["kernel_tol"]) for row in rows])
    image_values = [
        np.log10(np.abs(diff) + 1e-16) for diffs in (gram_diffs, section_diffs) for diff in diffs
    ]
    vmin, vmax = (
        min(float(values.min()) for values in image_values),
        max(float(values.max()) for values in image_values),
    )
    fig = plt.figure(figsize=(3.1 * len(tols), 7.6), constrained_layout=True)
    grid = fig.add_gridspec(3, len(tols))
    image_axes = []
    image = None
    for j, tol in enumerate(tols):
        for i, (diffs, label) in enumerate(((gram_diffs, "Gram"), (section_diffs, "sections"))):
            ax = fig.add_subplot(grid[i, j])
            image = ax.imshow(
                np.log10(np.abs(diffs[j]) + 1e-16),
                aspect="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{label}, tol={tol:g}")
            ax.set_xticks([])
            ax.set_yticks([])
            image_axes.append(ax)
    if image is not None:
        fig.colorbar(image, ax=image_axes, location="right", shrink=0.92, label="log10 abs diff")
    ax = fig.add_subplot(grid[2, :])
    ax.loglog(tols, [float(row["gram_max_abs"]) for row in rows], marker="o", label="Gram max abs")
    ax.loglog(
        tols,
        [float(row["section_max_abs"]) for row in rows],
        marker="s",
        label="section max abs",
    )
    ax.invert_xaxis()
    ax.set_xlabel("kernel_tol")
    ax.set_ylabel("dense-sparse difference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.suptitle(f"{case}: KernelSparseScDM minus KernelScDM")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case}_sparse_dense_verify.png", dpi=170)
    plt.close(fig)


_computed: dict[str, tuple[list[dict[str, float | str]], list[np.ndarray], list[np.ndarray]]] = {}

if ifrun:
    all_rows: list[dict[str, float | str]] = []
    for case_name in CASES:
        _computed[case_name] = compare_case(case_name)
        all_rows.extend(_computed[case_name][0])
        print(f"done {case_name}", flush=True)
    write_rows(all_rows)
    print(f"Wrote {CSV_PATH}")

if ifplt:
    if not _computed:
        _computed = {case_name: compare_case(case_name) for case_name in CASES}
    for case_name, result in _computed.items():
        plot_case(case_name, *result)
        print(f"Wrote {OUT / f'{case_name}_sparse_dense_verify.png'}")
