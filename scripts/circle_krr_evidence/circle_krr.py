"""Generate the nine figures for the ambient-circle KRR study."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_legendre

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dymad.kernel_analysis import KernelEigenbasis
from dymad.modules import KernelScalarValued, make_kernel, make_krr
from dymad.tuning import ParameterSpec, TuningSpec, tune

# fmt: off
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "runs" / "ambient_circle_centered_composite_l2"
METHODS = ("dm_krr", "rbf_krr")
COLORS = {"dm_krr": "#0072B2", "rbf_krr": "#D55E00"}
FIGURES = {"circle_labels.png", "circle_semi_lb.png", "circle_semi_mode.png", "circle_semi_rbf.png", "circle_full_lb.png", "circle_angles.png", "semicircle_endpoints.png", "circle_krr.png", "fullcircle_lb_endpoints.png"}
S_GRID = np.asarray([0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .985, .99, .995, .996, .997, .998, .999, .99925, .9995, .99975, .9999, 1.0])

@dataclass
class Geometry:
    name: str
    q: np.ndarray
    qw: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    l2: np.ndarray
    l2w: np.ndarray
    spaces_q: dict[str, np.ndarray]
    spaces_l2: dict[str, np.ndarray]
    y_train: np.ndarray
    y_valid: np.ndarray
    y_l2: np.ndarray
    kinds: list[str]
    @property
    def x_train(self) -> np.ndarray:
        return ambient(self.train)

@dataclass(frozen=True)
class Fit:
    method: str
    bandwidth: float
    ridge: float
    validation_gram: np.ndarray

@dataclass(frozen=True)
class Selection:
    method: str
    kind: str
    index: int
    s: float | None
    coefficient: np.ndarray
    fit: Fit

@dataclass(frozen=True)
class Error:
    selection: Selection
    total: float
    in_class: float
    leakage: float

def ambient(q: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(q), np.sin(q)))

def orth(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(np.sqrt(weights)[:, None] * values, mode="reduced")
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
    return q / np.sqrt(weights)[:, None]

def canonical(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    for j in range(values.shape[1]):
        pivot = int(np.argmax(np.abs(values[:, j])))
        values[:, j] *= 1.0 if values[pivot, j] >= 0 else -1.0
    return values

def lb(name: str, q: np.ndarray) -> np.ndarray:
    freq = np.asarray((1, 3, 5, 7) if name == "semicircle" else (1, 2, 3, 4))
    if name == "semicircle":
        return np.sqrt(2) * np.sin(q[:, None] * freq) * -np.sin(np.pi * freq / 2)
    return np.column_stack((np.sqrt(2) * np.cos(q[:, None] * freq), np.sqrt(2) * np.sin(q[:, None] * freq)))

def chord_kernel(q: np.ndarray, ref: np.ndarray, ell: float = .2) -> np.ndarray:
    delta = ambient(q)[:, None] - ambient(ref)[None, :]
    return np.exp(-np.sum(delta * delta, axis=2) / (2 * ell * ell))

def rbf_label_space(q: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    operator = np.sqrt(w)[:, None] * chord_kernel(q, q) * np.sqrt(w)[None, :]
    half = len(q) // 2
    even = np.zeros((len(q), half))
    even[np.arange(half), np.arange(half)] = 2 ** -.5
    even[-np.arange(half) - 1, np.arange(half)] = 2 ** -.5
    eig, vec = np.linalg.eigh(even.T @ operator @ even)
    order = np.argsort(eig)[::-1][:4]
    return canonical((even @ vec[:, order]) / np.sqrt(w)[:, None]), eig[order]

def composite_rule(n_train: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = roots_legendre(order)
    edges = np.linspace(-np.pi / 2, np.pi / 2, n_train)
    mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w / np.pi).ravel()

def build_geometry(name: str, quick: bool) -> Geometry:
    raw = json.loads((HERE / "label_coefficients.json").read_text())
    n_train = (48 if quick else 1024) if name == "semicircle" else (9 if quick else 13)
    qn = 128 if quick else 512
    if name == "semicircle":
        x, w = leggauss(qn)
        q, qw = np.pi * x / 2, w / 2
        train = np.linspace(-np.pi / 2, np.pi / 2, n_train)
        valid = (train[1:] + train[:-1]) / 2
        l2, l2w = composite_rule(n_train, 4 if quick else 16)
        rbf_q, eig = rbf_label_space(q, qw)
        raw_coeff = qw[:, None] * rbf_q / eig
        rbf = lambda z: chord_kernel(z, q) @ raw_coeff
        spaces_q = {"lb": lb(name, q), "rbf": rbf_q}
        spaces_l2 = {"lb": lb(name, l2), "rbf": rbf(l2)}
        base = lambda z: np.column_stack((lb(name, z), rbf(z)))
        n = 2 if quick else 12
        left, right = np.asarray(raw["semicircle_lb"][:n]).T, np.asarray(raw["semicircle_rbf"][:n]).T
        coefficients = np.block([[left, np.zeros_like(left)], [np.zeros_like(right), right]])
        kinds = ["lb"] * n + ["rbf"] * n
    else:
        q = 2 * np.pi * np.arange(qn) / qn
        qw = np.full(qn, 1 / qn)
        train = 2 * np.pi * np.arange(n_train) / n_train
        valid = 2 * np.pi * (np.arange(n_train) + .5) / n_train
        l2n = 256 if quick else 8192
        l2, l2w = 2 * np.pi * np.arange(l2n) / l2n, np.full(l2n, 1 / l2n)
        spaces_q, spaces_l2 = {"lb": lb(name, q)}, {"lb": lb(name, l2)}
        base = lambda z: lb(name, z)
        n = 2 if quick else 12
        coefficients = np.asarray(raw["full_circle_lb"][:n]).T
        kinds = ["lb"] * n
    y_l2 = base(l2) @ coefficients
    coefficients /= np.sqrt(np.sum(l2w[:, None] * y_l2**2, axis=0))[None, :]
    return Geometry(name, q, qw, train, valid, l2, l2w, spaces_q, spaces_l2, base(train) @ coefficients, base(valid) @ coefficients, base(l2) @ coefficients, kinds)

def kernel_config(method: str, bandwidth: float) -> dict[str, Any]:
    if method == "dm_krr":
        return {"type": "sc_dm", "input_dim": 2, "eps_init": bandwidth}
    return {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": bandwidth}

def scalar_kernel(method: str, bandwidth: float) -> KernelScalarValued:
    config = kernel_config(method, bandwidth)
    kind = cast(str, config.pop("type"))
    return cast(KernelScalarValued, make_kernel(k_type=kind, dtype=torch.float64, **config))

class FitCache:
    def __init__(self, geometry: Geometry):
        self.g, self.data, self.lock = geometry, {}, threading.Lock()
    def get(self, method: str, params: dict[str, Any]) -> Fit:
        key = method, float(params["bandwidth"]), float(params["ridge"])
        with self.lock:
            if key in self.data:
                return self.data[key]
        model = make_krr(type="share", kernel=kernel_config(method, key[1]), dtype=torch.float64, ridge_init=key[2], jitter=0.0)
        model.set_train_data(self.g.x_train, self.g.y_train)
        model.fit()
        with torch.no_grad():
            pred = model(torch.as_tensor(ambient(self.g.valid), dtype=torch.float64)).cpu().numpy()
            bandwidth = float(model.kernel.eps if method == "dm_krr" else model.kernel.ell)
            ridge = float(model.ridge)
        residual = pred - self.g.y_valid
        gram = residual.T @ residual / len(residual)
        fit = Fit(method, bandwidth, ridge, (gram + gram.T) / 2)
        with self.lock:
            return self.data.setdefault(key, fit)

def tuning_spec(quick: bool, fixed: bool = False) -> TuningSpec:
    if fixed:
        count = 3 if quick else 65
        params = (ParameterSpec("bandwidth", values=(.2,)), ParameterSpec("ridge", values=tuple(np.logspace(-16, -8, count))))
        return TuningSpec(params, "error", "minimize", (1, count), "grid", seed=0)
    params = (ParameterSpec("bandwidth", bounds=(1e-4, 1e2), scale="log"), ParameterSpec("ridge", bounds=(1e-16, 1e1), scale="log"))
    size = 2 if quick else 9
    return TuningSpec(params, "error", "minimize", (size, size), "grid", "multi_start_nelder_mead" if not quick else None, 0 if quick else 64, seed=0)

def tune_label(g: Geometry, cache: FitCache, method: str, coefficient: np.ndarray, kind: str, index: int, s: float | None, quick: bool, workers: int) -> Selection:
    def run(spec: TuningSpec) -> tuple[dict[str, Any], float]:
        def evaluator(params: dict[str, Any]) -> dict[str, float]:
            fit = cache.get(method, params)
            return {"error": math.sqrt(max(float(coefficient @ fit.validation_gram @ coefficient), 0))}
        result = tune(spec, evaluator, max_workers=workers)
        if not result.selected_params:
            raise RuntimeError(f"tuning failed for {g.name} {method}")
        return result.selected_params, result.selected_metric
    params, metric = run(tuning_spec(quick))
    if method == "rbf_krr":
        fixed, fixed_metric = run(tuning_spec(quick, True))
        if fixed_metric < metric:
            params = fixed
    return Selection(method, kind, index, s, coefficient, cache.get(method, params))

def select_models(g: Geometry, quick: bool, workers: int) -> list[Selection]:
    cache, selected = FitCache(g), []
    for j, kind in enumerate(g.kinds):
        coefficient = np.eye(len(g.kinds))[j]
        for method in METHODS:
            selected.append(tune_label(g, cache, method, coefficient, kind, j, None, quick, workers))
    if g.name == "semicircle":
        n = len(g.kinds) // 2
        for j in range(n):
            for s in S_GRID:
                coefficient = np.zeros(2 * n)
                coefficient[j], coefficient[j + n] = np.cos(np.pi * s / 2), np.sin(np.pi * s / 2)
                label = g.y_l2 @ coefficient
                coefficient /= math.sqrt(float(np.sum(g.l2w * label * label)))
                for method in METHODS:
                    selected.append(tune_label(g, cache, method, coefficient, "family", j, float(s), quick, workers))
    return selected

def realize(g: Geometry, selected: list[Selection]) -> list[Error]:
    groups: dict[tuple[str, float, float], list[Selection]] = {}
    for item in selected:
        groups.setdefault((item.method, item.fit.bandwidth, item.fit.ridge), []).append(item)
    predictions: dict[int, np.ndarray] = {}
    for (method, bandwidth, ridge), items in groups.items():
        model: Any = make_krr(type="share", kernel=kernel_config(method, bandwidth), dtype=torch.float64, ridge_init=ridge, jitter=0.0)
        model.set_train_data(g.x_train, g.y_train)
        model.fit()
        dual = model._alphas.detach().cpu().numpy().astype(np.longdouble) @ np.column_stack([item.coefficient for item in items]).astype(np.longdouble)
        points = torch.as_tensor(ambient(g.l2), dtype=torch.float64)
        with torch.no_grad():
            prediction = np.concatenate([model.kernel(points[i:i + 4096], model._Xref).cpu().numpy().astype(np.longdouble) @ dual for i in range(0, len(points), 4096)])
        predictions.update({id(item): prediction[:, j] for j, item in enumerate(items)})
    errors, n = [], len(g.kinds) // 2
    for item in selected:
        fitted, label = predictions[id(item)], g.y_l2 @ item.coefficient
        space = g.y_l2[:, [item.index, item.index + n]] if item.kind == "family" else g.spaces_l2[item.kind]
        q = orth(space, g.l2w).astype(np.longdouble)
        projection = q @ (q.T @ (g.l2w.astype(np.longdouble) * fitted))
        b = math.sqrt(float(np.sum(g.l2w * (label - projection) ** 2)))
        leakage = math.sqrt(float(np.sum(g.l2w * (fitted - projection) ** 2)))
        errors.append(Error(item, math.hypot(b, leakage), b, leakage))
    return errors

def reflection_basis(size: int) -> np.ndarray:
    half = size // 2
    basis = np.zeros((size, half + size % 2))
    basis[np.arange(half), np.arange(half)] = 2 ** -.5
    basis[-np.arange(half) - 1, np.arange(half)] = 2 ** -.5
    if size % 2:
        basis[half, -1] = 1
    return basis

def kernel_modes(g: Geometry, selection: Selection) -> np.ndarray:
    count, even, skip = (4, True, 0) if selection.kind == "rbf" else (8, False, int(g.name == "full_circle"))
    kernel = scalar_kernel(selection.method, selection.fit.bandwidth)
    train, evaluation = torch.as_tensor(g.x_train, dtype=torch.float64), torch.as_tensor(ambient(g.q), dtype=torch.float64)
    if not even:
        basis = KernelEigenbasis(kernel, count, skip=skip, eigenvalue_rtol=1e-14).solve(train)
        with torch.no_grad():
            return canonical(orth(basis.transform(evaluation).cpu().numpy(), g.qw))
    kernel.require_fixed_parameters()
    kernel.set_reference_data(train)
    with torch.no_grad():
        matrix, cross = kernel.materialize(train, train).cpu().numpy(), kernel.materialize(evaluation, train).cpu().numpy()
    symmetry = reflection_basis(len(train))
    eig, vec = np.linalg.eigh(symmetry.T @ ((matrix + matrix.T) / 2) @ symmetry)
    order = np.argsort(eig)[::-1][:count]
    return canonical(orth(cross @ symmetry @ vec[:, order] / eig[order], g.qw))

def angles_and_representatives(g: Geometry, endpoints: list[Selection]) -> tuple[dict[str, np.ndarray], dict[str, tuple[Selection, np.ndarray]]]:
    records: dict[str, list[tuple[float, Selection, np.ndarray]]] = {}
    for item in endpoints:
        modes = kernel_modes(g, item)
        left, right = orth(modes, g.qw), orth(g.spaces_q[item.kind], g.qw)
        singular = np.linalg.svd(left.T @ (g.qw[:, None] * right), compute_uv=False)
        angle = float(np.max(np.degrees(np.arccos(np.clip(singular, -1, 1)))))
        records.setdefault(f"{item.method}-{item.kind}", []).append((angle, item, modes))
    distributions, representatives = {}, {}
    for key, values in records.items():
        distribution = np.asarray([value[0] for value in values])
        chosen = min(values, key=lambda value: (abs(value[0] - np.median(distribution)), value[1].index))
        distributions[key] = distribution
        representatives[key] = (chosen[1], chosen[2])
    return distributions, representatives

def align(source: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source, labels = orth(source, weights), orth(labels, weights)
    left, _, right = np.linalg.svd(source.T @ (weights[:, None] * labels), full_matrices=False)
    return labels, source @ left @ right

def angle_text(value: float) -> str:
    exponent = math.floor(math.log10(abs(value))) if value else 0
    return rf"{value / 10**exponent:.3g}\times 10^{{{exponent}}}" if exponent <= -3 else f"{value:.3g}"

def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=180)
    plt.close(fig)

def plot_labels(out: Path, semi: Geometry, full: Geometry) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), constrained_layout=True)
    n = len(semi.kinds) // 2
    panels = [(semi, slice(0, n), "(a) Semicircle: 12 LB functions"), (semi, slice(n, None), "(b) Semicircle: 12 RBF functions"), (full, slice(None), "(d) Full circle: 12 common-LB functions")]
    for axis, (g, selected, title) in zip((axes[0, 0], axes[0, 1], axes[1, 1]), panels):
        for j, values in enumerate(g.y_l2[:, selected].T, 1):
            axis.plot(g.l2, values, lw=.95, label=str(j))
        axis.set(title=title, xlabel=r"$t$" if g.name == "semicircle" else r"$\theta$", ylabel="labels")
        axis.grid(alpha=.25)
        axis.legend(title="function", fontsize="xx-small", ncol=3)
    j, s, axis = 6 % n, .8, axes[1, 0]
    mixed = np.cos(np.pi * s / 2) * semi.y_l2[:, j] + np.sin(np.pi * s / 2) * semi.y_l2[:, n + j]
    for values, label, color in ((semi.y_l2[:, j], r"$u_7$ (LB)", None), (semi.y_l2[:, n + j], r"$v_7$ (RBF)", None), (mixed, r"$f_7(s=0.8)$", ".12")):
        axis.plot(semi.l2, values, lw=1.5 if color else 1.1, color=color, label=label)
    axis.set(title="(c) One representative semicircle family", xlabel=r"$t$", ylabel="labels")
    axis.grid(alpha=.25)
    axis.legend(fontsize="x-small")
    save(fig, out, "circle_labels.png")

def plot_mode_row(out: Path, name: str, g: Geometry, kind: str, reps: dict[str, tuple[Selection, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 9.1 / 3), constrained_layout=True)
    for letter, method, axis in zip("ab", METHODS, axes):
        _, raw_modes = reps[f"{method}-{kind}"]
        singular = np.linalg.svd(orth(raw_modes, g.qw).T @ (g.qw[:, None] * orth(g.spaces_q[kind], g.qw)), compute_uv=False)
        angle = np.max(np.degrees(np.arccos(np.clip(singular, -1, 1))))
        labels, modes = align(raw_modes, g.spaces_q[kind], g.qw)
        count, spacing = labels.shape[1], 2.6 * max(np.max(abs(labels)), np.max(abs(modes)))
        offset = spacing * np.arange(count)
        for j in range(count):
            axis.plot(g.q, labels[:, j] + offset[j], "--", color=".28", lw=.9, label="label-space mode" if j == 0 else None)
            axis.plot(g.q, modes[:, j] + offset[j], color=COLORS[method], lw=1.1, label="aligned kernel mode" if j == 0 else None)
        title = f"({letter}) {'Semicircle' if g.name == 'semicircle' else 'Full-circle'} {kind.upper()}: {'DM' if method == 'dm_krr' else 'RBF'}"
        axis.set(title=title + rf" ($\theta_{{\max}}={{{angle_text(float(angle))}}}^\circ$)", xlabel=r"$t$" if g.name == "semicircle" else r"$\theta$", ylabel="mode (vertically offset)", yticks=offset, yticklabels=range(1, count + 1))
        axis.grid(alpha=.18)
        if letter == "a":
            axis.legend(fontsize="x-small")
    save(fig, out, name)

def plot_inventory(out: Path, semi: Geometry, reps: dict[str, tuple[Selection, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), constrained_layout=True)
    panels = [("LB label space", semi.spaces_q["lb"], "#009E73"), ("RBF label space", semi.spaces_q["rbf"], "#CC79A7"), ("DM KRR modes", reps["dm_krr-lb"][1], COLORS["dm_krr"]), ("RBF KRR modes", reps["rbf_krr-lb"][1], COLORS["rbf_krr"])]
    for letter, axis, (title, values, color) in zip("abcd", axes.flat, panels):
        spacing, count = 2.6 * np.max(abs(values)), values.shape[1]
        offset = spacing * np.arange(count)
        for j in range(count):
            axis.plot(semi.q, values[:, j] + offset[j], color=color, lw=1.05)
        axis.set(title=f"({letter}) {title}: {count} modes", xlabel=r"$t$", ylabel="mode (vertically offset)", yticks=offset, yticklabels=range(1, count + 1))
        axis.grid(alpha=.18)
    save(fig, out, "circle_semi_mode.png")

def plot_angles(out: Path, semi: dict[str, np.ndarray], full: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 9.1 / 3), constrained_layout=True)
    for axis, values, title in ((axes[0], semi, "(a) Semicircle: 12 tuned models per pair"), (axes[1], full, "(b) Full circle: 12 tuned models per pair")):
        keys = list(values)
        for position, key in enumerate(keys, 1):
            color = COLORS[key.split("-")[0]]
            violin = axis.violinplot([values[key]], positions=[position], showmedians=True)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.28)
            for part in ("cbars", "cmins", "cmaxes", "cmedians"):
                violin[part].set_color(color)
            axis.scatter(position + np.linspace(-.09, .09, len(values[key])), values[key], s=10, color=color, edgecolors="white", lw=.25)
        axis.set(title=title, xticks=range(1, len(keys) + 1), xticklabels=[key.replace("_krr", "").upper() for key in keys], ylabel="largest principal angle (degrees)")
        axis.grid(axis="y", alpha=.25)
    axes[1].axhline(0, color=".35", ls="--", lw=1.2, label=r"$0^\circ$ reference")
    axes[1].legend(fontsize="x-small")
    save(fig, out, "circle_angles.png")

def endpoint_errors(errors: list[Error], kind: str, method: str) -> list[Error]:
    return sorted([e for e in errors if e.selection.s is None and e.selection.kind == kind and e.selection.method == method], key=lambda e: e.selection.index)

def plot_errors(out: Path, semi: list[Error], full: list[Error]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    for col, kind in enumerate(("lb", "rbf")):
        for method, marker in zip(METHODS, ("o", "s")):
            rows = endpoint_errors(semi, kind, method)
            x = np.arange(1, len(rows) + 1)
            axes[0, col].semilogy(x, [e.total for e in rows], color=COLORS[method], marker=marker, ms=3, label=f"{method.split('_')[0].upper()} total $E$")
            axes[1, col].plot(x, [(e.leakage / e.total) ** 2 for e in rows], color=COLORS[method], marker=marker, ms=3, label=rf"{method.split('_')[0].upper()} $L^2/E^2$")
        axes[0, col].set(title=f"({'ab'[col]}) {kind.upper()} endpoints: total errors", ylabel=r"$L^2$ error norm")
        axes[1, col].set(title=f"({'cd'[col]}) {kind.upper()} endpoints: leakage shares", xlabel="endpoint", ylabel=r"leakage energy $L^2/E^2$", ylim=(-.04, 1.04))
        for axis in axes[:, col]:
            axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize="x-small")
    axes[1, 0].legend(fontsize="x-small")
    save(fig, out, "semicircle_endpoints.png")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 9.1 / 3), constrained_layout=True)
    families = sorted({e.selection.index for e in semi if e.selection.kind == "family"})
    finite = S_GRID < 1
    x = np.empty_like(S_GRID)
    x[finite] = -np.log10(1 - S_GRID[finite])
    x[~finite] = x[finite][-1] + .28
    for method, marker in zip(METHODS, ("o", "s")):
        matrix = np.asarray([[e.total for e in semi if e.selection.kind == "family" and e.selection.method == method and e.selection.index == j] for j in families])
        median, bounds = np.median(matrix, axis=0), np.quantile(matrix, (.1, .9), axis=0)
        axes[0].fill_between(x, *bounds, color=COLORS[method], alpha=.18)
        axes[0].semilogy(x, median, color=COLORS[method], marker=marker, ms=3, label=method.split("_")[0].upper())
    axes[0].set(title="(a) Semicircle: median family errors", xlabel=r"family parameter $s$", ylabel=r"$L^2$ error norm")
    axes[0].set_xticks([0, 1, 2, 3, x[-1]], ["0", ".9", ".99", ".999", "1"])
    axes[0].legend(fontsize="x-small")
    axes[0].grid(alpha=.25)
    for method, shift in zip(METHODS, (-.19, .19)):
        rows = endpoint_errors(full, "lb", method)
        axes[1].bar(np.arange(1, len(rows) + 1) + shift, [e.total for e in rows], .38, color=COLORS[method], edgecolor="black", lw=.8, label=f"{method.split('_')[0].upper()} total $E$")
    axes[1].set(title="(b) Full circle: total errors", xlabel="label index", ylabel=r"$L^2$ error norm", yscale="log")
    axes[1].legend(fontsize="x-small")
    axes[1].grid(axis="y", alpha=.25)
    save(fig, out, "circle_krr.png")
    fig, axis = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
    for method, marker in zip(METHODS, ("o", "s")):
        rows = endpoint_errors(full, "lb", method)
        x, label = np.arange(1, len(rows) + 1), method.split("_")[0].upper()
        axis.semilogy(x, [e.in_class for e in rows], "--", color=COLORS[method], marker=marker, label=f"{label} in-class $B$")
        axis.semilogy(x, [e.leakage for e in rows], ":", color=COLORS[method], marker=marker, label=f"{label} leakage $L$")
        axis.semilogy(x, [e.total for e in rows], color=COLORS[method], marker=marker, lw=2.2, label=f"{label} total $E$")
    axis.set(xlabel="index", ylabel=r"$L^2$ error norm")
    axis.tick_params(labelsize=16)
    axis.legend(fontsize=16, ncol=2)
    axis.grid(alpha=.25)
    save(fig, out, "fullcircle_lb_endpoints.png")

def run(output: Path, quick: bool = False, workers: int = 4) -> None:
    output.mkdir(parents=True, exist_ok=True)
    semi, full = build_geometry("semicircle", quick), build_geometry("full_circle", quick)
    semi_selected, full_selected = select_models(semi, quick, workers), select_models(full, quick, workers)
    semi_angles, semi_reps = angles_and_representatives(semi, [s for s in semi_selected if s.s is None])
    full_angles, full_reps = angles_and_representatives(full, [s for s in full_selected if s.s is None])
    semi_errors, full_errors = realize(semi, semi_selected), realize(full, full_selected)
    plot_labels(output, semi, full)
    plot_mode_row(output, "circle_semi_lb.png", semi, "lb", semi_reps)
    plot_inventory(output, semi, semi_reps)
    plot_mode_row(output, "circle_semi_rbf.png", semi, "rbf", semi_reps)
    plot_mode_row(output, "circle_full_lb.png", full, "lb", full_reps)
    plot_angles(output, semi_angles, full_angles)
    plot_errors(output, semi_errors, full_errors)
    print(f"Wrote {len(FIGURES)} figures to {output}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="small smoke-test protocol")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.quick, max(1, min(args.workers, 4)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
