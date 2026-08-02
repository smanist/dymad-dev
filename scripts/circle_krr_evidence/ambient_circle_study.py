"""Ambient-Euclidean KRR evidence study on a semicircle and a full circle.

The coordinate ``theta`` is used only to sample the curves and evaluate the
continuum ``L2`` inner product.  Both KRR methods receive points in ``R^2``:
``(cos(theta), sin(theta))``.  The module uses independently tuned endpoint
and paired-family fits while treating the full-circle Fourier degeneracy as a
subspace question rather than constructing a meaningless LB--RBF family.
"""

from __future__ import annotations

import csv
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from ambient_circle_plots import write_report_figures
from ambient_circle_report import write_ambient_circle_report
from numpy.polynomial.legendre import leggauss

from dymad.kernel_analysis import KernelEigenbasis
from dymad.modules import KernelScalarValued, make_kernel, make_krr
from dymad.tuning import ParameterSpec, TuningEvaluation, TuningSpec, tune

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "ambient_circle_full13_resolved"
DEFAULT_REPORT_PATH = Path("output/pdf/circle_krr_ambient/circle_krr_ambient_study_note.pdf")
METHODS = ("dm_krr", "rbf_krr")
METHOD_LABELS = {"dm_krr": "DM", "rbf_krr": "RBF"}
RBF_WINDOWS = (
    (1, 3, 5),
    (1, 3, 5, 7),
    (1, 3, 5, 7, 9),
    (3, 5, 7),
    (3, 5, 7, 9),
    (3, 5, 7, 9, 11),
    (5, 7, 9),
    (5, 7, 9, 11),
    (5, 7, 9, 11, 13),
    (7, 9, 11),
    (7, 9, 11, 13),
    (7, 9, 11, 13, 15),
)
KRR_MODE_COUNT = 8


@dataclass(frozen=True)
class AmbientCircleStudyConfig:
    """Reproducible semicircle/full-circle evidence protocol."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    report_path: Path = DEFAULT_REPORT_PATH
    semi_n_train: int = 1024
    # The rotationally invariant full-circle problem converges much faster than
    # the boundary-bearing semicircle.  Thirteen points keeps the comparison
    # resolved (DM is just below 1e-10) instead of at the interpolation floor.
    full_n_train: int = 13
    semi_n_valid: int = 1023
    full_n_valid: int = 13
    test_count: int = 65_536
    quadrature_order: int = 512
    endpoint_count: int = 12
    rbf_target_lengthscale: float = 0.2
    semi_lb_frequencies: tuple[int, ...] = (1, 3, 5, 7)
    full_lb_frequencies: tuple[int, ...] = (1, 2, 3, 4)
    bandwidth_bounds: tuple[float, float] = (1.0e-4, 1.0e2)
    ridge_bounds: tuple[float, float] = (1.0e-16, 1.0e1)
    fixed_rbf_ridge_bounds: tuple[float, float] = (1.0e-16, 1.0e-8)
    initial_grid_size: int = 9
    refinement_budget: int = 64
    fixed_rbf_ridge_count: int = 65
    seed: int = 0
    endpoint_seed: int = 20_260_729
    max_workers: int = 4
    plot: bool = True
    write_report: bool = True


@dataclass(frozen=True)
class GeometryModes:
    """Quadrature modes and Nyström data for one normalized geometry."""

    name: Literal["semi_circle", "full_circle"]
    nodes: np.ndarray
    weights: np.ndarray
    lb_frequencies: tuple[int, ...]
    lb_values: np.ndarray
    raw_rbf_values: np.ndarray
    rbf_values: np.ndarray
    raw_to_rbf: np.ndarray
    rbf_eigenvalues: np.ndarray
    rbf_to_lb_subspace_gap: float
    reflection_cross_gram_error: float | None

    @property
    def period(self) -> float:
        return math.pi if self.name == "semi_circle" else 2.0 * math.pi


@dataclass(frozen=True)
class TargetData:
    """Shared points and test-normalized endpoint functions for one geometry."""

    modes: GeometryModes
    theta_train: np.ndarray
    theta_valid: np.ndarray
    theta_test: np.ndarray
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    endpoint_train: np.ndarray
    endpoint_valid: np.ndarray
    endpoint_test: np.ndarray
    lb_test_basis: np.ndarray
    rbf_test_basis: np.ndarray | None
    endpoint_kinds: tuple[str, ...]
    endpoint_labels: tuple[str, ...]

    @property
    def endpoint_dimension(self) -> int:
        return self.endpoint_test.shape[1]


@dataclass(frozen=True)
class CandidateRecord:
    """A cached shared-output KRR fit evaluated on all validation endpoints."""

    method: str
    bandwidth: float
    ridge: float
    validation_q: np.ndarray


@dataclass(frozen=True)
class TuningSelection:
    """One independently validation-selected endpoint or family map."""

    method: str
    geometry: str
    target_kind: str
    target_label: str
    coefficient: np.ndarray
    candidate: CandidateRecord
    validation_error: float
    evaluations: tuple[TuningEvaluation, ...]
    family_index: int | None = None
    family_s: float | None = None


def _ambient(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _weighted_gram(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    gram = values.T @ (weights[:, None] * values)
    return 0.5 * (gram + gram.T)


def _weighted_orthonormalize(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.sqrt(weights)[:, None] * values, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q *= signs[None, :]
    r = signs[:, None] * r
    return q / np.sqrt(weights)[:, None], np.linalg.inv(r)


def _subspace_gap(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    q_left, _ = _weighted_orthonormalize(left, weights)
    q_right, _ = _weighted_orthonormalize(right, weights)
    singular_values = np.linalg.svd(q_left.T @ (weights[:, None] * q_right), compute_uv=False)
    return float(math.sqrt(max(0.0, 1.0 - float(np.min(singular_values)) ** 2)))


def _principal_angles_degrees(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return the principal angles between two weighted function subspaces."""

    left_q, _ = _weighted_orthonormalize(left, weights)
    right_q, _ = _weighted_orthonormalize(right, weights)
    singular_values = np.linalg.svd(
        left_q.T @ (weights[:, None] * right_q), compute_uv=False
    )
    return np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0)))


def _quadrature(
    name: Literal["semi_circle", "full_circle"], order: int
) -> tuple[np.ndarray, np.ndarray]:
    if name == "semi_circle":
        raw_nodes, raw_weights = leggauss(order)
        # Weights integrate the probability measure dtheta / pi.
        return 0.5 * math.pi * (raw_nodes + 1.0), 0.5 * raw_weights
    nodes = 2.0 * math.pi * np.arange(order, dtype=float) / order
    return nodes, np.full(order, 1.0 / order)


def _chord_kernel(theta: np.ndarray, reference: np.ndarray, lengthscale: float) -> np.ndarray:
    difference = _ambient(theta)[:, None, :] - _ambient(reference)[None, :, :]
    return np.exp(-np.sum(difference * difference, axis=2) / (2.0 * lengthscale**2))


def _lb_values(
    name: Literal["semi_circle", "full_circle"], theta: np.ndarray, frequencies: tuple[int, ...]
) -> np.ndarray:
    frequency = np.asarray(frequencies, dtype=float)
    if name == "semi_circle":
        return math.sqrt(2.0) * np.cos(theta[:, None] * frequency[None, :])
    return np.column_stack(
        (
            math.sqrt(2.0) * np.cos(theta[:, None] * frequency[None, :]),
            math.sqrt(2.0) * np.sin(theta[:, None] * frequency[None, :]),
        )
    )


def _raw_rbf_modes(
    name: Literal["semi_circle", "full_circle"],
    nodes: np.ndarray,
    weights: np.ndarray,
    lengthscale: float,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonconstant continuum Gaussian modes with the correct symmetries."""

    kernel = _chord_kernel(nodes, nodes, lengthscale)
    operator = np.sqrt(weights)[:, None] * kernel * np.sqrt(weights)[None, :]
    if name == "semi_circle":
        if len(nodes) % 2:
            raise ValueError("semicircle Gauss--Legendre rule must have even order")
        half = len(nodes) // 2
        even = np.zeros((len(nodes), half))
        indices = np.arange(half)
        even[indices, indices] = 1.0 / math.sqrt(2.0)
        even[-indices - 1, indices] = 1.0 / math.sqrt(2.0)
        eigenvalues, vectors = np.linalg.eigh(even.T @ operator @ even)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        vectors = even @ vectors[:, order]
        start = 0
    else:
        eigenvalues, vectors = np.linalg.eigh(operator)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        vectors = vectors[:, order]
        # Drop the constant eigenfunction; retain four Fourier pairs.
        start = 1
    stop = start + dimension
    return vectors[:, start:stop] / np.sqrt(weights)[:, None], eigenvalues[start:stop]


def build_geometry_modes(
    name: Literal["semi_circle", "full_circle"], config: AmbientCircleStudyConfig
) -> GeometryModes:
    """Build normalized LB and ambient-RBF continuum mode spaces."""

    nodes, weights = _quadrature(name, config.quadrature_order)
    frequencies = (
        config.semi_lb_frequencies if name == "semi_circle" else config.full_lb_frequencies
    )
    lb_values = _lb_values(name, nodes, frequencies)
    rbf_dimension = 2 * len(frequencies) if name == "semi_circle" else lb_values.shape[1]
    raw_rbf_values, eigenvalues = _raw_rbf_modes(
        name, nodes, weights, config.rbf_target_lengthscale, rbf_dimension
    )
    rbf_values, raw_to_rbf = _weighted_orthonormalize(raw_rbf_values, weights)
    cross_error: float | None = None
    if name == "semi_circle":
        cross_error = float(np.max(np.abs(lb_values.T @ (weights[:, None] * rbf_values))))
    return GeometryModes(
        name=name,
        nodes=nodes,
        weights=weights,
        lb_frequencies=frequencies,
        lb_values=lb_values,
        raw_rbf_values=raw_rbf_values,
        rbf_values=rbf_values,
        raw_to_rbf=raw_to_rbf,
        rbf_eigenvalues=eigenvalues,
        rbf_to_lb_subspace_gap=_subspace_gap(rbf_values, lb_values, weights),
        reflection_cross_gram_error=cross_error,
    )


def _evaluate_rbf(theta: np.ndarray, modes: GeometryModes, lengthscale: float) -> np.ndarray:
    kernel = _chord_kernel(np.asarray(theta, dtype=float), modes.nodes, lengthscale)
    raw_coefficients = (
        modes.weights[:, None] * modes.raw_rbf_values / modes.rbf_eigenvalues[None, :]
    )
    return (kernel @ raw_coefficients) @ modes.raw_to_rbf


def _rbf_boundary_derivatives(
    modes: GeometryModes, lengthscale: float, maximum_order: int
) -> np.ndarray:
    """Automatic-differentiation boundary jets for the semicircle RBF modes."""

    theta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    reference = torch.as_tensor(_ambient(modes.nodes), dtype=torch.float64)
    coefficients = torch.as_tensor(
        modes.weights[:, None]
        * modes.raw_rbf_values
        / modes.rbf_eigenvalues[None, :]
        @ modes.raw_to_rbf,
        dtype=torch.float64,
    )
    result = np.empty((maximum_order + 1, modes.rbf_values.shape[1]))
    for column in range(modes.rbf_values.shape[1]):
        point = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        affinity = torch.exp(
            -torch.sum((point[:, None, :] - reference[None, :, :]) ** 2, dim=2)
            / (2.0 * lengthscale**2)
        )
        current = (affinity @ coefficients[:, column]).reshape(())
        result[0, column] = float(current.detach())
        for order in range(1, maximum_order + 1):
            current = torch.autograd.grad(current, theta, create_graph=order < maximum_order)[
                0
            ].reshape(())
            result[order, column] = float(current.detach())
    return result


def _semicircle_rbf_directions(
    modes: GeometryModes, endpoint_count: int, lengthscale: float
) -> np.ndarray:
    if endpoint_count > len(RBF_WINDOWS):
        raise ValueError(f"endpoint_count must not exceed {len(RBF_WINDOWS)}")
    maximum_order = max(2 * len(window) - 3 for window in RBF_WINDOWS[:endpoint_count])
    derivatives = _rbf_boundary_derivatives(modes, lengthscale, maximum_order)
    directions = np.zeros((modes.rbf_values.shape[1], endpoint_count))
    for endpoint, window in enumerate(RBF_WINDOWS[:endpoint_count]):
        indices = (np.asarray(window, dtype=int) - 1) // 2
        odd_orders = np.arange(1, 2 * len(indices) - 1, 2)
        constraints = derivatives[odd_orders[:, None], indices]
        _left, _singular_values, right = np.linalg.svd(constraints, full_matrices=True)
        direction = right[-1]
        if derivatives[0, indices] @ direction < 0.0:
            direction *= -1.0
        directions[indices, endpoint] = direction
    norms = np.sqrt(np.diag(_weighted_gram(modes.rbf_values @ directions, modes.weights)))
    return directions / norms[None, :]


def _unit_directions(dimension: int, count: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(size=(dimension, count))
    return values / np.linalg.norm(values, axis=0, keepdims=True)


def _endpoint_values(
    theta: np.ndarray, modes: GeometryModes, config: AmbientCircleStudyConfig
) -> np.ndarray:
    lb = _lb_values(modes.name, np.asarray(theta, dtype=float), modes.lb_frequencies)
    if modes.name == "full_circle":
        return lb
    return np.column_stack((lb, _evaluate_rbf(theta, modes, config.rbf_target_lengthscale)))


def _design_thetas(
    name: str, n_train: int, n_valid: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    period = math.pi if name == "semi_circle" else 2.0 * math.pi
    if name == "semi_circle":
        train = np.linspace(0.0, period, n_train)
        valid = 0.5 * (train[:-1] + train[1:])
        if len(valid) != n_valid:
            raise ValueError("semicircle validation count must equal n_train - 1")
    else:
        train = period * np.arange(n_train, dtype=float) / n_train
        valid = period * (np.arange(n_valid, dtype=float) + 0.5) / n_valid
    test = np.random.default_rng(seed).uniform(0.0, period, size=n_test)
    return train, valid, test


def build_target_data(
    name: Literal["semi_circle", "full_circle"], config: AmbientCircleStudyConfig
) -> TargetData:
    """Create shared midpoint/periodic samples and test-normalized endpoint targets."""

    modes = build_geometry_modes(name, config)
    n_valid = config.semi_n_valid if name == "semi_circle" else config.full_n_valid
    n_train = config.semi_n_train if name == "semi_circle" else config.full_n_train
    train, valid, test = _design_thetas(
        name,
        n_train,
        n_valid,
        config.test_count,
        config.seed + (0 if name == "semi_circle" else 1),
    )
    rng = np.random.default_rng(config.endpoint_seed + (0 if name == "semi_circle" else 1))
    lb_directions = _unit_directions(modes.lb_values.shape[1], config.endpoint_count, rng)
    if name == "semi_circle":
        rbf_directions = _semicircle_rbf_directions(
            modes, config.endpoint_count, config.rbf_target_lengthscale
        )
        coefficients = np.zeros(
            (modes.lb_values.shape[1] + modes.rbf_values.shape[1], 2 * config.endpoint_count)
        )
        coefficients[: modes.lb_values.shape[1], : config.endpoint_count] = lb_directions
        coefficients[modes.lb_values.shape[1] :, config.endpoint_count :] = rbf_directions
        kinds = ("lb",) * config.endpoint_count + ("rbf",) * config.endpoint_count
        labels = tuple(
            [f"LB endpoint {index + 1}" for index in range(config.endpoint_count)]
            + [f"RBF endpoint {index + 1}" for index in range(config.endpoint_count)]
        )
    else:
        coefficients = lb_directions
        kinds = ("lb",) * config.endpoint_count
        labels = tuple(f"LB endpoint {index + 1}" for index in range(config.endpoint_count))
    test_values = _endpoint_values(test, modes, config) @ coefficients
    test_norms = np.sqrt(np.mean(test_values * test_values, axis=0))
    coefficients /= test_norms[None, :]
    return TargetData(
        modes=modes,
        theta_train=train,
        theta_valid=valid,
        theta_test=test,
        x_train=_ambient(train),
        x_valid=_ambient(valid),
        x_test=_ambient(test),
        endpoint_train=_endpoint_values(train, modes, config) @ coefficients,
        endpoint_valid=_endpoint_values(valid, modes, config) @ coefficients,
        endpoint_test=_endpoint_values(test, modes, config) @ coefficients,
        lb_test_basis=_lb_values(name, test, modes.lb_frequencies),
        rbf_test_basis=(
            _evaluate_rbf(test, modes, config.rbf_target_lengthscale)
            if name == "semi_circle"
            else None
        ),
        endpoint_kinds=kinds,
        endpoint_labels=labels,
    )


def _kernel_config(method: str, bandwidth: float) -> dict[str, Any]:
    if method == "dm_krr":
        return {"type": "sc_dm", "input_dim": 2, "eps_init": bandwidth}
    if method == "rbf_krr":
        return {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": bandwidth}
    raise ValueError(f"unknown KRR method {method!r}")


def _scalar_kernel(method: str, bandwidth: float) -> KernelScalarValued:
    """Build the study kernel through DyMAD's public kernel factory."""

    config = _kernel_config(method, bandwidth)
    kernel_type = str(config.pop("type"))
    return cast(
        KernelScalarValued,
        make_kernel(k_type=kernel_type, dtype=torch.float64, **config),
    )


def _fit_candidate(method: str, targets: TargetData, params: dict[str, Any]) -> CandidateRecord:
    model = make_krr(
        type="share",
        kernel=_kernel_config(method, float(params["bandwidth_init"])),
        dtype=torch.float64,
        ridge_init=float(params["ridge_init"]),
        jitter=0.0,
    )
    model.set_train_data(targets.x_train, targets.endpoint_train)
    model.fit()
    with torch.no_grad():
        prediction = model(torch.as_tensor(targets.x_valid, dtype=torch.float64)).cpu().numpy()
        bandwidth = float(
            model.kernel.eps.detach().cpu()
            if method == "dm_krr"
            else model.kernel.ell.detach().cpu()
        )
        ridge = float(model.ridge.detach().cpu())
    residual = prediction - targets.endpoint_valid
    return CandidateRecord(
        method=method,
        bandwidth=bandwidth,
        ridge=ridge,
        validation_q=_weighted_gram(residual, np.full(len(residual), 1.0 / len(residual))),
    )


class CandidateCache:
    """Thread-safe candidate cache shared across all targets of one geometry."""

    def __init__(self, targets: TargetData) -> None:
        self.targets = targets
        self._records: dict[tuple[str, float, float], CandidateRecord] = {}
        self._failures: dict[tuple[str, float, float], str] = {}
        self._lock = threading.Lock()

    def get(self, method: str, params: dict[str, Any]) -> CandidateRecord:
        key = (method, float(params["bandwidth_init"]), float(params["ridge_init"]))
        with self._lock:
            cached = self._records.get(key)
            failure = self._failures.get(key)
        if cached is not None:
            return cached
        if failure is not None:
            raise RuntimeError(failure)
        try:
            fitted = _fit_candidate(method, self.targets, params)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._failures[key] = message
            raise RuntimeError(message) from exc
        with self._lock:
            return self._records.setdefault(key, fitted)


def _search_spec(config: AmbientCircleStudyConfig) -> TuningSpec:
    return TuningSpec(
        parameters=(
            ParameterSpec("bandwidth_init", bounds=config.bandwidth_bounds, scale="log"),
            ParameterSpec("ridge_init", bounds=config.ridge_bounds, scale="log"),
        ),
        metric_name="validation_normalized_rmse",
        goal="minimize",
        initial_budget=(config.initial_grid_size, config.initial_grid_size),
        initial_strategy="grid",
        refinement_strategy="multi_start_nelder_mead" if config.refinement_budget else None,
        refinement_budget=config.refinement_budget,
        seed=config.seed,
        metadata={"study": "ambient_circle_unit_style", "kernel_distance": "ambient_chord"},
    )


def _fixed_rbf_spec(config: AmbientCircleStudyConfig) -> TuningSpec:
    return TuningSpec(
        parameters=(
            ParameterSpec("bandwidth_init", values=(config.rbf_target_lengthscale,)),
            ParameterSpec(
                "ridge_init",
                values=tuple(
                    np.logspace(
                        math.log10(config.fixed_rbf_ridge_bounds[0]),
                        math.log10(config.fixed_rbf_ridge_bounds[1]),
                        config.fixed_rbf_ridge_count,
                    )
                ),
            ),
        ),
        metric_name="validation_normalized_rmse",
        goal="minimize",
        initial_budget=(1, config.fixed_rbf_ridge_count),
        initial_strategy="grid",
        refinement_strategy=None,
        seed=config.seed,
        metadata={
            "study": "ambient_circle_fixed_rbf",
            "ambient_lengthscale": config.rbf_target_lengthscale,
        },
    )


def _metric(record: CandidateRecord, coefficient: np.ndarray) -> float:
    return math.sqrt(max(float(coefficient @ record.validation_q @ coefficient), 0.0))


def _tune_target(
    method: str,
    targets: TargetData,
    cache: CandidateCache,
    config: AmbientCircleStudyConfig,
    coefficient: np.ndarray,
    target_kind: str,
    target_label: str,
    family_index: int | None = None,
    family_s: float | None = None,
) -> TuningSelection:
    def run(spec: TuningSpec) -> tuple[dict[str, Any], float, tuple[TuningEvaluation, ...]]:
        def evaluator(params: dict[str, Any]) -> dict[str, float]:
            record = cache.get(method, params)
            return {
                "validation_normalized_rmse": _metric(record, coefficient),
                "realized_bandwidth": record.bandwidth,
                "realized_ridge": record.ridge,
            }

        result = tune(spec, evaluator, max_workers=min(config.max_workers, 4))
        if not result.selected_params:
            raise RuntimeError(f"all candidates failed for {method}: {target_label}")
        return result.selected_params, result.selected_metric, tuple(result.evaluations)

    selected_params, selected_metric, evaluations = run(_search_spec(config))
    if method == "rbf_krr":
        fixed_params, fixed_metric, fixed_evaluations = run(_fixed_rbf_spec(config))
        if fixed_metric < selected_metric:
            selected_params, selected_metric = fixed_params, fixed_metric
        evaluations = evaluations + fixed_evaluations
    candidate = cache.get(method, selected_params)
    return TuningSelection(
        method=method,
        geometry=targets.modes.name,
        target_kind=target_kind,
        target_label=target_label,
        coefficient=coefficient.copy(),
        candidate=candidate,
        validation_error=_metric(candidate, coefficient),
        evaluations=evaluations,
        family_index=family_index,
        family_s=family_s,
    )


def _endpoint_selections(
    targets: TargetData, cache: CandidateCache, config: AmbientCircleStudyConfig
) -> list[TuningSelection]:
    selections: list[TuningSelection] = []
    for endpoint, (kind, label) in enumerate(
        zip(targets.endpoint_kinds, targets.endpoint_labels, strict=True)
    ):
        coefficient = np.zeros(targets.endpoint_dimension)
        coefficient[endpoint] = 1.0
        for method in METHODS:
            selections.append(
                _tune_target(method, targets, cache, config, coefficient, kind, label)
            )
    return selections


def _family_s_values() -> np.ndarray:
    return np.concatenate((np.linspace(0.0, 0.95, 31), np.linspace(0.975, 1.0, 7)))


def _family_coefficient(targets: TargetData, index: int, s: float) -> np.ndarray:
    coefficient = np.zeros(targets.endpoint_dimension)
    coefficient[index] = math.cos(math.pi * s / 2.0)
    coefficient[index + targets.endpoint_dimension // 2] = math.sin(math.pi * s / 2.0)
    target = targets.endpoint_test @ coefficient
    return coefficient / math.sqrt(float(np.mean(target * target)))


def _family_selections(
    targets: TargetData, cache: CandidateCache, config: AmbientCircleStudyConfig
) -> list[TuningSelection]:
    selections: list[TuningSelection] = []
    for index in range(config.endpoint_count):
        for s in _family_s_values():
            coefficient = _family_coefficient(targets, index, float(s))
            for method in METHODS:
                selections.append(
                    _tune_target(
                        method,
                        targets,
                        cache,
                        config,
                        coefficient,
                        "family",
                        f"family {index + 1}",
                        family_index=index + 1,
                        family_s=float(s),
                    )
                )
    return selections


def _selected_predictions(
    selections: list[TuningSelection], targets: TargetData
) -> dict[int, np.ndarray]:
    """Refit each selected map once and evaluate its scalar target on all test points."""

    grouped: dict[tuple[str, float, float], list[TuningSelection]] = {}
    for selection in selections:
        key = (selection.method, selection.candidate.bandwidth, selection.candidate.ridge)
        grouped.setdefault(key, []).append(selection)
    predictions: dict[int, np.ndarray] = {}
    test_tensor = torch.as_tensor(targets.x_test, dtype=torch.float64)
    for (method, bandwidth, ridge), group in grouped.items():
        model = make_krr(
            type="share",
            kernel=_kernel_config(method, bandwidth),
            dtype=torch.float64,
            ridge_init=ridge,
            jitter=0.0,
        )
        model.set_train_data(targets.x_train, targets.endpoint_train)
        model.fit()
        with torch.no_grad():
            endpoint_prediction = model(test_tensor).cpu().numpy()
        for selection in group:
            predictions[id(selection)] = endpoint_prediction @ selection.coefficient
    return predictions


def _typical_krr_parameters(
    selections: list[TuningSelection], method: str, target_kind: str
) -> tuple[float, float]:
    """Return exponentiated coordinate-wise log-medians of selected parameters."""

    matching = [
        selection
        for selection in selections
        if selection.method == method and selection.target_kind == target_kind
    ]
    if not matching:
        raise ValueError(f"no {method} selections for target kind {target_kind!r}")
    bandwidth = float(
        math.exp(np.median([math.log(selection.candidate.bandwidth) for selection in matching]))
    )
    ridge = float(
        math.exp(np.median([math.log(selection.candidate.ridge) for selection in matching]))
    )
    return bandwidth, ridge


def _reflection_basis(size: int, parity: Literal["even", "odd"]) -> np.ndarray:
    """Return an orthonormal basis for reversal-even or reversal-odd vectors."""

    pair_count = size // 2
    column_count = pair_count + int(parity == "even" and size % 2 == 1)
    basis = np.zeros((size, column_count))
    scale = 1.0 / math.sqrt(2.0)
    for column in range(pair_count):
        basis[column, column] = scale
        basis[size - column - 1, column] = scale if parity == "even" else -scale
    if parity == "even" and size % 2 == 1:
        basis[pair_count, -1] = 1.0
    return basis


def _kernel_nystrom_modes(
    targets: TargetData,
    method: str,
    bandwidth: float,
    *,
    mode_count: int,
    parity: Literal["even", "odd"] | None = None,
    drop_constant: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-sample kernel eigenmodes extended to the quadrature rule.

    These are eigenvectors of the selected kernel Gram matrix, extended by
    the Nyström formula.  They are deliberately not singular vectors of the
    regularized interpolation map: the latter are amplification directions
    for arbitrary training data and are not DM/RBF kernel modes.
    """

    kernel = _scalar_kernel(method, bandwidth)
    train = torch.as_tensor(targets.x_train, dtype=torch.float64)
    quadrature = torch.as_tensor(_ambient(targets.modes.nodes), dtype=torch.float64)

    if parity is None:
        skip = int(drop_constant)
        requested_count = min(mode_count, len(train) - skip)
        while requested_count > 0:
            try:
                basis = KernelEigenbasis(
                    kernel,
                    requested_count,
                    skip=skip,
                    eigenvalue_rtol=1.0e-14,
                ).solve(train)
                break
            except ValueError as exc:
                if "too close to zero" not in str(exc):
                    raise
                requested_count -= 1
        else:
            raise RuntimeError("kernel has no numerically resolved modes")
        with torch.no_grad():
            nystrom = basis.transform(quadrature).cpu().numpy()
        eigenvalues = basis.eigenvalues.detach().cpu().numpy()
        relative = eigenvalues / max(float(eigenvalues[0]), np.finfo(float).tiny)
        nystrom, _ = _weighted_orthonormalize(nystrom, targets.modes.weights)
        return nystrom, relative

    if targets.modes.name != "semi_circle":
        raise ValueError("reflection sectors are only defined for the semicircle")
    kernel.require_fixed_parameters()
    kernel.set_reference_data(train)
    with torch.no_grad():
        train_kernel = kernel.materialize(train, train)
        evaluation_kernel = kernel.materialize(quadrature, train)
    matrix = train_kernel.detach().cpu().numpy()
    matrix = 0.5 * (matrix + matrix.T)
    evaluation = evaluation_kernel.detach().cpu().numpy()

    symmetry_basis = _reflection_basis(len(train), parity)
    matrix = symmetry_basis.T @ matrix @ symmetry_basis

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvectors = symmetry_basis @ eigenvectors
    relative = np.maximum(eigenvalues, 0.0) / max(float(eigenvalues[0]), np.finfo(float).tiny)
    resolved_rank = int(np.sum(relative >= 1.0e-14))
    start = int(drop_constant)
    available = max(0, resolved_rank - start)
    actual_count = min(mode_count, available)
    if actual_count == 0:
        raise RuntimeError("kernel has no numerically resolved modes in the requested sector")
    stop = start + actual_count
    selected_values = eigenvalues[start:stop]
    nystrom = evaluation @ eigenvectors[:, start:stop] / selected_values[None, :]
    nystrom, _ = _weighted_orthonormalize(nystrom, targets.modes.weights)
    return nystrom, relative[start:stop]


def _align_modes_for_plot(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a source basis to its closest target basis for visual comparison."""

    source_q, _ = _weighted_orthonormalize(source, weights)
    target_q, _ = _weighted_orthonormalize(target, weights)
    left, _singular_values, right = np.linalg.svd(
        source_q.T @ (weights[:, None] * target_q), full_matrices=False
    )
    aligned_source = source_q @ left @ right
    return target_q, aligned_source


def _krr_mode_angles(
    targets: TargetData, endpoint_selections: list[TuningSelection]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compare target spaces with genuine finite-sample kernel eigenmodes."""

    mode_spaces: dict[str, np.ndarray] = {}
    settings: dict[str, dict[str, Any]] = {}
    target_kinds = ("lb", "rbf") if targets.modes.name == "semi_circle" else ("lb",)
    for target_kind in target_kinds:
        for method in METHODS:
            bandwidth, ridge = _typical_krr_parameters(endpoint_selections, method, target_kind)
            parity = "even" if target_kind == "rbf" else None
            drop_constant = targets.modes.name == "full_circle"
            modes, relative_eigenvalues = _kernel_nystrom_modes(
                targets,
                method,
                bandwidth,
                mode_count=KRR_MODE_COUNT,
                parity=parity,
                drop_constant=drop_constant,
            )
            label = f"{METHOD_LABELS[method]}-{target_kind.upper()}"
            mode_spaces[label] = modes
            settings[label] = {
                "method": METHOD_LABELS[method],
                "target_space": target_kind.upper(),
                "bandwidth": bandwidth,
                "typical_ridge_not_used_for_modes": ridge,
                "mode_rule": (
                    "first eight reflection-even kernel modes"
                    if parity == "even"
                    else "first eight nonconstant kernel modes"
                    if drop_constant
                    else "first eight kernel modes"
                ),
                "leading_mode_count": modes.shape[1],
                "relative_eigenvalues": [float(value) for value in relative_eigenvalues],
            }

    comparisons: list[tuple[str, np.ndarray, np.ndarray]] = []
    if targets.modes.name == "semi_circle":
        comparisons.extend(
            (
                ("LB versus RBF target spaces", targets.modes.lb_values, targets.modes.rbf_values),
                (
                    "DM (LB-tuned), first 8 versus LB",
                    mode_spaces["DM-LB"],
                    targets.modes.lb_values,
                ),
                (
                    "RBF (LB-tuned), first 8 versus LB",
                    mode_spaces["RBF-LB"],
                    targets.modes.lb_values,
                ),
                (
                    "DM (RBF-tuned), first 8 even versus RBF",
                    mode_spaces["DM-RBF"],
                    targets.modes.rbf_values,
                ),
                (
                    "RBF (RBF-tuned), first 8 even versus RBF",
                    mode_spaces["RBF-RBF"],
                    targets.modes.rbf_values,
                ),
            )
        )
    else:
        comparisons.extend(
            (
                ("DM, first 8 nonconstant versus LB", mode_spaces["DM-LB"], targets.modes.lb_values),
                (
                    "RBF, first 8 nonconstant versus LB",
                    mode_spaces["RBF-LB"],
                    targets.modes.lb_values,
                ),
            )
        )

    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for index, (label, left, right) in enumerate(comparisons, start=1):
        angles = _principal_angles_degrees(left, right, targets.modes.weights)
        result = {
            "comparison": label,
            "common_dimension": len(angles),
            "principal_angles_degrees": [float(angle) for angle in angles],
            "minimum_angle_degrees": float(np.min(angles)),
            "median_angle_degrees": float(np.median(angles)),
            "maximum_angle_degrees": float(np.max(angles)),
        }
        key = f"comparison_{index}"
        results[key] = result
        rows.append(
            {
                "geometry": targets.modes.name,
                "comparison": label,
                **result,
                "principal_angles_degrees": ";".join(f"{angle:.10g}" for angle in angles),
            }
        )
    plot_panels: list[dict[str, Any]] = []
    if targets.modes.name == "semi_circle":
        panel_specs = (
            ("Semicircle LB: DM", "DM-LB", targets.modes.lb_values),
            ("Semicircle LB: RBF", "RBF-LB", targets.modes.lb_values),
            ("Semicircle RBF: DM", "DM-RBF", targets.modes.rbf_values),
            ("Semicircle RBF: RBF", "RBF-RBF", targets.modes.rbf_values),
        )
    else:
        panel_specs = (
            ("Full-circle LB: DM", "DM-LB", targets.modes.lb_values),
            ("Full-circle LB: RBF", "RBF-LB", targets.modes.lb_values),
        )
    for title, label, target_space in panel_specs:
        target_basis, aligned_modes = _align_modes_for_plot(
            mode_spaces[label], target_space, targets.modes.weights
        )
        plot_panels.append(
            {
                "title": title,
                "target": target_basis,
                "kernel": aligned_modes,
                "maximum_angle_degrees": float(
                    np.max(
                        _principal_angles_degrees(
                            mode_spaces[label], target_space, targets.modes.weights
                        )
                    )
                ),
            }
        )
    return (
        {"settings": settings, "comparisons": results},
        rows,
        {
            "geometry": targets.modes.name,
            "theta": targets.modes.nodes,
            "panels": plot_panels,
        },
    )


def _orthonormal_test_basis(values: np.ndarray) -> np.ndarray:
    left, singular_values, _right = np.linalg.svd(values, full_matrices=False)
    tolerance = max(values.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    return left[:, :rank]


def _decomposition_row(
    selection: TuningSelection, targets: TargetData, prediction: np.ndarray
) -> dict[str, Any]:
    target = targets.endpoint_test @ selection.coefficient
    if selection.target_kind == "lb":
        basis = targets.lb_test_basis
    elif selection.target_kind == "rbf":
        if targets.rbf_test_basis is None:
            raise ValueError("full-circle targets do not have a separate RBF endpoint space")
        basis = targets.rbf_test_basis
    elif selection.target_kind == "family":
        assert selection.family_index is not None
        index = selection.family_index - 1
        basis = targets.endpoint_test[:, [index, index + targets.endpoint_dimension // 2]]
    else:
        raise ValueError(f"unknown target kind {selection.target_kind!r}")
    q = _orthonormal_test_basis(basis)
    projected_target = q @ (q.T @ target)
    target_projection_defect = float(np.sqrt(np.mean((target - projected_target) ** 2)))
    target = projected_target
    in_class_prediction = q @ (q.T @ prediction)
    in_class_error = target - in_class_prediction
    leakage = prediction - in_class_prediction
    residual = target - prediction
    direct_total_squared = float(np.mean(residual * residual))
    in_class_squared = float(np.mean(in_class_error * in_class_error))
    leakage_squared = float(np.mean(leakage * leakage))
    # With f represented by its test-space projection, this is the orthogonal
    # definition of E^2.  It avoids a cancellation-limited direct residual
    # only when a selected fit itself is already at double-precision scale.
    total_squared = in_class_squared + leakage_squared
    total = math.sqrt(max(total_squared, 0.0))
    leakage_norm = math.sqrt(max(leakage_squared, 0.0))
    leakage_to_total = leakage_norm / max(total, np.finfo(float).tiny)
    if leakage_to_total > 1.0 + 64.0 * np.finfo(float).eps:
        raise RuntimeError("orthogonal decomposition invariant L <= E was violated")
    return {
        "geometry": selection.geometry,
        "target_kind": selection.target_kind,
        "target_label": selection.target_label,
        "family_index": selection.family_index,
        "s": selection.family_s,
        "method": selection.method,
        "bandwidth": selection.candidate.bandwidth,
        "effective_lengthscale": math.sqrt(2.0 * selection.candidate.bandwidth)
        if selection.method == "dm_krr"
        else selection.candidate.bandwidth,
        "ridge": selection.candidate.ridge,
        "validation_error": selection.validation_error,
        "population_error": total,
        "direct_population_error": math.sqrt(max(direct_total_squared, 0.0)),
        "in_class_error": math.sqrt(max(in_class_squared, 0.0)),
        "leakage": leakage_norm,
        "in_class_squared": in_class_squared,
        "leakage_squared": leakage_squared,
        "leakage_share": leakage_squared / max(total_squared, np.finfo(float).tiny),
        "leakage_to_total_ratio": leakage_to_total,
        "decomposition_defect": abs(direct_total_squared - total_squared),
        "target_projection_defect": target_projection_defect,
        "evaluation_count": len(selection.evaluations),
        "unique_evaluation_count": sum(not item.cache_hit for item in selection.evaluations),
        "failed_evaluation_count": sum(item.status != "ok" for item in selection.evaluations),
    }


def _selection_rows(selections: list[TuningSelection]) -> list[dict[str, Any]]:
    return [
        {
            "geometry": selection.geometry,
            "target_kind": selection.target_kind,
            "target_label": selection.target_label,
            "family_index": selection.family_index,
            "s": selection.family_s,
            "method": selection.method,
            "bandwidth": selection.candidate.bandwidth,
            "ridge": selection.candidate.ridge,
            "validation_error": selection.validation_error,
            "evaluation_count": len(selection.evaluations),
            "unique_evaluation_count": sum(not item.cache_hit for item in selection.evaluations),
            "failed_evaluation_count": sum(item.status != "ok" for item in selection.evaluations),
        }
        for selection in selections
    ]


def _tuning_rows(selections: list[TuningSelection]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selection in selections:
        for evaluation in selection.evaluations:
            rows.append(
                {
                    "geometry": selection.geometry,
                    "target_kind": selection.target_kind,
                    "target_label": selection.target_label,
                    "family_index": selection.family_index,
                    "s": selection.family_s,
                    "method": selection.method,
                    "phase": evaluation.phase,
                    "status": evaluation.status,
                    "cache_hit": evaluation.cache_hit,
                    "bandwidth": evaluation.params.get("bandwidth_init"),
                    "ridge": evaluation.params.get("ridge_init"),
                    "validation_error": evaluation.extra_metrics.get("validation_normalized_rmse"),
                }
            )
    return rows


def _paired_endpoint_summary(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target_kind"] == kind]
    labels = sorted({str(row["target_label"]) for row in selected})
    ratios: list[float] = []
    dm_shares: list[float] = []
    rbf_shares: list[float] = []
    dm_errors: list[float] = []
    rbf_errors: list[float] = []
    dm_in_class: list[float] = []
    rbf_in_class: list[float] = []
    dm_leakage: list[float] = []
    rbf_leakage: list[float] = []
    rbf_wins = 0
    for label in labels:
        dm = next(
            row for row in selected if row["target_label"] == label and row["method"] == "dm_krr"
        )
        rbf = next(
            row for row in selected if row["target_label"] == label and row["method"] == "rbf_krr"
        )
        ratios.append(
            float(rbf["population_error"])
            / max(float(dm["population_error"]), np.finfo(float).tiny)
        )
        dm_shares.append(float(dm["leakage_share"]))
        rbf_shares.append(float(rbf["leakage_share"]))
        dm_errors.append(float(dm["population_error"]))
        rbf_errors.append(float(rbf["population_error"]))
        dm_in_class.append(float(dm["in_class_error"]))
        rbf_in_class.append(float(rbf["in_class_error"]))
        dm_leakage.append(float(dm["leakage"]))
        rbf_leakage.append(float(rbf["leakage"]))
        rbf_wins += int(float(rbf["population_error"]) < float(dm["population_error"]))
    return {
        "count": len(labels),
        "dm_win_count": len(labels) - rbf_wins,
        "rbf_win_count": rbf_wins,
        "median_rbf_to_dm_error_ratio": float(np.median(ratios)),
        "dm_median_leakage_share": float(np.median(dm_shares)),
        "rbf_median_leakage_share": float(np.median(rbf_shares)),
        "dm_median_population_error": float(np.median(dm_errors)),
        "rbf_median_population_error": float(np.median(rbf_errors)),
        "dm_median_in_class_error": float(np.median(dm_in_class)),
        "rbf_median_in_class_error": float(np.median(rbf_in_class)),
        "dm_median_leakage": float(np.median(dm_leakage)),
        "rbf_median_leakage": float(np.median(rbf_leakage)),
    }


def _crossings(s: np.ndarray, difference: np.ndarray) -> list[float]:
    roots: list[float] = []
    for left in range(len(s) - 1):
        a, b = difference[left], difference[left + 1]
        if a == 0.0:
            roots.append(float(s[left]))
        elif a * b < 0.0:
            roots.append(float(s[left] + (s[left + 1] - s[left]) * (-a) / (b - a)))
    if difference[-1] == 0.0:
        roots.append(float(s[-1]))
    return roots


def _safe_log_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_log = np.log(np.maximum(left, 1.0e-300))
    right_log = np.log(np.maximum(right, 1.0e-300))
    if np.std(left_log) == 0.0 or np.std(right_log) == 0.0:
        return 0.0
    return float(np.corrcoef(left_log, right_log)[0, 1])


def _family_summary(
    rows: list[dict[str, Any]], endpoint_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    for family_index in range(1, endpoint_count + 1):
        family = [row for row in rows if row["family_index"] == family_index]
        dm = sorted(
            (row for row in family if row["method"] == "dm_krr"), key=lambda row: float(row["s"])
        )
        rbf = sorted(
            (row for row in family if row["method"] == "rbf_krr"), key=lambda row: float(row["s"])
        )
        s = np.asarray([float(row["s"]) for row in dm])
        dm_error = np.asarray([float(row["population_error"]) for row in dm])
        rbf_error = np.asarray([float(row["population_error"]) for row in rbf])
        rbf_in = np.asarray([float(row["in_class_error"]) for row in rbf])
        rbf_leakage = np.asarray([float(row["leakage"]) for row in rbf])
        layer = s >= 0.975 - 1.0e-12
        floor = float(np.median(dm_error[layer]))
        exact = _crossings(s, rbf_error - dm_error)
        floor_roots = _crossings(s, rbf_error - floor)
        diagnostics.append(
            {
                "family_index": family_index,
                "dm_floor": floor,
                "exact_crossing_count": len(exact),
                "exact_crossing": exact[0] if len(exact) == 1 else None,
                "floor_crossing": floor_roots[0] if len(floor_roots) == 1 else None,
                "floor_crossing_shift": abs(exact[0] - floor_roots[0])
                if len(exact) == len(floor_roots) == 1
                else None,
                "dm_endpoint_layer_variation": float(
                    (np.max(dm_error[layer]) - np.min(dm_error[layer]))
                    / max(floor, np.finfo(float).tiny)
                ),
                "rbf_error_leakage_log_correlation": _safe_log_correlation(rbf_error, rbf_leakage),
                "leakage_collapse": float(
                    rbf_leakage[np.flatnonzero(layer)[0]]
                    / max(rbf_leakage[-1], np.finfo(float).tiny)
                ),
                "crossing_rbf_leakage_squared_share": (
                    float(rbf[int(np.argmin(np.abs(s - exact[0])))]["leakage_share"])
                    if len(exact) == 1
                    else None
                ),
                "endpoint_in_class_over_floor_squared": float(
                    rbf_in[-1] ** 2 / max(floor**2, np.finfo(float).tiny)
                ),
                "endpoint_leakage_over_floor_squared": float(
                    rbf_leakage[-1] ** 2 / max(floor**2, np.finfo(float).tiny)
                ),
            }
        )
    crossings = [row for row in diagnostics if row["exact_crossing"] is not None]
    # Use one representative family for the detailed plot without cherry
    # picking the strongest crossover: choose the crossing whose leakage
    # reduction is closest to the geometric-median reduction among crossings.
    if crossings:
        median_log_collapse = float(
            np.median([math.log(row["leakage_collapse"]) for row in crossings])
        )
        representative = min(
            crossings,
            key=lambda row: (
                abs(math.log(row["leakage_collapse"]) - median_log_collapse),
                float("inf")
                if row["floor_crossing_shift"] is None
                else row["floor_crossing_shift"],
                row["family_index"],
            ),
        )
    else:
        representative = diagnostics[0]
    return (
        {
            "count": endpoint_count,
            "crossing_count": len(crossings),
            "median_dm_endpoint_layer_variation": float(
                np.median([row["dm_endpoint_layer_variation"] for row in diagnostics])
            ),
            "median_rbf_error_leakage_log_correlation": float(
                np.nanmedian([row["rbf_error_leakage_log_correlation"] for row in diagnostics])
            ),
            "median_leakage_collapse": float(
                np.median([row["leakage_collapse"] for row in diagnostics])
            ),
            "maximum_floor_crossing_shift": float(
                np.nanmax(
                    [
                        row["floor_crossing_shift"]
                        for row in diagnostics
                        if row["floor_crossing_shift"] is not None
                    ]
                )
            )
            if any(row["floor_crossing_shift"] is not None for row in diagnostics)
            else None,
            "representative_family_index": int(representative["family_index"]),
            "representative_exact_crossing": representative["exact_crossing"],
            "representative_family_selection": "crossing family nearest the geometric-median leakage collapse",
        },
        diagnostics,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def run_ambient_circle_study(
    config: AmbientCircleStudyConfig = AmbientCircleStudyConfig(),
) -> dict[str, Any]:
    """Execute the full semicircle/full-circle ambient-kernel evidence protocol."""

    if config.endpoint_count > len(RBF_WINDOWS):
        raise ValueError(f"endpoint_count must not exceed {len(RBF_WINDOWS)}")
    if config.semi_n_valid != config.semi_n_train - 1:
        raise ValueError("semi_n_valid must equal n_train - 1 for midpoint validation")
    if min(config.semi_n_train, config.full_n_train) < 2 or config.test_count < 4:
        raise ValueError("sampling counts are too small")
    if not 0.0 < config.fixed_rbf_ridge_bounds[0] < config.fixed_rbf_ridge_bounds[1]:
        raise ValueError("fixed_rbf_ridge_bounds must be increasing and positive")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    semi = build_target_data("semi_circle", config)
    full = build_target_data("full_circle", config)
    semi_cache = CandidateCache(semi)
    full_cache = CandidateCache(full)
    semi_endpoints = _endpoint_selections(semi, semi_cache, config)
    semi_families = _family_selections(semi, semi_cache, config)
    full_endpoints = _endpoint_selections(full, full_cache, config)
    semi_krr_mode_angles, semi_krr_mode_angle_rows, semi_mode_plot = _krr_mode_angles(
        semi, semi_endpoints
    )
    full_krr_mode_angles, full_krr_mode_angle_rows, full_mode_plot = _krr_mode_angles(
        full, full_endpoints
    )

    semi_all = semi_endpoints + semi_families
    semi_predictions = _selected_predictions(semi_all, semi)
    full_predictions = _selected_predictions(full_endpoints, full)
    semi_decompositions = [
        _decomposition_row(selection, semi, semi_predictions[id(selection)])
        for selection in semi_all
    ]
    full_decompositions = [
        _decomposition_row(selection, full, full_predictions[id(selection)])
        for selection in full_endpoints
    ]
    semi_endpoint_rows = [row for row in semi_decompositions if row["target_kind"] in {"lb", "rbf"}]
    semi_family_rows = [row for row in semi_decompositions if row["target_kind"] == "family"]
    family_summary, family_diagnostics = _family_summary(semi_family_rows, config.endpoint_count)
    full_summary = _paired_endpoint_summary(full_decompositions, "lb")
    full_summary["median_population_error"] = float(
        np.median([row["population_error"] for row in full_decompositions])
    )
    full_summary["maximum_target_projection_defect"] = float(
        max(row["target_projection_defect"] for row in full_decompositions)
    )
    full_summary["precision_limited"] = bool(
        full_summary["median_population_error"] <= full_summary["maximum_target_projection_defect"]
    )
    for method, prefix in (("dm_krr", "dm"), ("rbf_krr", "rbf")):
        method_rows = [row for row in full_decompositions if row["method"] == method]
        full_summary[f"{prefix}_median_population_error"] = float(
            np.median([row["population_error"] for row in method_rows])
        )
        full_summary[f"{prefix}_median_in_class_error"] = float(
            np.median([row["in_class_error"] for row in method_rows])
        )
        full_summary[f"{prefix}_median_leakage"] = float(
            np.median([row["leakage"] for row in method_rows])
        )
        full_summary[f"{prefix}_ridge_near_lower_bound_count"] = int(
            sum(row["ridge"] <= 1.1 * config.ridge_bounds[0] for row in method_rows)
        )
    all_decompositions = semi_decompositions + full_decompositions
    summary: dict[str, Any] = {
        "title": "Ambient Euclidean DM versus RBF KRR on semicircle and full circle",
        "config": asdict(config),
        "protocol": {
            "kernel_distance": "ambient Euclidean chord distance in R^2",
            "test_count": config.test_count,
            "semi_circle": {
                "n_train": config.semi_n_train,
                "n_valid": config.semi_n_valid,
                "test_sampling": "seeded uniform random theta",
            },
            "full_circle": {
                "n_train": config.full_n_train,
                "n_valid": config.full_n_valid,
                "test_sampling": "seeded uniform random theta",
            },
            "tuning": {
                "shared": f"{config.initial_grid_size}x{config.initial_grid_size} log grid plus four-start Nelder--Mead",
                "rbf_fixed_sweep": {
                    "ambient_lengthscale": config.rbf_target_lengthscale,
                    "ridge_count": config.fixed_rbf_ridge_count,
                    "ridge_bounds": config.fixed_rbf_ridge_bounds,
                },
            },
        },
        "semicircle": {
            "reflection_cross_gram_error": semi.modes.reflection_cross_gram_error,
            "krr_mode_angles": semi_krr_mode_angles,
            "endpoints": {
                "lb": _paired_endpoint_summary(semi_endpoint_rows, "lb"),
                "rbf": _paired_endpoint_summary(semi_endpoint_rows, "rbf"),
            },
            "families": family_summary,
        },
        "full_circle": {
            "rbf_to_lb_subspace_gap": full.modes.rbf_to_lb_subspace_gap,
            "krr_mode_angles": full_krr_mode_angles,
            "endpoints": full_summary,
        },
        "audit": {
            "maximum_decomposition_defect": float(
                max(row["decomposition_defect"] for row in all_decompositions)
            ),
            "maximum_target_projection_defect": float(
                max(row["target_projection_defect"] for row in all_decompositions)
            ),
            "maximum_leakage_to_total_ratio": float(
                max(row["leakage_to_total_ratio"] for row in all_decompositions)
            ),
            "leakage_exceeds_total_count": int(
                sum(row["leakage_to_total_ratio"] > 1.0 for row in all_decompositions)
            ),
            "maximum_direct_vs_recomposed_error_gap": float(
                max(
                    abs(row["direct_population_error"] - row["population_error"])
                    for row in all_decompositions
                )
            ),
            "direct_residual_below_leakage_count": int(
                sum(
                    row["direct_population_error"] < row["leakage"]
                    for row in all_decompositions
                )
            ),
            "maximum_leakage_to_direct_residual_ratio": float(
                max(
                    row["leakage"]
                    / max(row["direct_population_error"], np.finfo(float).tiny)
                    for row in all_decompositions
                )
            ),
        },
    }
    _write_csv(output_dir / "selected_models.csv", _selection_rows(semi_all + full_endpoints))
    _write_csv(output_dir / "tuning_evaluations.csv", _tuning_rows(semi_all + full_endpoints))
    _write_csv(output_dir / "decompositions.csv", all_decompositions)
    _write_csv(output_dir / "semicircle_family_diagnostics.csv", family_diagnostics)
    _write_csv(
        output_dir / "krr_mode_angles.csv",
        semi_krr_mode_angle_rows + full_krr_mode_angle_rows,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if config.plot:
        representative_family = int(family_summary["representative_family_index"])
        representative_s = 0.8
        write_report_figures(
            output_dir=output_dir,
            semi=semi,
            full=full,
            representative_family=representative_family,
            representative_s=representative_s,
            representative_values=(
                semi.endpoint_test
                @ _family_coefficient(semi, representative_family - 1, representative_s)
            ),
            semi_mode_plot=semi_mode_plot,
            full_mode_plot=full_mode_plot,
            semi_endpoint_rows=semi_endpoint_rows,
            semi_family_rows=semi_family_rows,
            family_diagnostics=family_diagnostics,
            full_decompositions=full_decompositions,
        )
    if config.write_report:
        write_ambient_circle_report(
            output_path=config.report_path, run_dir=output_dir, summary=summary
        )
    return summary


__all__ = [
    "AmbientCircleStudyConfig",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_PATH",
    "build_geometry_modes",
    "build_target_data",
    "run_ambient_circle_study",
]
