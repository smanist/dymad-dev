import copy
import os
import random
import shutil
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from dymad.io import TrajectoryManager, TrajectoryManagerGraph
from dymad.training.execution_services import ExecutionServices
from dymad.training.helper import (
    CVResult,
    aggregate_cv_results,
    batch_pattern_search_indices,
    batch_pattern_search_points,
    bounded_nelder_mead_search_points,
    get_by_dotted_key,
    iter_param_grid,
    multi_start_bounded_nelder_mead_search_points,
    nelder_mead_like_search_indices,
    select_best_cv_result,
    set_by_dotted_key,
)
from dymad.training.phase_runtime import PhaseContext, build_initial_trainer_state
from dymad.training.trainer_run import TrainerRun
from dymad.tuning import TuningEvaluation, TuningResult, write_tuning_artifacts
from dymad.utils import load_config, plot_cv_results

TrajectoryManagerLike = TrajectoryManager | TrajectoryManagerGraph


@dataclass
class CVSearchRunResult:
    all_results: list[CVResult]
    selection_combo_indices: list[int] | None = None


@dataclass
class CVSearchBoundSpec:
    key: str
    lower: float
    upper: float
    value_kind: str
    parity: str | None = None


CVSearchHandler = Callable[..., CVSearchRunResult]


def _resolve_cv_base_seed(config: dict[str, Any]) -> int:
    configured_seed = config.get("seed")
    return int(torch.initial_seed() if configured_seed is None else configured_seed)


def _seed_cv_trial(seed: int) -> None:
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------
# Standalone single CV run for multi-processing compatibility
# --------------------
def _apply_combo_to_config(
    combo_idx: int,
    fold_id: int,
    cfg: dict[str, Any],
    combo: dict[str, Any],
    base_name: str,
    checkpoint_prefix: str,
    results_prefix: str,
) -> tuple[dict[str, Any], str]:
    """
    Apply dotted-key hyperparameters in combo onto a deep-copied config.
    """
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in combo.items():
        set_by_dotted_key(cfg, dotted_key, value)
    _suffix = f"_c{combo_idx}_f{fold_id}"
    cfg["model"]["name"] = f"{base_name}{_suffix}"
    cfg.update(
        {
            "path": {
                "checkpoint_prefix": f"{checkpoint_prefix}/{_suffix}",
                "results_prefix": f"{results_prefix}/{_suffix}",
            }
        }
    )
    model_prefix = cfg["path"]["checkpoint_prefix"] + f"/{cfg['model']['name']}"
    return cfg, model_prefix


def _build_phase_context(
    fold_id: int,
    cfg: dict[str, Any],
    train_sets: Sequence[TrajectoryManagerLike],
    valid_sets: Sequence[TrajectoryManagerLike],
) -> PhaseContext:
    """Setup typed phase context (datasets/loaders/metadata) for one fold."""
    trainset = train_sets[fold_id]
    trainset.update_config(cfg)
    train_loader_raw, train_set_raw, train_md = trainset.process_data()

    validset = valid_sets[fold_id]
    validset.update_config(cfg)
    if isinstance(validset, TrajectoryManagerGraph):
        if not isinstance(trainset, TrajectoryManagerGraph):
            raise TypeError("Graph validation manager requires a graph training manager.")
        validset.set_transforms(trajmgr=trainset)
    else:
        validset.set_transforms(trajmgr=cast(TrajectoryManager, trainset))
    valid_loader_raw, valid_set_raw, valid_md = validset.process_data()

    return PhaseContext(
        train_loader=cast(DataLoader[Any], train_loader_raw),
        valid_loader=cast(DataLoader[Any], valid_loader_raw),
        train_set=cast(list[Any], train_set_raw),
        valid_set=cast(list[Any], valid_set_raw),
        train_md=train_md,
        valid_md=valid_md,
    )


def run_cv_single(args: dict[str, Any]):
    _seed_cv_trial(args.get("seed", 0))

    # Apply hyperparameter overrides to this fold's config
    cfg, model_prefix = _apply_combo_to_config(
        args["combo_idx"],
        args["fold_idx"],
        args["fold_cfg"],
        args["combo"],
        args["base_name"],
        args["checkpoint_prefix"],
        args["results_prefix"],
    )

    # Build the typed context for this concrete run.
    phase_context = _build_phase_context(
        args["fold_idx"],
        cfg,
        args["train_sets"],
        args["valid_sets"],
    )
    execution_services = ExecutionServices.from_config(cfg, default_device=args["device"])
    trainer_state = build_initial_trainer_state(
        cfg,
        execution_services=execution_services,
    )

    # Run one concrete trainer run for this fold+combo.
    trainer_run = TrainerRun(
        config=cfg,
        model_class=args["model_class"],
        device=execution_services.device,
        dtype=args["train_sets"][0].dtype,
        run_name=cfg["model"]["name"],
        checkpoint_prefix=execution_services.checkpoint_prefix,
        results_prefix=execution_services.results_prefix,
        execution_services=execution_services,
    )
    results = trainer_run.run(
        initial_context=phase_context,
        initial_state=trainer_state,
    )

    metric_value = results[-1].get_metric(args["metric"])

    return {
        "combo_idx": args["combo_idx"],
        "fold_idx": args["fold_idx"],
        "combo": args["combo"],
        "metric_value": metric_value,
        "model_prefix": model_prefix,
    }


# --------------------
# The main driver of training
# --------------------
class DriverBase:
    """
    Base driver: loops over (parameter combos x folds) and calls the optimizer.
    """

    CV_SEARCH_HANDLERS: dict[str, str] = {
        "grid": "_execute_cv_search_grid",
        "batch_pattern_search": "_execute_cv_search_batch_pattern_search",
        "multi_start_nelder_mead": "_execute_cv_search_multi_start_nelder_mead",
        "nelder_mead_like": "_execute_cv_search_nelder_mead_like",
    }

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        self.train_sets: list[TrajectoryManagerLike] = []
        self.valid_sets: list[TrajectoryManagerLike] = []
        self.base_config = load_config(config_path, config_mod)
        self.base_seed = _resolve_cv_base_seed(self.base_config)
        self.model_class = model_class
        self.execution_services = ExecutionServices.from_driver_config(
            self.base_config,
            config_path=config_path,
            default_device=device,
        )
        self.base_config = self.execution_services.apply_to_config(self.base_config)
        self.execution_services.ensure_artifact_dirs()
        self.device = self.execution_services.device
        self.max_workers = max_workers

        cv_config = self.base_config.get("cv", {})
        self.param_grid = cv_config.get("param_grid", None)  # None = single combo
        self.metric = cv_config.get("metric", "total")
        selection = cv_config.get("selection", {})
        if not isinstance(selection, dict):
            raise TypeError("cv.selection must be a mapping when provided.")
        tie_breakers = selection.get("tie_breakers", ("std_metric", "combo_index"))
        if not isinstance(tie_breakers, (list, tuple)):
            raise TypeError("cv.selection.tie_breakers must be a list or tuple when provided.")
        self.cv_selection_goal = str(selection.get("goal", "minimize"))
        self.cv_selection_tie_breakers = tuple(str(item) for item in tie_breakers)
        search = cv_config.get("search", {})
        if not isinstance(search, dict):
            raise TypeError("cv.search must be a mapping when provided.")
        self.cv_search_mode = str(search.get("mode", "grid"))
        if self.cv_search_mode not in self.CV_SEARCH_HANDLERS:
            raise ValueError(
                f"cv.search.mode must be one of {tuple(self.CV_SEARCH_HANDLERS)} when provided."
            )
        bounds = search.get("bounds")
        if bounds is not None:
            if self.cv_search_mode not in {
                "nelder_mead_like",
                "batch_pattern_search",
                "multi_start_nelder_mead",
            }:
                raise TypeError(
                    "cv.search.bounds is only supported with cv.search.mode='nelder_mead_like' "
                    "'batch_pattern_search', or 'multi_start_nelder_mead'."
                )
            if self.param_grid is not None:
                raise TypeError(
                    "cv.search.bounds cannot be combined with cv.param_grid; choose one search space."
                )
            if not isinstance(bounds, dict) or not bounds:
                raise TypeError("cv.search.bounds must be a non-empty mapping when provided.")
        elif self.cv_search_mode == "multi_start_nelder_mead":
            raise TypeError("cv.search.mode='multi_start_nelder_mead' requires cv.search.bounds.")
        self.cv_search_bounds = cast(dict[str, Any] | None, bounds)

        max_iterations = search.get("max_iterations")
        if max_iterations is not None:
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                raise TypeError(
                    "cv.search.max_iterations must be a positive integer when provided."
                )
        self.cv_search_max_iterations = max_iterations

        reflection = search.get("reflection", 1.0)
        expansion = search.get("expansion", 2.0)
        contraction = search.get("contraction", 0.5)
        shrink = search.get("shrink", 0.5)
        if not isinstance(reflection, (int, float)) or float(reflection) <= 0.0:
            raise TypeError("cv.search.reflection must be a positive number when provided.")
        if not isinstance(expansion, (int, float)) or float(expansion) <= 1.0:
            raise TypeError("cv.search.expansion must be greater than 1 when provided.")
        if not isinstance(contraction, (int, float)) or not (0.0 < float(contraction) < 1.0):
            raise TypeError("cv.search.contraction must be in (0, 1) when provided.")
        if not isinstance(shrink, (int, float)) or not (0.0 < float(shrink) < 1.0):
            raise TypeError("cv.search.shrink must be in (0, 1) when provided.")
        self.cv_search_reflection = float(reflection)
        self.cv_search_expansion = float(expansion)
        self.cv_search_contraction = float(contraction)
        self.cv_search_shrink = float(shrink)

        if self.cv_search_mode == "nelder_mead_like" and self.max_workers > 1:
            warnings.warn(
                "cv.search.mode='nelder_mead_like' is sequential; use "
                "cv.search.mode='batch_pattern_search' to use parallel workers.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.cv_search_mode == "batch_pattern_search" and self.max_workers == 1:
            warnings.warn(
                "cv.search.mode='batch_pattern_search' is intended for max_workers > 1; "
                "with max_workers=1 it runs as sequential batched pattern search.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Setup paths
        self.base_name = self.base_config["model"]["name"]
        self.checkpoint_prefix = self.execution_services.checkpoint_prefix
        self.results_prefix = self.execution_services.results_prefix

        # Setup logging
        self.cv_logger_prefix = (
            ""
            if self.execution_services.log_stdout
            else f"{self.results_prefix}/{self.base_name}_cv"
        )
        self.cv_logger = self.execution_services.configure_logger(
            "dymad.cv",
            prefix=self.cv_logger_prefix,
        )

        # Initialize data sets
        self._init_trajectory_managers()
        self._init_fold_split()
        self.cv_search_bound_specs = self._build_cv_search_bound_specs()

    # --------------------
    # Abstract methods to be implemented by subclasses
    # --------------------

    def _init_trajectory_managers(self):
        """
        Depending on how folds are defined, create TrajectoryManager(s) for data loading.
        """
        raise NotImplementedError

    def _init_fold_split(self):
        """
        Determine how to split data into folds.
        """
        raise NotImplementedError

    def iter_folds(self) -> Iterable[tuple[int, dict[str, Any]]]:
        """
        Yield (fold_id, fold_config) pairs.

        fold_config is a *full* config dict (deep copy of base_config with
        fold-specific overrides, e.g. split_seed).
        """
        raise NotImplementedError

    # --------------------
    # The main training loop
    # --------------------

    def train(self, continue_training: bool = False) -> tuple[int, CVResult, list[CVResult]]:
        """
        Core loop over hyperparameter and folds combinations.

        Returns:
          best_result, all_results
        """
        # Reload previous results if continuing training
        file_name = f"{self.results_prefix}/{self.base_name}_cv.npz"
        prev_all_results, combo_offset = [], 0
        if continue_training:
            self.cv_logger.info(f"Continuing training from existing CV results {file_name}.")
            if os.path.exists(file_name):
                loaded = np.load(file_name, allow_pickle=True)
                assert loaded["metric_name"] == self.metric, (
                    f"Metric mismatch: existing {loaded['metric_name']} vs current {self.metric}"
                )
                prev_all_results = loaded["all_results"].tolist()
                prev_best_result = prev_all_results[loaded["best_idx"]]
                combo_offset = len(prev_all_results)
                self.cv_logger.info(f"Found {combo_offset} previous results.")
                self.cv_logger.info(
                    f"Previous best: {prev_best_result.params} with {self.metric} = {prev_best_result.mean_metric:.4e}"
                )
            else:
                self.cv_logger.info(f"CV results {file_name} not found, starting from scratch.")

        fold_specs = list(self.iter_folds())
        search_result = self._execute_cv_search(
            fold_specs=fold_specs,
            combo_offset=combo_offset,
            continue_training=continue_training,
        )
        all_results = search_result.all_results
        selection_combo_indices = search_result.selection_combo_indices

        all_results = prev_all_results + all_results

        best_idx = select_best_cv_result(
            all_results,
            goal=self.cv_selection_goal,
            tie_breakers=self.cv_selection_tie_breakers,
            combo_indices=selection_combo_indices,
        )
        best_result = all_results[best_idx]
        self.cv_logger.info(
            f"Best combo: {best_result.params} with {self.metric} = {best_result.mean_metric:.4e} "
            f"(selection goal={self.cv_selection_goal}, tie_breakers={self.cv_selection_tie_breakers})"
        )

        # Save CV results
        np.savez_compressed(
            file_name,
            all_results=np.asarray(all_results, dtype=object),
            metric_name=self.metric,
            best_idx=best_idx,
        )
        self.cv_logger.info(f"Saved CV results to {file_name}")
        self._write_cv_tuning_artifacts(
            best_idx=best_idx,
            best_result=best_result,
            all_results=all_results,
            selection_combo_indices=selection_combo_indices,
        )
        plot_cv_results(file_name, ifclose=True, prefix=self.results_prefix)
        self.cv_logger.info(f"Saved CV plot to {self.results_prefix}/cv_results.png")

        # Copy best model checkpoint to a separate file
        best_checkpoint = best_result.checkpoint_paths[0]
        best_model = f"{self.checkpoint_prefix}/{self.base_name}.pt"
        best_summary = f"{self.checkpoint_prefix}/{self.base_name}_summary.npz"
        shutil.copy2(best_checkpoint + ".pt", best_model)
        shutil.copy2(best_checkpoint + "_summary.npz", best_summary)
        self.cv_logger.info(
            f"Copied best model {best_checkpoint} to {best_model} and {best_summary}"
        )

        # Close the logger to flush buffers and release file handles
        for handler in self.cv_logger.handlers[:]:
            handler.close()
            self.cv_logger.removeHandler(handler)

        return best_idx, best_result, all_results

    def _write_cv_tuning_artifacts(
        self,
        *,
        best_idx: int,
        best_result: CVResult,
        all_results: Sequence[CVResult],
        selection_combo_indices: Sequence[int] | None,
    ) -> None:
        evaluations: list[TuningEvaluation] = []
        for index, result in enumerate(all_results):
            combo_index = (
                int(selection_combo_indices[index])
                if selection_combo_indices is not None and index < len(selection_combo_indices)
                else index
            )
            evaluations.append(
                TuningEvaluation(
                    params=dict(result.params),
                    phase=str(self.cv_search_mode),
                    index=combo_index,
                    metric_value=float(result.mean_metric),
                    status="ok",
                    elapsed_seconds=0.0,
                    boundary_hit=self._combo_hits_boundary(result.params),
                    extra_metrics={
                        "std_metric": float(result.std_metric),
                        "fold_metrics": list(result.fold_metrics),
                        "checkpoint_paths": list(result.checkpoint_paths),
                    },
                )
            )
        result = TuningResult(
            selected_params=dict(best_result.params),
            selected_metric=float(best_result.mean_metric),
            evaluations=evaluations,
            failures=[item for item in evaluations if item.boundary_hit],
            candidate_plan={
                "source": "trainer_cv",
                "search_mode": self.cv_search_mode,
                "param_grid": self.param_grid,
                "bounds": self.cv_search_bounds,
                "evaluated_count": len(evaluations),
                "best_idx": best_idx,
            },
            policy={
                "metric_name": self.metric,
                "goal": self.cv_selection_goal,
                "selection_tie_breakers": list(self.cv_selection_tie_breakers),
            },
        )
        write_tuning_artifacts(result, f"{self.results_prefix}/{self.base_name}_tuning")

    def _combo_hits_boundary(self, combo: dict[str, Any]) -> bool:
        for spec in self.cv_search_bound_specs:
            if spec.key not in combo:
                continue
            value = float(combo[spec.key])
            if np.isclose(value, spec.lower) or np.isclose(value, spec.upper):
                return True
        return False

    # --------------------
    # Helper functions
    # --------------------

    def _trial_args_for_combo(
        self,
        *,
        combo_idx: int,
        combo: dict[str, Any],
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        trial_args_list = []
        for fold_idx, fold_cfg in fold_specs:
            trial_args_list.append(
                {
                    "combo_idx": combo_idx,
                    "fold_idx": fold_idx,
                    "fold_cfg": fold_cfg,
                    "combo": combo,
                    "base_name": self.base_name,
                    "checkpoint_prefix": self.checkpoint_prefix,
                    "results_prefix": self.results_prefix,
                    "train_sets": self.train_sets,
                    "valid_sets": self.valid_sets,
                    "model_class": self.model_class,
                    "device": self.device,
                    "metric": self.metric,
                    "seed": self.base_seed + fold_idx,
                }
            )
        return trial_args_list

    def _run_single_combo(
        self,
        *,
        combo_idx: int,
        combo: dict[str, Any],
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
    ) -> CVResult:
        trial_args_list = self._trial_args_for_combo(
            combo_idx=combo_idx,
            combo=combo,
            fold_specs=fold_specs,
        )
        if self.max_workers > 1:
            combo_results = self._parallel_run(trial_args_list)
        else:
            combo_results = self._serial_run(trial_args_list)
        if len(combo_results) != 1:
            raise RuntimeError(
                f"Expected one aggregated CVResult for combo {combo_idx}, got {len(combo_results)}."
            )
        return combo_results[0]

    def _materialize_param_grid_combos(self) -> list[dict[str, Any]]:
        if self.param_grid is None:
            return [{}]
        return list(iter_param_grid(self.param_grid))

    def _run_combos(
        self,
        *,
        combos: Sequence[dict[str, Any]],
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
    ) -> list[CVResult]:
        trial_args_list = []
        for combo_idx, combo in enumerate(combos):
            trial_args_list.extend(
                self._trial_args_for_combo(
                    combo_idx=combo_idx + combo_offset,
                    combo=combo,
                    fold_specs=fold_specs,
                )
            )
        if self.max_workers > 1:
            return self._parallel_run(trial_args_list)
        return self._serial_run(trial_args_list)

    def _run_indexed_combos(
        self,
        *,
        indexed_combos: Sequence[tuple[int, dict[str, Any]]],
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
    ) -> list[CVResult]:
        trial_args_list = []
        for combo_idx, combo in indexed_combos:
            trial_args_list.extend(
                self._trial_args_for_combo(
                    combo_idx=combo_idx,
                    combo=combo,
                    fold_specs=fold_specs,
                )
            )
        raw_results = (
            self._parallel_trial_results(trial_args_list)
            if self.max_workers > 1
            else self._serial_trial_results(trial_args_list)
        )
        rows_by_combo: dict[int, list[dict[str, Any]]] = {
            combo_idx: [] for combo_idx, _combo in indexed_combos
        }
        for row in raw_results:
            rows_by_combo[int(row["combo_idx"])].append(row)
        results: list[CVResult] = []
        for combo_idx, combo in indexed_combos:
            rows = sorted(rows_by_combo[combo_idx], key=lambda row: int(row["fold_idx"]))
            results.append(
                CVResult(
                    params=dict(combo),
                    fold_metrics=[float(row["metric_value"]) for row in rows],
                    checkpoint_paths=[str(row["model_prefix"]) for row in rows],
                )
            )
        return results

    def _execute_cv_search(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
        continue_training: bool,
    ) -> CVSearchRunResult:
        handler_name = self.CV_SEARCH_HANDLERS[self.cv_search_mode]
        handler = cast(
            CVSearchHandler,
            getattr(self, handler_name),
        )
        return handler(
            fold_specs=fold_specs,
            combo_offset=combo_offset,
            continue_training=continue_training,
        )

    def _execute_cv_search_grid(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
        continue_training: bool,
    ) -> CVSearchRunResult:
        del continue_training
        return CVSearchRunResult(
            all_results=self._run_combos(
                combos=self._materialize_param_grid_combos(),
                fold_specs=fold_specs,
                combo_offset=combo_offset,
            )
        )

    def _execute_cv_search_nelder_mead_like(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
        continue_training: bool,
    ) -> CVSearchRunResult:
        if continue_training:
            raise ValueError(
                "continue_training is not supported with cv.search.mode='nelder_mead_like'."
            )
        if self.cv_search_bounds is not None:
            return self._execute_cv_search_nelder_mead_like_bounds(
                fold_specs=fold_specs,
                combo_offset=combo_offset,
            )
        return self._execute_cv_search_nelder_mead_like_param_grid(
            fold_specs=fold_specs,
            combo_offset=combo_offset,
        )

    def _execute_cv_search_batch_pattern_search(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
        continue_training: bool,
    ) -> CVSearchRunResult:
        if continue_training:
            raise ValueError(
                "continue_training is not supported with cv.search.mode='batch_pattern_search'."
            )
        if self.cv_search_bounds is not None:
            return self._execute_cv_search_batch_pattern_search_bounds(
                fold_specs=fold_specs,
                combo_offset=combo_offset,
            )
        return self._execute_cv_search_batch_pattern_search_param_grid(
            fold_specs=fold_specs,
            combo_offset=combo_offset,
        )

    def _execute_cv_search_batch_pattern_search_bounds(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
    ) -> CVSearchRunResult:
        bounded_combo_results: dict[tuple[tuple[str, Any], ...], CVResult] = {}
        ordered_combo_keys: list[tuple[tuple[str, Any], ...]] = []

        def _evaluate_points(points: Sequence[np.ndarray]) -> list[float]:
            indexed_combos: list[tuple[int, dict[str, Any]]] = []
            key_order: list[tuple[tuple[str, Any], ...]] = []
            for point in points:
                combo = self._bounded_search_combo(point)
                combo_key = self._combo_key(combo)
                if combo_key in bounded_combo_results or combo_key in key_order:
                    continue
                combo_idx = combo_offset + len(bounded_combo_results) + len(indexed_combos)
                indexed_combos.append((combo_idx, combo))
                key_order.append(combo_key)
            if indexed_combos:
                results = self._run_indexed_combos(
                    indexed_combos=indexed_combos,
                    fold_specs=fold_specs,
                )
                for combo_key, result in zip(key_order, results, strict=True):
                    bounded_combo_results[combo_key] = result
                    ordered_combo_keys.append(combo_key)
            metrics: list[float] = []
            for point in points:
                combo_key = self._combo_key(self._bounded_search_combo(point))
                metrics.append(bounded_combo_results[combo_key].mean_metric)
            return metrics

        evaluated_points = batch_pattern_search_points(
            lower_bounds=[spec.lower for spec in self.cv_search_bound_specs],
            upper_bounds=[spec.upper for spec in self.cv_search_bound_specs],
            evaluate_points=_evaluate_points,
            goal=self.cv_selection_goal,
            max_evaluations=self.cv_search_max_iterations,
            batch_size=self.max_workers,
            step_shrink=self.cv_search_shrink,
        )
        all_results = [bounded_combo_results[key] for key in ordered_combo_keys]
        self.cv_logger.info(
            "Batch pattern search evaluated %d search points and %d unique parameter "
            "combinations across %d dimensions.",
            len(evaluated_points),
            len(all_results),
            len(self.cv_search_bound_specs),
        )
        return CVSearchRunResult(
            all_results=all_results,
            selection_combo_indices=list(range(combo_offset, combo_offset + len(all_results))),
        )

    def _execute_cv_search_batch_pattern_search_param_grid(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
    ) -> CVSearchRunResult:
        combos = self._materialize_param_grid_combos()
        combo_results: dict[int, CVResult] = {}

        def _evaluate_indices(indices: Sequence[int]) -> list[float]:
            fresh = [index for index in indices if index not in combo_results]
            if fresh:
                indexed_combos = [(index + combo_offset, combos[index]) for index in fresh]
                results = self._run_indexed_combos(
                    indexed_combos=indexed_combos,
                    fold_specs=fold_specs,
                )
                for index, result in zip(fresh, results, strict=True):
                    combo_results[index] = result
            return [combo_results[index].mean_metric for index in indices]

        evaluated_indices = batch_pattern_search_indices(
            combos,
            evaluate_indices=_evaluate_indices,
            goal=self.cv_selection_goal,
            max_evaluations=self.cv_search_max_iterations,
            batch_size=self.max_workers,
            step_shrink=self.cv_search_shrink,
        )
        self.cv_logger.info(
            "Batch pattern search evaluated %d/%d candidates.",
            len(evaluated_indices),
            len(combos),
        )
        return CVSearchRunResult(
            all_results=[combo_results[index] for index in evaluated_indices],
            selection_combo_indices=[index + combo_offset for index in evaluated_indices],
        )

    def _execute_cv_search_nelder_mead_like_bounds(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
    ) -> CVSearchRunResult:
        if len(fold_specs) != 1:
            raise ValueError(
                "cv.search.bounds with cv.search.mode='nelder_mead_like' is only supported "
                "for single-split CV."
            )

        bound_eval_count = 0
        bounded_combo_results: dict[tuple[tuple[str, Any], ...], CVResult] = {}
        ordered_combo_keys: list[tuple[tuple[str, Any], ...]] = []

        def _evaluate_point(point: np.ndarray) -> float:
            nonlocal bound_eval_count
            bound_eval_count += 1
            combo = self._bounded_search_combo(point)
            combo_key = self._combo_key(combo)
            if combo_key not in bounded_combo_results:
                bounded_combo_results[combo_key] = self._run_single_combo(
                    combo_idx=combo_offset + len(bounded_combo_results),
                    combo=combo,
                    fold_specs=fold_specs,
                )
                ordered_combo_keys.append(combo_key)
            return bounded_combo_results[combo_key].mean_metric

        bounded_nelder_mead_search_points(
            lower_bounds=[spec.lower for spec in self.cv_search_bound_specs],
            upper_bounds=[spec.upper for spec in self.cv_search_bound_specs],
            evaluate_point=_evaluate_point,
            goal=self.cv_selection_goal,
            max_iterations=self.cv_search_max_iterations,
            reflection=self.cv_search_reflection,
            expansion=self.cv_search_expansion,
            contraction=self.cv_search_contraction,
            shrink=self.cv_search_shrink,
        )
        all_results = [bounded_combo_results[key] for key in ordered_combo_keys]
        self.cv_logger.info(
            "Bounded Nelder-Mead search evaluated %d search points and %d unique parameter "
            "combinations across %d dimensions.",
            bound_eval_count,
            len(all_results),
            len(self.cv_search_bound_specs),
        )
        return CVSearchRunResult(
            all_results=all_results,
            selection_combo_indices=list(range(combo_offset, combo_offset + len(all_results))),
        )

    def _execute_cv_search_multi_start_nelder_mead(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
        continue_training: bool,
    ) -> CVSearchRunResult:
        if continue_training:
            raise ValueError(
                "continue_training is not supported with cv.search.mode='multi_start_nelder_mead'."
            )
        if self.cv_search_bounds is None:
            raise ValueError("cv.search.mode='multi_start_nelder_mead' requires cv.search.bounds.")
        return self._execute_cv_search_multi_start_nelder_mead_bounds(
            fold_specs=fold_specs,
            combo_offset=combo_offset,
        )

    def _execute_cv_search_multi_start_nelder_mead_bounds(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
    ) -> CVSearchRunResult:
        if len(fold_specs) != 1:
            raise ValueError(
                "cv.search.bounds with cv.search.mode='multi_start_nelder_mead' is only "
                "supported for single-split CV."
            )

        records: list[tuple[int, CVResult]] = []
        records_lock = Lock()
        next_combo_idx = combo_offset

        def _evaluate_point(point: np.ndarray) -> float:
            nonlocal next_combo_idx
            combo = self._bounded_search_combo(point)
            with records_lock:
                combo_idx = next_combo_idx
                next_combo_idx += 1
            results = self._serial_run(
                self._trial_args_for_combo(
                    combo_idx=combo_idx,
                    combo=combo,
                    fold_specs=fold_specs,
                )
            )
            if len(results) != 1:
                raise RuntimeError(
                    f"Expected one aggregated CVResult for combo {combo_idx}, got {len(results)}."
                )
            result = results[0]
            with records_lock:
                records.append((combo_idx, result))
            return result.mean_metric

        evaluated_points = multi_start_bounded_nelder_mead_search_points(
            lower_bounds=[spec.lower for spec in self.cv_search_bound_specs],
            upper_bounds=[spec.upper for spec in self.cv_search_bound_specs],
            evaluate_point=_evaluate_point,
            goal=self.cv_selection_goal,
            max_iterations=self.cv_search_max_iterations,
            num_simplices=self.max_workers,
            max_workers=self.max_workers,
            seed=int(self.base_config.get("seed", 0)),
            reflection=self.cv_search_reflection,
            expansion=self.cv_search_expansion,
            contraction=self.cv_search_contraction,
            shrink=self.cv_search_shrink,
        )
        records = sorted(records, key=lambda item: item[0])
        self.cv_logger.info(
            "Multi-start Nelder-Mead search evaluated %d search points and %d parameter "
            "combinations across %d dimensions using %d Sobol-started simplices.",
            len(evaluated_points),
            len(records),
            len(self.cv_search_bound_specs),
            min(self.max_workers, self.cv_search_max_iterations or self.max_workers),
        )
        return CVSearchRunResult(
            all_results=[result for _combo_idx, result in records],
            selection_combo_indices=[combo_idx for combo_idx, _result in records],
        )

    def _execute_cv_search_nelder_mead_like_param_grid(
        self,
        *,
        fold_specs: Sequence[tuple[int, dict[str, Any]]],
        combo_offset: int,
    ) -> CVSearchRunResult:
        combos = self._materialize_param_grid_combos()
        if len(fold_specs) != 1:
            self.cv_logger.warning(
                "cv.search.mode='nelder_mead_like' is only supported for single-split CV; "
                "falling back to grid search."
            )
            return CVSearchRunResult(
                all_results=self._run_combos(
                    combos=combos,
                    fold_specs=fold_specs,
                    combo_offset=combo_offset,
                )
            )

        combo_results: dict[int, CVResult] = {}

        def _evaluate_combo(index: int) -> float:
            if index not in combo_results:
                combo_results[index] = self._run_single_combo(
                    combo_idx=index + combo_offset,
                    combo=combos[index],
                    fold_specs=fold_specs,
                )
            return combo_results[index].mean_metric

        evaluated_indices = nelder_mead_like_search_indices(
            combos,
            evaluate_index=_evaluate_combo,
            goal=self.cv_selection_goal,
            max_iterations=self.cv_search_max_iterations,
            reflection=self.cv_search_reflection,
            expansion=self.cv_search_expansion,
            contraction=self.cv_search_contraction,
            shrink=self.cv_search_shrink,
        )
        self.cv_logger.info(
            "Nelder-Mead-like single-split search evaluated %d/%d candidates.",
            len(evaluated_indices),
            len(combos),
        )
        return CVSearchRunResult(
            all_results=[combo_results[index] for index in evaluated_indices],
            selection_combo_indices=[index + combo_offset for index in evaluated_indices],
        )

    def _build_cv_search_bound_specs(self) -> list[CVSearchBoundSpec]:
        if self.cv_search_bounds is None:
            return []

        specs: list[CVSearchBoundSpec] = []
        for key, bounds in self.cv_search_bounds.items():
            if not isinstance(key, str) or not key:
                raise TypeError("cv.search.bounds keys must be non-empty dotted config paths.")
            parity: str | None = None
            if isinstance(bounds, dict):
                lower = bounds.get("lower")
                upper = bounds.get("upper")
                parity_value = bounds.get("parity")
                if parity_value is not None:
                    if not isinstance(parity_value, str) or parity_value not in {"odd", "even"}:
                        raise TypeError(
                            f"cv.search.bounds[{key!r}].parity must be 'odd' or 'even'."
                        )
                    parity = parity_value
            elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lower, upper = bounds
            else:
                raise TypeError(
                    f"cv.search.bounds[{key!r}] must be [lower, upper] or a mapping with "
                    "lower/upper and optional parity."
                )
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise TypeError(f"cv.search.bounds[{key!r}] values must be numeric.")
            lower_float = float(lower)
            upper_float = float(upper)
            if lower_float >= upper_float:
                raise TypeError(
                    f"cv.search.bounds[{key!r}] must satisfy lower < upper; got "
                    f"{lower_float} >= {upper_float}."
                )
            try:
                current_value = get_by_dotted_key(self.base_config, key)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise TypeError(
                    f"cv.search.bounds[{key!r}] does not resolve in the config."
                ) from exc
            if isinstance(current_value, bool):
                raise TypeError(
                    f"cv.search.bounds[{key!r}] must target an integer or floating-point config value."
                )
            if isinstance(current_value, (int, np.integer)):
                lower_int = int(np.ceil(lower_float))
                upper_int = int(np.floor(upper_float))
                if parity is not None:
                    lower_int = self._adjust_to_parity(lower_int, parity=parity, direction="up")
                    upper_int = self._adjust_to_parity(upper_int, parity=parity, direction="down")
                if lower_int > upper_int:
                    raise TypeError(
                        f"cv.search.bounds[{key!r}] contains no valid integer values in "
                        f"[{lower_float}, {upper_float}] for parity={parity!r}."
                    )
                specs.append(
                    CVSearchBoundSpec(
                        key=key,
                        lower=float(lower_int),
                        upper=float(upper_int),
                        value_kind="int",
                        parity=parity,
                    )
                )
                continue
            if isinstance(current_value, (float, np.floating)):
                if parity is not None:
                    raise TypeError(
                        f"cv.search.bounds[{key!r}] parity is only supported for integer-valued "
                        "config fields."
                    )
                specs.append(
                    CVSearchBoundSpec(
                        key=key,
                        lower=lower_float,
                        upper=upper_float,
                        value_kind="float",
                    )
                )
                continue
            raise TypeError(
                f"cv.search.bounds[{key!r}] must target an integer or floating-point config value."
            )
        return specs

    def _bounded_search_combo(self, point: np.ndarray) -> dict[str, Any]:
        combo: dict[str, Any] = {}
        for value, spec in zip(
            np.asarray(point, dtype=float),
            self.cv_search_bound_specs,
            strict=False,
        ):
            clipped = float(np.clip(value, spec.lower, spec.upper))
            if spec.value_kind == "int":
                lower_int = int(np.ceil(spec.lower))
                upper_int = int(np.floor(spec.upper))
                candidate = int(np.clip(int(round(clipped)), lower_int, upper_int))
                if spec.parity is not None and candidate % 2 != (1 if spec.parity == "odd" else 0):
                    lower_candidate = candidate - 1
                    upper_candidate = candidate + 1
                    valid_candidates = [
                        value
                        for value in (lower_candidate, upper_candidate)
                        if lower_int <= value <= upper_int
                        and value % 2 == (1 if spec.parity == "odd" else 0)
                    ]
                    if not valid_candidates:
                        raise RuntimeError(
                            f"No valid parity-constrained integer candidates for {spec.key}."
                        )
                    candidate = min(valid_candidates, key=lambda current: abs(current - clipped))
                combo[spec.key] = candidate
            else:
                combo[spec.key] = float(np.clip(clipped, spec.lower, spec.upper))
        return combo

    @staticmethod
    def _adjust_to_parity(value: int, *, parity: str, direction: str) -> int:
        want_mod = 1 if parity == "odd" else 0
        if value % 2 == want_mod:
            return value
        if direction == "up":
            return value + 1
        if direction == "down":
            return value - 1
        raise ValueError(f"Unsupported direction {direction!r}.")

    @staticmethod
    def _combo_key(combo: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple((key, combo[key]) for key in sorted(combo))

    def _create_trajectory_manager(
        self, data_key: str
    ) -> TrajectoryManager | TrajectoryManagerGraph:
        md = {"config": copy.deepcopy(self.base_config)}
        if bool(getattr(self.model_class, "GRAPH", False)):
            tm = TrajectoryManagerGraph(md, data_key=data_key, device=self.device)
        else:
            tm = TrajectoryManager(md, data_key=data_key, device=self.device)
        tm.prepare_data()
        return tm

    def _parallel_run(self, trial_args_list: list[dict[str, Any]]) -> list[CVResult]:
        return aggregate_cv_results(self._parallel_trial_results(trial_args_list))

    def _parallel_trial_results(
        self, trial_args_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(run_cv_single, args) for args in trial_args_list]

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)

                self.cv_logger.info(
                    f"Combo {res['combo_idx']}, fold {res['fold_idx']}: "
                    f"{self.metric} = {res['metric_value']:.4e} "
                    f"Params {res['combo']}"
                )
        return results

    def _serial_run(self, trial_args_list: list[dict[str, Any]]) -> list[CVResult]:
        return aggregate_cv_results(self._serial_trial_results(trial_args_list))

    def _serial_trial_results(self, trial_args_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for args in trial_args_list:
            res = run_cv_single(args)
            results.append(res)

            self.cv_logger.info(
                f"Combo {res['combo_idx']}, fold {res['fold_idx']}: "
                f"{self.metric} = {res['metric_value']:.4e} "
                f"Params {res['combo']}"
            )
        return results


class KFoldDriver(DriverBase):
    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        k_folds: int = 5,
        base_seed: int = 123,
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )
        self.k_folds = k_folds
        self.base_seed = base_seed

        raise NotImplementedError("KFoldDriver is not implemented yet.")

    def iter_folds(self):
        """
        For fold i, set data.split_seed = base_seed + i and yield the config.
        """
        for i in range(self.k_folds):
            fold_cfg = copy.deepcopy(self.base_config)
            split_seed = self.base_seed + i
            set_by_dotted_key(fold_cfg, "data.split_seed", split_seed)
            yield i, fold_cfg


class SingleSplitDriver(DriverBase):
    """
    Single fixed split; can still scan param_grid.

    Extreme case

      - schedule has only one phase,
      - param_grid empty or singleton,

    Just "one trainer of one phase."
    """

    def __init__(
        self,
        config_path: str,
        model_class: type[torch.nn.Module],
        config_mod: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers,
        )

    def iter_folds(self):
        # Just one “fold 0” with the base config (or enforce a specific split_seed)
        fold_cfg = copy.deepcopy(self.base_config)
        if "split_seed" not in fold_cfg.get("data", {}):
            set_by_dotted_key(fold_cfg, "data.split_seed", 0)
        yield 0, fold_cfg

    def _init_trajectory_managers(self):
        assert "data" in self.base_config, "Config must contain 'data' section."
        if "data_valid" in self.base_config:
            # A separate validation dataset is specified
            # This is necessary esp when valid set format is different from train set
            self.train_sets = [self._create_trajectory_manager(data_key="train")]
            self.valid_sets = [self._create_trajectory_manager(data_key="valid")]
        else:
            # The same dataset is used for training and validation
            # We will adjust later
            self.train_sets = [self._create_trajectory_manager(data_key="train")]
            self.valid_sets = [self._create_trajectory_manager(data_key="train")]

    def _init_fold_split(self):
        """
        Split the dataset into training and validation sets, if not done.

        The training fraction is specified in the YAML config (default 0.75).
        The split is performed by shuffling whole trajectories.
        """
        data_cfg = self.base_config.setdefault("data", {})
        split_seed = data_cfg.get("split_seed")
        if split_seed is None:
            split_seed = 0
        if not isinstance(split_seed, (int, np.integer)):
            raise TypeError("data.split_seed must be an integer when provided.")
        split_seed = int(split_seed)
        data_cfg["split_seed"] = split_seed

        if "data_valid" in self.base_config:
            # A separate validation dataset is specified
            # No need to split
            self.train_set_index = torch.arange(self.train_sets[0].metadata["n_samples"])
            self.valid_set_index = torch.arange(self.valid_sets[0].metadata["n_samples"])
            self.train_sets[0].set_data_index(self.train_set_index)
            self.valid_sets[0].set_data_index(self.valid_set_index)
            return

        # Otherwise, split the training dataset into train/valid
        split_cfg = self.base_config.get("split", {})
        train_frac = split_cfg.get("train_frac", 0.75)
        n_samples = self.train_sets[0].metadata["n_samples"]
        if train_frac >= 1.0:
            n_train = n_samples
            n_val = n_samples
            self.train_set_index = torch.arange(n_samples)
            self.valid_set_index = torch.arange(n_samples)
        else:
            n_train = int(n_samples * train_frac)
            n_val = n_samples - n_train
            generator = torch.Generator()
            generator.manual_seed(split_seed)
            perm = torch.randperm(n_samples, generator=generator)
            self.train_set_index = perm[:n_train]
            self.valid_set_index = perm[n_train:]
        assert n_train > 0, f"Training set must have at least one sample. Got {n_train}."
        assert n_val > 0, f"Validation set must have at least one sample. Got {n_val}."

        self.train_sets[0].set_data_index(self.train_set_index)
        self.valid_sets[0].set_data_index(self.valid_set_index)
