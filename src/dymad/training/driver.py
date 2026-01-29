from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import logging
import numpy as np
import os
import shutil
import torch
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

from dymad.io import TrajectoryManager, TrajectoryManagerGraph
from dymad.training.helper import aggregate_cv_results, CVResult, iter_param_grid, RunState, set_by_dotted_key
from dymad.training.stacked_opt import StackedOpt
from dymad.utils import config_logger, load_config

# --------------------
# Standalone single CV run for multi-processing compatibility
# --------------------
def _apply_combo_to_config(
        combo_idx, fold_id: int, cfg: Dict[str, Any], combo: Dict[str, Any],
        base_name, checkpoint_prefix, results_prefix) -> Dict[str, Any]:
    """
    Apply dotted-key hyperparameters in combo onto a deep-copied config.
    """
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in combo.items():
        set_by_dotted_key(cfg, dotted_key, value)
    _suffix = f"_c{combo_idx}_f{fold_id}"
    cfg["model"]["name"] = f"{base_name}{_suffix}"
    cfg.update({
        "path" : {
            "checkpoint_prefix": f"{checkpoint_prefix}/{_suffix}",
            "results_prefix": f"{results_prefix}/{_suffix}"
        }})
    model_prefix = cfg["path"]["checkpoint_prefix"] + f"/{cfg['model']['name']}"
    return cfg, model_prefix

def _build_data_state(fold_id: int, cfg: Dict[str, Any], train_sets, valid_sets, device) -> RunState:
    """Setup data loaders and datasets."""
    # Config can contain different data transforms
    # So we need to update the datasets accordingly
    # The data transforms in valid set should be determined by train set
    trainset: TrajectoryManager | TrajectoryManagerGraph = train_sets[fold_id]
    trainset.update_config(cfg)
    train_loader, train_set, train_md = trainset.process_data()

    validset: TrajectoryManager | TrajectoryManagerGraph = valid_sets[fold_id]
    validset.update_config(cfg)
    validset.set_transforms(trajmgr=trainset)
    valid_loader, valid_set, valid_md = validset.process_data()

    return RunState(
        config=cfg,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        train_set=train_set,
        valid_set=valid_set,
        train_md=train_md,
        valid_md=valid_md,
    )

def run_cv_single(args: Dict[str, Any]):
    # Apply hyperparameter overrides to this fold's config
    cfg, model_prefix = _apply_combo_to_config(
        args['combo_idx'],
        args['fold_idx'],
        args['fold_cfg'],
        args['combo'],
        args['base_name'],
        args['checkpoint_prefix'],
        args['results_prefix'])

    # Build data-only RunState per fold+combo
    data_state = _build_data_state(
        args['fold_idx'],
        cfg,
        args['train_sets'],
        args['valid_sets'],
        args['device'])

    # Run the optimizer with this config and data state
    opt = StackedOpt(
        config=cfg,
        model_class=args['model_class'],
        device=args['device'],
        dtype=args['train_sets'][0].dtype,
    )
    results = opt.run(initial_state=data_state)

    metric_value = results[-1].run_state.get_metric(args['metric'])

    return {
        'combo_idx': args['combo_idx'],
        'fold_idx': args['fold_idx'],
        'combo': args['combo'],
        'metric_value': metric_value,
        'model_prefix': model_prefix}


# --------------------
# The main driver of training
# --------------------
class DriverBase:
    """
    Base driver: loops over (parameter combos x folds) and calls the optimizer.
    """

    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        self.base_config = load_config(config_path, config_mod)
        self.model_class = model_class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = max_workers

        cv_config = self.base_config.get("cv", {})
        self.param_grid = cv_config.get("param_grid", None)   # None = single combo
        self.metric = cv_config.get("metric", "total")

        # Setup paths
        self.base_name = self.base_config['model']['name']
        _dir = os.path.dirname(config_path)
        _dir = '.' if _dir == '' else _dir
        os.makedirs(f'{_dir}/{self.base_name}', exist_ok=True)
        self.checkpoint_prefix = f'{_dir}/{self.base_name}'
        self.results_prefix = f'{_dir}/{self.base_name}'

        # Setup logging
        log_config = self.base_config.get("log", {})
        ifstdout = log_config.get("stdout", False)
        self.cv_logger = logging.getLogger("dymad.cv")
        self.cv_logger_level = log_config.get("level", "info")
        self.cv_logger_prefix = '' if ifstdout else f"{self.results_prefix}/{self.base_name}_cv"

        # Logger created here so that the initialization process is logged too
        config_logger(
            self.cv_logger,
            mode=self.cv_logger_level,
            prefix=self.cv_logger_prefix)

        # Initialize data sets
        self._init_trajectory_managers()
        self._init_fold_split()

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

    def iter_folds(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """
        Yield (fold_id, fold_config) pairs.

        fold_config is a *full* config dict (deep copy of base_config with
        fold-specific overrides, e.g. split_seed).
        """
        raise NotImplementedError

    # --------------------
    # The main training loop
    # --------------------

    def train(self, continue_training: bool = False) -> Tuple[int, CVResult, List[CVResult]]:
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
                assert loaded["metric_name"] == self.metric, \
                    f"Metric mismatch: existing {loaded['metric_name']} vs current {self.metric}"
                prev_all_results = loaded['all_results'].tolist()
                prev_best_result = prev_all_results[loaded['best_idx']]
                combo_offset = len(prev_all_results)
                self.cv_logger.info(f"Found {combo_offset} previous results.")
                self.cv_logger.info(f"Previous best: {prev_best_result.params} with {self.metric} = {prev_best_result.mean_metric:.4e}")
            else:
                self.cv_logger.info(f"CV results {file_name} not found, starting from scratch.")

        # empty grid => treat as single combo with no overrides
        if self.param_grid is None:
            combos = [ {} ]
        else:
            combos = list(iter_param_grid(self.param_grid))

        trial_args_list = []
        for combo_idx, combo in enumerate(combos):
            for fold_idx, fold_cfg in self.iter_folds():
                args = {
                    'combo_idx': combo_idx + combo_offset,
                    'fold_idx': fold_idx,
                    'fold_cfg': fold_cfg,
                    'combo': combo,
                    'base_name': self.base_name,
                    'checkpoint_prefix': self.checkpoint_prefix,
                    'results_prefix': self.results_prefix,
                    'train_sets': self.train_sets,
                    'valid_sets': self.valid_sets,
                    'model_class': self.model_class,
                    'device': self.device,
                    'metric': self.metric,
                }
                trial_args_list.append(args)

        if self.max_workers > 1:
            all_results = self._parallel_run(trial_args_list)
        else:
            all_results = self._serial_run(trial_args_list)
        all_results = prev_all_results + all_results

        # Select best, assuming lower is better
        best_idx = int(np.argmin([r.mean_metric for r in all_results]))
        best_result = all_results[best_idx]
        self.cv_logger.info(f"Best combo: {best_result.params} with {self.metric} = {best_result.mean_metric:.4e}")

        # Save CV results
        np.savez_compressed(file_name, all_results=all_results, metric_name=self.metric, best_idx=best_idx)
        self.cv_logger.info(f"Saved CV results to {file_name}")

        # Copy best model checkpoint to a separate file
        best_checkpoint = best_result.checkpoint_paths[0]
        best_model = f"{self.checkpoint_prefix}/{self.base_name}.pt"
        best_summary = f"{self.checkpoint_prefix}/{self.base_name}_summary.npz"
        shutil.copy2(best_checkpoint + '.pt', best_model)
        shutil.copy2(best_checkpoint + '_summary.npz', best_summary)
        self.cv_logger.info(f"Copied best model {best_checkpoint} to {best_model} and {best_summary}")

        # Close the logger to flush buffers and release file handles
        for handler in self.cv_logger.handlers[:]:
            handler.close()
            self.cv_logger.removeHandler(handler)

        return best_idx, best_result, all_results

    # --------------------
    # Helper functions
    # --------------------

    def _create_trajectory_manager(self, data_key: str) -> Union[TrajectoryManager, TrajectoryManagerGraph]:
        md = {'config': copy.deepcopy(self.base_config)}
        if self.model_class.GRAPH:
            tm = TrajectoryManagerGraph(md, data_key=data_key, device=self.device)
        else:
            tm = TrajectoryManager(md, data_key=data_key, device=self.device)
        tm.prepare_data()
        return tm

    def _parallel_run(self, trial_args_list: List[Dict[str, Any]]) -> List[CVResult]:
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [
                ex.submit(run_cv_single, args)
                for args in trial_args_list
            ]

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)

                self.cv_logger.info(
                    f"Combo {res['combo_idx']}, fold {res['fold_idx']}: "
                    f"{self.metric} = {res['metric_value']:.4e} "
                    f"Params {res['combo']}")
        all_results = aggregate_cv_results(results)
        return all_results

    def _serial_run(self, trial_args_list: List[Dict[str, Any]]) -> List[CVResult]:
        results = []
        for args in trial_args_list:
            res = run_cv_single(args)
            results.append(res)

            self.cv_logger.info(
                f"Combo {res['combo_idx']}, fold {res['fold_idx']}: "
                f"{self.metric} = {res['metric_value']:.4e} "
                f"Params {res['combo']}")
        all_results = aggregate_cv_results(results)
        return all_results


class KFoldDriver(DriverBase):
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        k_folds: int = 5,
        base_seed: int = 123,
        config_mod: Dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers
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
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_workers: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
            max_workers=max_workers
        )

    def iter_folds(self):
        # Just one “fold 0” with the base config (or enforce a specific split_seed)
        fold_cfg = copy.deepcopy(self.base_config)
        if "split_seed" not in fold_cfg.get("data", {}):
            set_by_dotted_key(fold_cfg, "data.split_seed", 0)
        yield 0, fold_cfg

    def _init_trajectory_managers(self):
        assert 'data' in self.base_config, "Config must contain 'data' section."
        if 'data_valid' in self.base_config:
            # A separate validation dataset is specified
            # This is necessary esp when valid set format is different from train set
            self.train_sets = [self._create_trajectory_manager(data_key='train')]
            self.valid_sets = [self._create_trajectory_manager(data_key='valid')]
        else:
            # The same dataset is used for training and validation
            # We will adjust later
            self.train_sets = [self._create_trajectory_manager(data_key='train')]
            self.valid_sets = [self._create_trajectory_manager(data_key='train')]

    def _init_fold_split(self):
        """
        Split the dataset into training and validation sets, if not done.

        The training fraction is specified in the YAML config (default 0.75).
        The split is performed by shuffling whole trajectories.
        """
        if 'data_valid' in self.base_config:
            # A separate validation dataset is specified
            # No need to split
            self.train_set_index = torch.arange(self.train_sets[0].metadata['n_samples'])
            self.valid_set_index = torch.arange(self.valid_sets[0].metadata['n_samples'])
            self.train_sets[0].set_data_index(self.train_set_index)
            self.valid_sets[0].set_data_index(self.valid_set_index)
            return

        # Otherwise, split the training dataset into train/valid
        split_cfg = self.base_config.get("split", {})
        train_frac = split_cfg.get("train_frac", 0.75)
        n_samples = self.train_sets[0].metadata['n_samples']
        if train_frac >= 1.0:
            n_train = n_samples
            n_val = n_samples
            self.train_set_index = torch.arange(n_samples)
            self.valid_set_index = torch.arange(n_samples)
        else:
            n_train = int(n_samples * train_frac)
            n_val = n_samples - n_train
            perm = torch.randperm(n_samples)
            self.train_set_index = perm[:n_train]
            self.valid_set_index = perm[n_train:]
        assert n_train > 0, f"Training set must have at least one sample. Got {n_train}."
        assert n_val > 0, f"Validation set must have at least one sample. Got {n_val}."

        self.train_sets[0].set_data_index(self.train_set_index)
        self.valid_sets[0].set_data_index(self.valid_set_index)
